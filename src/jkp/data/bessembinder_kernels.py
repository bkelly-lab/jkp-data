"""
Numba kernels for the Bessembinder Section 6 corrections and Section 8 filters.

Description:
    Per-security loops over date-sorted array slices, parallelized with prange
    over group boundaries; kernels mutate preallocated arrays in place.
Steps:
    1) detect_single_period_all — single-period spike-and-reversal (window 1).
    2) detect_multi_period_all — FULL/SUB_A/SUB_B window passes per nlag.
    3) validate_cascading_all — both-endpoints-flagged rejection.
    4) section8_all — the 8a-8h filter chain, one pass per security.
Output:
    factor = correction multiplier (1.0 = clean), wsize = detection window
    (-1 = null), ep_l/ep_r = clean endpoints (interpolation); Section 8 marks
    removed rows in `reason` (0 = kept, 1 = removed).

Equivalence notes (load-bearing — do not "simplify"):
    - Out-of-range endpoint factor counts as clean (1.0) but its VALUE is NaN,
      so the reversal ratio fails and no detection fires.
    - Interior variation max/min skip NaN cells; an all-NaN interior kills it.
    - Propagated rows carry the detection point's endpoints and window_size=nlag,
      claimable only while their factor is still 1.0.
    - Multi-period classes magnitudes with the same nested 500/50/5 chain as
      single-period (deliberate divergence from the original dead-code loop;
      commit 507df7c).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True, error_model="numpy")
def _magnitude_factor(rl, rr):
    """
    Correction factor for a spike whose ratios to both endpoints are rl, rr.
    Nested first-match order (power-of-10 magnitude); 1.0 means no error.
    """
    if rl > 500.0 and rr > 500.0:
        return 0.001
    if rl > 50.0 and rr > 50.0:
        return 0.01
    if rl > 5.0 and rr > 5.0:
        return 0.1
    if rl < 0.002 and rr < 0.002:
        return 1000.0
    if rl < 0.02 and rr < 0.02:
        return 100.0
    if rl < 0.2 and rr < 0.2:
        return 10.0
    return 1.0


@njit(cache=True, error_model="numpy")
def _detect_single_one(x, factor, ep_l, ep_r):
    """Single-period detection for one security slice (arrays are views)."""
    n = len(x)
    for t in range(1, n - 1):
        xt = x[t]
        xp = x[t - 1]
        xn = x[t + 1]
        if np.isnan(xt) or np.isnan(xp) or np.isnan(xn):
            continue
        f = _magnitude_factor(xt / xp, xt / xn)
        if f == 1.0:
            continue
        factor[t] = f
        ep_l[t] = xp
        ep_r[t] = xn


@njit(parallel=True, cache=True, error_model="numpy")
def detect_single_period_all(x, starts, factor, ep_l, ep_r):
    """Run single-period detection over all securities in parallel."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        _detect_single_one(x[s:e], factor[s:e], ep_l[s:e], ep_r[s:e])


@njit(cache=True, error_model="numpy")
def _multi_pass_one(
    x, factor, wsize, ep_l, ep_r, nlag, ep_lag, ep_lead, ilo, ihi, vthr, det, det_epl, det_epr
):
    """
    One window pass (FULL/SUB_A/SUB_B) for one security. Phase 1 detects into
    the det* scratch arrays against the pass-start factor state; phase 2
    propagates first-write-wins in ascending offset order onto still-1.0 rows.
    """
    n = len(x)
    for k in range(n):
        det[k] = np.nan

    # Phase 1: detection
    for t in range(n):
        if factor[t] != 1.0:
            continue
        il = t - ep_lag
        ir = t + ep_lead
        # Out-of-range endpoint: clean guard sees 1.0, value is NaN
        fl = factor[il] if 0 <= il < n else 1.0
        fr = factor[ir] if 0 <= ir < n else 1.0
        if fl != 1.0 or fr != 1.0:
            continue
        xt = x[t]
        if np.isnan(xt):
            continue
        xl = x[il] if 0 <= il < n else np.nan
        xr = x[ir] if 0 <= ir < n else np.nan
        rl = xt / xl
        rr = xt / xr
        high = rl > 5.0 and rr > 5.0
        low = rl < 0.2 and rr < 0.2
        if not (high or low):
            continue
        # Interior variation: skip NaN/out-of-range cells; all-NaN -> no detect
        imax = -np.inf
        imin = np.inf
        count = 0
        for off in range(ilo, ihi + 1):
            j = t + off
            if j < 0 or j >= n:
                continue
            v = x[j]
            if np.isnan(v):
                continue
            count += 1
            if v > imax:
                imax = v
            if v < imin:
                imin = v
        if count == 0:
            continue
        if not (imax / imin < vthr):
            continue
        # high/low gate above guarantees a non-1.0 magnitude here
        det[t] = _magnitude_factor(rl, rr)
        det_epl[t] = xl
        det_epr[t] = xr

    # Phase 2: propagation, first-write-wins in ascending offset order
    for off in range(ilo, ihi + 1):
        for r in range(n):
            t = r - off  # shift(off) moves value from row r-off to row r
            if t < 0 or t >= n:
                continue
            if np.isnan(det[t]) or factor[r] != 1.0:
                continue
            factor[r] = det[t]
            wsize[r] = nlag
            ep_l[r] = det_epl[t]
            ep_r[r] = det_epr[t]


@njit(cache=True, error_model="numpy")
def _multi_one(x, factor, wsize, ep_l, ep_r, nlags, vthr):
    """All multi-period passes for one security: nlags ascending, FULL -> SUB_A -> SUB_B."""
    n = len(x)
    det = np.empty(n, dtype=np.float64)
    det_epl = np.empty(n, dtype=np.float64)
    det_epr = np.empty(n, dtype=np.float64)
    for i in range(len(nlags)):
        nlag = nlags[i]
        if nlag <= 1:
            continue
        # Three window passes, each (ep_lag, ep_lead, ilo, ihi):
        #   FULL : endpoints t-nlag  / t+nlag   , interior -(nlag-1)..(nlag-1)
        #   SUB_A: endpoints t-nlag  / t+nlag-1 , interior -(nlag-1)..(nlag-2)
        #   SUB_B: endpoints t-nlag+1/ t+nlag   , interior -(nlag-2)..(nlag-1)
        for ep_lag, ep_lead, ilo, ihi in (
            (nlag, nlag, -(nlag - 1), nlag - 1),
            (nlag, nlag - 1, -(nlag - 1), nlag - 2),
            (nlag - 1, nlag, -(nlag - 2), nlag - 1),
        ):
            _multi_pass_one(
                x,
                factor,
                wsize,
                ep_l,
                ep_r,
                nlag,
                ep_lag,
                ep_lead,
                ilo,
                ihi,
                vthr,
                det,
                det_epl,
                det_epr,
            )


@njit(parallel=True, cache=True, error_model="numpy")
def detect_multi_period_all(x, starts, factor, wsize, ep_l, ep_r, nlags, vthr):
    """Run multi-period detection over all securities in parallel."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        _multi_one(x[s:e], factor[s:e], wsize[s:e], ep_l[s:e], ep_r[s:e], nlags, vthr)


@njit(cache=True, error_model="numpy")
def _validate_one(factor, wsize, reject):
    """
    Cascading validation for one security: reject corrections whose BOTH
    endpoint positions (p +/- window_size, null -> 1) are themselves flagged.
    Rejections applied simultaneously per round, max 10 rounds.
    """
    n = len(factor)
    for _ in range(10):
        any_reject = False
        for p in range(n):
            reject[p] = False
        for p in range(n):
            if factor[p] == 1.0:
                continue
            w = wsize[p] if wsize[p] != -1 else 1  # -1 = null window size
            lp = p - w
            rp = p + w
            lf = 0 <= lp < n and factor[lp] != 1.0
            rf = 0 <= rp < n and factor[rp] != 1.0
            if lf and rf:
                reject[p] = True
                any_reject = True
        if not any_reject:
            break
        for p in range(n):
            if reject[p]:
                factor[p] = 1.0


@njit(parallel=True, cache=True, error_model="numpy")
def validate_cascading_all(starts, factor, wsize):
    """Run cascading validation over all securities (factor reset in place)."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        scratch = np.empty(e - s, dtype=np.bool_)
        _validate_one(factor[s:e], wsize[s:e], scratch)


# Section 8 filter kernels (thresholds = Bessembinder et al. 2023 Data Appendix)


@njit(cache=True, error_model="numpy")
def _s8_kill_all(reason):
    """Remove every still-alive row of the security."""
    for i in range(reason.size):
        if reason[i] == 0:
            reason[i] = 1


@njit(cache=True, error_model="numpy")
def _s8_early_jump_stage(reason, jump_series, confirm_series, return_based, chn):
    """
    Shared early-jump stage for filters 8e (adjCSHO) and 8f (ME). A jump in
    jump_series (confirmed by confirm_series) within the early period deletes
    alive rows 0..max-jump-obs. 8f (return_based) reads confirm_series as a
    return; 8e uses looser CHN bounds. NaN operands fail all tests (like polars).
    """
    n = reason.size
    total = 0
    for i in range(n):
        if reason[i] == 0:
            total += 1
    if total == 0:
        return
    delete_through = -1
    obs = -1
    prev = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        obs += 1
        if prev >= 0:
            jump = jump_series[i] / jump_series[prev]
            if return_based:
                # filter 8f: ME jump not backed by a commensurate return
                ret = confirm_series[i] / confirm_series[prev] - 1.0
                up = jump > 10.0 and ret < 2.0
                down = jump < 0.1 and ret > -0.5
            else:
                # filter 8e: adjCSHO jump vs ME (CHN gets looser bounds)
                confirm = confirm_series[i] / confirm_series[prev]
                if chn[i]:
                    up = jump >= 50.0 and confirm >= 25.0
                else:
                    up = jump >= 5.0 and confirm >= 2.5
                down = jump <= 0.2 and confirm <= 0.4
            # early period: first ~24 months (obs < 504) or 20% of the history
            in_early_period = obs < 504.0 or obs < 0.2 * total
            if (up or down) and in_early_period:
                delete_through = obs
        prev = i
    if delete_through < 0:
        return
    obs = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        obs += 1
        if obs <= delete_through:
            reason[i] = 1


@njit(cache=True, error_model="numpy")
def _s8_one_security(reason, remove_8a, ajexdi, prc, me, ri, cshoc, dates, low_thr, chn, gap_days):
    """
    Run the Section 8 filter chain on one security's date-sorted slice. Filters
    run sequentially, each deriving its shifts/obs-counts over the rows still
    alive at its start (matching the polars chain). reason is mutated in place
    (0 = kept, 1 = removed); gap_days is filter 8d's calendar-day bound.
    """
    n = reason.size

    # 8a: bottom-percentile average positive volume (global decision)
    if remove_8a:
        _s8_kill_all(reason)
        return

    # 8b: any AJEXDI == 0 removes the security
    for i in range(n):
        if ajexdi[i] == 0.0:
            _s8_kill_all(reason)
            return

    # 8c: low price / market equity
    first = -1
    for i in range(n):
        if reason[i] == 0:
            first = i
            break
    if first < 0:
        return
    breach_idx = -1
    for i in range(first, n):
        if reason[i] != 0:
            continue
        thr = 0.001 if low_thr[i] else 0.01
        if me[i] < 1.0 or prc[i] < thr:
            breach_idx = i
            break
    if breach_idx >= 0:
        # Initial breach (any alive row sharing the first alive date) removes
        # the whole security; otherwise remove history from the breach date on
        initial = False
        for i in range(first, n):
            if reason[i] != 0 or dates[i] != dates[first]:
                break
            thr = 0.001 if low_thr[i] else 0.01
            if me[i] < 1.0 or prc[i] < thr:
                initial = True
                break
        if initial:
            _s8_kill_all(reason)
            return
        for i in range(n):
            if reason[i] == 0 and dates[i] >= dates[breach_idx]:
                reason[i] = 1

    # 8d: drop observations after calendar gaps > ~11 months
    prev = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        if prev >= 0 and dates[i] - dates[prev] > gap_days:
            reason[i] = 1
            # polars computes all gaps against the pre-stage frame, so the
            # removed row still serves as the next row's gap reference
        prev = i

    # 8e: adjCSHO jumps in early history
    adjcsho = cshoc * ajexdi
    _s8_early_jump_stage(reason, adjcsho, me, False, chn)

    # 8f: ME jumps without commensurate returns in early history
    _s8_early_jump_stage(reason, me, ri, True, chn)

    # 8g: returns inconsistent with ME changes
    prev = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        if prev >= 0:
            ret = ri[i] / ri[prev] - 1.0
            me_chg = me[i] / me[prev] - 1.0
            # |return| > 80% not backed by a matching (>50%) ME move
            if abs(ret) > 0.8 and abs(me_chg) < 0.5:
                # prev still advances below: the flagged row stays the shift
                # source for the next row, matching polars stage-input shifts
                reason[i] = 1
        prev = i

    # 8h: large price/ME ratios within the first few observations
    prev = -1
    obs = -1
    for i in range(n):
        if reason[i] != 0:
            continue
        obs += 1
        if obs > 2.0:  # only the first 3 observations (0-indexed)
            break
        if prev >= 0:
            pr = prc[i] / prc[prev]
            mr = me[i] / me[prev]
            # price or ME ratio outside [0.1, 10] on an initial observation
            if pr > 10.0 or pr < 0.1 or mr > 10.0 or mr < 0.1:
                reason[i] = 1  # prev still advances: see 8g note
        prev = i


@njit(parallel=True, cache=True, error_model="numpy")
def section8_all(
    starts, reason, remove_8a, ajexdi, prc, me, ri, cshoc, dates, low_thr, chn, gap_days
):
    """Run the Section 8 filter chain over all securities in parallel."""
    for g in prange(len(starts) - 1):
        s, e = starts[g], starts[g + 1]
        _s8_one_security(
            reason[s:e],
            remove_8a[g],
            ajexdi[s:e],
            prc[s:e],
            me[s:e],
            ri[s:e],
            cshoc[s:e],
            dates[s:e],
            low_thr[s:e],
            chn[s:e],
            gap_days,
        )
