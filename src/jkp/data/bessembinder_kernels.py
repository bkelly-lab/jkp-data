"""
Numba kernels for Bessembinder Section 6 decimal-error detection.

Description:
    Per-security loop kernels that replicate, byte-for-byte, the original
    Polars-expression implementation (see _bessembinder_polars_reference.py).
    Detection is strictly per security, so kernels parallelize with prange
    over group slices of date-sorted contiguous arrays.
Steps:
    1) detect_single_period_all: single-period spike-and-reversal (window 1).
    2) detect_multi_period_all: FULL / SUB_A / SUB_B window passes per nlag,
       two-phase (detect on start-of-pass factor state, then propagate
       first-write-wins in ascending offset order).
    3) validate_cascading_all: iterative both-endpoints-flagged rejection.
Output:
    Kernels mutate preallocated output arrays in place (factor, error-type
    codes, window size/type codes, debug columns). Null sentinels: NaN for
    floats, -1 for int8 codes and int32 window sizes.

Equivalence notes (load-bearing — do not "simplify"):
    - Out-of-range ENDPOINT factor counts as clean (fill_null(1.0)), but its
      VALUE is NaN, so the reversal ratio fails and no detection fires.
    - Interior variation max/min SKIP NaN cells (polars max/min_horizontal
      ignore nulls); only an all-NaN interior kills detection.
    - Propagated rows carry the DETECTION POINT's endpoint/ratio/variation
      values, window_size = nlag, and the pass's window-type code. A target
      row is claimable only while its factor is still 1.0.
    - Multi-period classes magnitudes with the nested 500/50/5 chain (like
      single-period). DELIBERATE DIVERGENCE from the original: its magnitude
      [1, 2, 3] loop was dead code beyond 10x (proven in commit 507df7c), so
      multi-day 100x/1000x errors were under-corrected by 10x. Everything
      else remains byte-identical to the original.
"""

import numpy as np
from numba import njit, prange

# Error-type codes (string mapping lives in bessembinder.py)
ET_NULL = -1
ET_HIGH_10X = 0
ET_HIGH_100X = 1
ET_HIGH_1000X = 2
ET_LOW_10X = 3
ET_LOW_100X = 4
ET_LOW_1000X = 5

# Window-type codes
WT_NULL = -1
WT_SINGLE = 0
WT_FULL = 1
WT_SUB_A = 2
WT_SUB_B = 3

# Sentinel for null window size (Int32)
WS_NULL = -1


@njit(cache=True, error_model="numpy")
def _detect_single_one(x, factor, etype, wtype, ep_l, ep_r, rat_l, rat_r):
    """Single-period detection for one security slice (arrays are views)."""
    n = len(x)
    for t in range(1, n - 1):
        xt = x[t]
        xp = x[t - 1]
        xn = x[t + 1]
        if np.isnan(xt) or np.isnan(xp) or np.isnan(xn):
            continue
        rp = xt / xp
        rn = xt / xn
        # Nested first-match order identical to the original when-chain
        if rp > 500.0 and rn > 500.0:
            f, c = 0.001, ET_HIGH_1000X
        elif rp > 50.0 and rn > 50.0:
            f, c = 0.01, ET_HIGH_100X
        elif rp > 5.0 and rn > 5.0:
            f, c = 0.1, ET_HIGH_10X
        elif rp < 0.002 and rn < 0.002:
            f, c = 1000.0, ET_LOW_1000X
        elif rp < 0.02 and rn < 0.02:
            f, c = 100.0, ET_LOW_100X
        elif rp < 0.2 and rn < 0.2:
            f, c = 10.0, ET_LOW_10X
        else:
            continue
        factor[t] = f
        etype[t] = c
        wtype[t] = WT_SINGLE
        ep_l[t] = xp
        ep_r[t] = xn
        rat_l[t] = rp
        rat_r[t] = rn


@njit(parallel=True, cache=True, error_model="numpy")
def detect_single_period_all(x, starts, factor, etype, wtype, ep_l, ep_r, rat_l, rat_r):
    """Run single-period detection over all securities in parallel."""
    n_groups = len(starts) - 1
    for g in prange(n_groups):
        s = starts[g]
        e = starts[g + 1]
        _detect_single_one(
            x[s:e],
            factor[s:e],
            etype[s:e],
            wtype[s:e],
            ep_l[s:e],
            ep_r[s:e],
            rat_l[s:e],
            rat_r[s:e],
        )


@njit(cache=True, error_model="numpy")
def _multi_pass_one(
    x,
    factor,
    etype,
    wsize,
    wtype,
    ep_l,
    ep_r,
    rat_l,
    rat_r,
    var,
    nlag,
    ep_lag,
    ep_lead,
    ilo,
    ihi,
    wt_code,
    vthr,
    det,
    det_type,
    det_epl,
    det_epr,
    det_ratl,
    det_ratr,
    det_var,
):
    """
    One window-type pass (FULL, SUB_A or SUB_B) for one security.

    Two phases: detection against the factor state at pass start (markers in
    det* scratch arrays, immutable during propagation), then propagation
    first-write-wins in ascending offset order onto rows whose factor is
    still 1.0.
    """
    n = len(x)
    for k in range(n):
        det[k] = np.nan

    # ---- Phase 1: detection ----
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
        # Interior variation: skip NaN/out-of-range cells (horizontal max/min
        # ignore nulls); all-NaN interior -> no detection
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
        variation = imax / imin
        if not (variation < vthr):
            continue
        # Nested magnitude classing, mirroring the single-period chain. The
        # original implementation's magnitude [1, 2, 3] loop was dead code
        # beyond 10x (the 10x pass always claimed the detection point first),
        # so multi-day 100x/1000x errors were under-corrected by a factor of
        # 10/100. This is the deliberate fix: class by both endpoint ratios.
        if high:
            if rl > 500.0 and rr > 500.0:
                det[t] = 0.001
                det_type[t] = ET_HIGH_1000X
            elif rl > 50.0 and rr > 50.0:
                det[t] = 0.01
                det_type[t] = ET_HIGH_100X
            else:
                det[t] = 0.1
                det_type[t] = ET_HIGH_10X
        else:
            if rl < 0.002 and rr < 0.002:
                det[t] = 1000.0
                det_type[t] = ET_LOW_1000X
            elif rl < 0.02 and rr < 0.02:
                det[t] = 100.0
                det_type[t] = ET_LOW_100X
            else:
                det[t] = 10.0
                det_type[t] = ET_LOW_10X
        det_epl[t] = xl
        det_epr[t] = xr
        det_ratl[t] = rl
        det_ratr[t] = rr
        det_var[t] = variation

    # ---- Phase 2: propagation, first-write-wins in ascending offset order ----
    for off in range(ilo, ihi + 1):
        for r in range(n):
            t = r - off  # shift(off) moves value from row r-off to row r
            if t < 0 or t >= n:
                continue
            if np.isnan(det[t]):
                continue
            if factor[r] != 1.0:
                continue
            factor[r] = det[t]
            etype[r] = det_type[t]
            wsize[r] = nlag
            wtype[r] = wt_code
            ep_l[r] = det_epl[t]
            ep_r[r] = det_epr[t]
            rat_l[r] = det_ratl[t]
            rat_r[r] = det_ratr[t]
            var[r] = det_var[t]


@njit(cache=True, error_model="numpy")
def _multi_one(x, factor, etype, wsize, wtype, ep_l, ep_r, rat_l, rat_r, var, nlags, vthr):
    """All multi-period passes for one security: nlags ascending, FULL -> SUB_A -> SUB_B."""
    n = len(x)
    det = np.empty(n, dtype=np.float64)
    det_type = np.empty(n, dtype=np.int8)
    det_epl = np.empty(n, dtype=np.float64)
    det_epr = np.empty(n, dtype=np.float64)
    det_ratl = np.empty(n, dtype=np.float64)
    det_ratr = np.empty(n, dtype=np.float64)
    det_var = np.empty(n, dtype=np.float64)
    for i in range(len(nlags)):
        nlag = nlags[i]
        if nlag <= 1:
            continue
        # FULL window: endpoints t-nlag / t+nlag, interior -(nlag-1)..(nlag-1)
        _multi_pass_one(
            x,
            factor,
            etype,
            wsize,
            wtype,
            ep_l,
            ep_r,
            rat_l,
            rat_r,
            var,
            nlag,
            nlag,
            nlag,
            -(nlag - 1),
            nlag - 1,
            WT_FULL,
            vthr,
            det,
            det_type,
            det_epl,
            det_epr,
            det_ratl,
            det_ratr,
            det_var,
        )
        if 2 * nlag - 2 >= 1:
            # SUB_A: endpoints t-nlag / t+nlag-1, interior -(nlag-1)..(nlag-2)
            _multi_pass_one(
                x,
                factor,
                etype,
                wsize,
                wtype,
                ep_l,
                ep_r,
                rat_l,
                rat_r,
                var,
                nlag,
                nlag,
                nlag - 1,
                -(nlag - 1),
                nlag - 2,
                WT_SUB_A,
                vthr,
                det,
                det_type,
                det_epl,
                det_epr,
                det_ratl,
                det_ratr,
                det_var,
            )
            # SUB_B: endpoints t-nlag+1 / t+nlag, interior -(nlag-2)..(nlag-1)
            _multi_pass_one(
                x,
                factor,
                etype,
                wsize,
                wtype,
                ep_l,
                ep_r,
                rat_l,
                rat_r,
                var,
                nlag,
                nlag - 1,
                nlag,
                -(nlag - 2),
                nlag - 1,
                WT_SUB_B,
                vthr,
                det,
                det_type,
                det_epl,
                det_epr,
                det_ratl,
                det_ratr,
                det_var,
            )


@njit(parallel=True, cache=True, error_model="numpy")
def detect_multi_period_all(
    x, starts, factor, etype, wsize, wtype, ep_l, ep_r, rat_l, rat_r, var, nlags, vthr
):
    """Run multi-period detection over all securities in parallel."""
    n_groups = len(starts) - 1
    for g in prange(n_groups):
        s = starts[g]
        e = starts[g + 1]
        _multi_one(
            x[s:e],
            factor[s:e],
            etype[s:e],
            wsize[s:e],
            wtype[s:e],
            ep_l[s:e],
            ep_r[s:e],
            rat_l[s:e],
            rat_r[s:e],
            var[s:e],
            nlags,
            vthr,
        )


@njit(cache=True, error_model="numpy")
def _validate_one(factor, etype, wsize, reject):
    """
    Cascading validation for one security: reject corrections whose BOTH
    endpoint positions (p +/- window_size, null -> 1) are themselves flagged.
    Rejections applied simultaneously per round, max 10 rounds. Only factor
    and error_type are reset (window size / debug columns intentionally kept,
    matching the original; rejected rows drop out of the log via factor != 1).
    """
    n = len(factor)
    for _ in range(10):
        any_reject = False
        for p in range(n):
            reject[p] = False
        for p in range(n):
            if factor[p] == 1.0:
                continue
            w = wsize[p] if wsize[p] != WS_NULL else 1
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
                etype[p] = ET_NULL


@njit(parallel=True, cache=True, error_model="numpy")
def validate_cascading_all(starts, factor, etype, wsize, rejected_mask):
    """Run cascading validation over all securities; record rejected rows."""
    n_groups = len(starts) - 1
    for g in prange(n_groups):
        s = starts[g]
        e = starts[g + 1]
        f = factor[s:e]
        before = np.empty(e - s, dtype=np.bool_)
        for k in range(e - s):
            before[k] = f[k] != 1.0
        scratch = np.empty(e - s, dtype=np.bool_)
        _validate_one(f, etype[s:e], wsize[s:e], scratch)
        for k in range(e - s):
            rejected_mask[s + k] = before[k] and f[k] == 1.0


# =============================================================================
# Section 8 filter kernels
# =============================================================================

# Removal-reason codes (string mapping lives in bessembinder.py). -1 = kept.
R_NULL = -1
R_8A_VOLUME = 0
R_8B_AJEX = 1
R_8B_QUNIT = 2  # reserved: qunit column never present on the post-merge frame
R_8C_INITIAL = 3
R_8C_LOW_PRICE_ME = 4
R_8D_GAP = 5
R_8E_ADJCSHO = 6
R_8F_ME_JUMP = 7
R_8G_RETURN = 8
R_8H_INITIAL = 9

_GAP_THRESHOLD_DAYS = int(231 * 365 / 252)  # 334: ~11 months of trading days

# Indices into the tunable-threshold array for filters 8e-8h (values built by
# Section8Params.to_array() in bessembinder.py; paper defaults documented there)
P8_E_UP_JUMP = 0
P8_E_UP_CONFIRM = 1
P8_E_CHN_UP_JUMP = 2
P8_E_CHN_UP_CONFIRM = 3
P8_E_DOWN_JUMP = 4
P8_E_DOWN_CONFIRM = 5
P8_EARLY_OBS = 6
P8_EARLY_FRAC = 7
P8_F_UP_RATIO = 8
P8_F_UP_RET = 9
P8_F_DOWN_RATIO = 10
P8_F_DOWN_RET = 11
P8_G_RET = 12
P8_G_ME_CHANGE = 13
P8_H_MAX_OBS = 14
P8_H_RATIO_HI = 15
P8_H_RATIO_LO = 16
P8_SIZE = 17


@njit(cache=True, error_model="numpy")
def _s8_kill_all(reason, code):
    """Remove every still-alive row of the security with the given reason."""
    for i in range(reason.size):
        if reason[i] == R_NULL:
            reason[i] = code


@njit(cache=True, error_model="numpy")
def _s8_early_jump_stage(reason, num, aux, code, is_ret, chn, p):
    """
    Shared early-jump stage for filters 8e (adjCSHO) and 8f (ME).

    Scans alive rows with prev-alive shift semantics; a jump inside the early
    period (obs_num < p[P8_EARLY_OBS] or < p[P8_EARLY_FRAC] of the alive
    count) marks the security for deletion of alive rows 0..max-jump-obs.
    `num` is the jump series (adjCSHO or ME); `aux` is the confirming series
    (ME ratio for 8e; ri-based return for 8f, is_ret=True). NaN operands fail
    all comparisons, matching polars null semantics.
    """
    n = reason.size
    total = 0
    for i in range(n):
        if reason[i] == R_NULL:
            total += 1
    if total == 0:
        return
    delete_through = -1
    obs = -1
    prev = -1
    for i in range(n):
        if reason[i] != R_NULL:
            continue
        obs += 1
        if prev >= 0:
            r_num = num[i] / num[prev]
            if is_ret:
                a = aux[i] / aux[prev] - 1.0
                up = r_num > p[P8_F_UP_RATIO] and a < p[P8_F_UP_RET]
                down = r_num < p[P8_F_DOWN_RATIO] and a > p[P8_F_DOWN_RET]
            else:
                a = aux[i] / aux[prev]
                if chn[i]:
                    up = r_num >= p[P8_E_CHN_UP_JUMP] and a >= p[P8_E_CHN_UP_CONFIRM]
                else:
                    up = r_num >= p[P8_E_UP_JUMP] and a >= p[P8_E_UP_CONFIRM]
                down = r_num <= p[P8_E_DOWN_JUMP] and a <= p[P8_E_DOWN_CONFIRM]
            if (up or down) and (obs < p[P8_EARLY_OBS] or obs < p[P8_EARLY_FRAC] * total):
                delete_through = obs
        prev = i
    if delete_through < 0:
        return
    obs = -1
    for i in range(n):
        if reason[i] != R_NULL:
            continue
        obs += 1
        if obs <= delete_through:
            reason[i] = code


@njit(cache=True, error_model="numpy")
def _s8_one_security(reason, remove_8a, ajexdi, prc, me, ri, cshoc, dates, low_thr, chn, p):
    """
    Apply the Section 8 filter chain to one security (arrays are views over
    the date-sorted slice). Filters run sequentially: each stage's shift /
    obs_num / totals are defined over the rows still alive at stage start,
    exactly like the polars chain re-deriving them on each filtered frame.
    reason is mutated in place (-1 = kept). p carries the tunable 8e-8h
    thresholds (see P8_* indices).
    """
    n = reason.size

    # ---- 8a: bottom-percentile average positive volume (global decision) ----
    if remove_8a:
        _s8_kill_all(reason, R_8A_VOLUME)
        return

    # ---- 8b: any AJEXDI == 0 removes the security ----
    for i in range(n):
        if ajexdi[i] == 0.0:
            _s8_kill_all(reason, R_8B_AJEX)
            return

    # ---- 8c: low price / market equity ----
    first = -1
    for i in range(n):
        if reason[i] == R_NULL:
            first = i
            break
    if first < 0:
        return
    breach_idx = -1
    for i in range(first, n):
        if reason[i] != R_NULL:
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
            if reason[i] != R_NULL or dates[i] != dates[first]:
                break
            thr = 0.001 if low_thr[i] else 0.01
            if me[i] < 1.0 or prc[i] < thr:
                initial = True
                break
        if initial:
            _s8_kill_all(reason, R_8C_INITIAL)
            return
        for i in range(n):
            if reason[i] == R_NULL and dates[i] >= dates[breach_idx]:
                reason[i] = R_8C_LOW_PRICE_ME

    # ---- 8d: drop observations after calendar gaps > ~11 months ----
    prev = -1
    for i in range(n):
        if reason[i] != R_NULL:
            continue
        if prev >= 0 and dates[i] - dates[prev] > _GAP_THRESHOLD_DAYS:
            reason[i] = R_8D_GAP
            # polars computes all gaps against the pre-stage frame, so the
            # removed row still serves as the next row's gap reference
        prev = i

    # ---- 8e: adjCSHO jumps in early history ----
    adjcsho = cshoc * ajexdi
    _s8_early_jump_stage(reason, adjcsho, me, R_8E_ADJCSHO, False, chn, p)

    # ---- 8f: ME jumps without commensurate returns in early history ----
    _s8_early_jump_stage(reason, me, ri, R_8F_ME_JUMP, True, chn, p)

    # ---- 8g: returns inconsistent with ME changes ----
    prev = -1
    for i in range(n):
        if reason[i] != R_NULL:
            continue
        if prev >= 0:
            ret = ri[i] / ri[prev] - 1.0
            me_chg = me[i] / me[prev] - 1.0
            if abs(ret) > p[P8_G_RET] and abs(me_chg) < p[P8_G_ME_CHANGE]:
                # prev still advances below: the flagged row stays the shift
                # source for the next row, matching polars stage-input shifts
                reason[i] = R_8G_RETURN
        prev = i

    # ---- 8h: large price/ME ratios within the first few observations ----
    prev = -1
    obs = -1
    for i in range(n):
        if reason[i] != R_NULL:
            continue
        obs += 1
        if obs > p[P8_H_MAX_OBS]:
            break
        if prev >= 0:
            pr = prc[i] / prc[prev]
            mr = me[i] / me[prev]
            if (
                pr > p[P8_H_RATIO_HI]
                or pr < p[P8_H_RATIO_LO]
                or mr > p[P8_H_RATIO_HI]
                or mr < p[P8_H_RATIO_LO]
            ):
                reason[i] = R_8H_INITIAL  # prev still advances: see 8g note
        prev = i


@njit(parallel=True, cache=True, error_model="numpy")
def section8_all(starts, reason, remove_8a, ajexdi, prc, me, ri, cshoc, dates, low_thr, chn, p):
    """Run the Section 8 filter chain over all securities in parallel."""
    n_groups = len(starts) - 1
    for g in prange(n_groups):
        s = starts[g]
        e = starts[g + 1]
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
            p,
        )
