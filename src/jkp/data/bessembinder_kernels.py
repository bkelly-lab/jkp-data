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
    - Multi-period uses single-magnitude thresholds (5 / 0.2, factors
      0.1 / 10): the original's magnitude-[1,2,3] loop is dead code for
      magnitudes 2 and 3 (proven byte-identical in commit 507df7c).
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


@njit(cache=True)
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


@njit(parallel=True, cache=True)
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


@njit(cache=True)
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
        if high:
            det[t] = 0.1
            det_type[t] = ET_HIGH_10X
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


@njit(cache=True)
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


@njit(parallel=True, cache=True)
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


@njit(cache=True)
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


@njit(parallel=True, cache=True)
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
