# Bessembinder Corrections — Validation Handoff

**Date:** 2026-06-10
**Branch:** `bessembinder_correction`
**Goal:** Validate the Bessembinder (2023) Section 6 (decimal-error correction) and
Section 8 (outlier filters) against CRSP by comparing Compustat-derived daily
returns/prices to CRSP for US common stocks, and produce a LaTeX/PDF report.

---

## TL;DR

- Compustat (Bessembinder-corrected) daily returns match CRSP to the penny for
  **93.5%** of matched obs; **99.28%** agree within 1 pp (`|Δ| < 0.01`).
- The headline Pearson correlation is destroyed by **~100 extreme rows** out of
  56M. Dropping `|ret_comp| > 10` restores All-stocks corr from **0.034 → 0.953**
  (Raw) and **0.821 → 0.976** (New/Bessembinder).
- The extreme spikes are a **US-only, Nano/Micro** phenomenon driven by very
  low-priced prior-day quotes (a $0.03–$0.30 print jumping to a real trade).
  They are **not** decimal errors and **not** (mostly) splits.
- **Section 8 already runs** (incl. the 8c penny floor) but its `< $0.01` price
  gate and `< $1M` ME gate sit far below where these spikes live, so ~21/23 of
  the surviving New-panel spikes evade 8c.
- In **production this is moot for US** (CRSP is the main source via `obs_main`)
  and **for world** (Compustat Global has **zero** `prcstd=4`); JKP also
  **winsorizes** Compustat returns to CRSP per-month 0.1/99.9 cutoffs
  (`wins_ret=True`) and uses ECDF-rank weighting.

---

## Deliverables (in repo root)

- `bessembinder_before_after.tex` / `.pdf` — **5-page landscape report**:
  1. **No Clip** — Raw vs New, all matched stocks, by size group (all returns).
  2. **Clipped** — same, with `|ret_comp| > 10` dropped from both panels.
  3. **|Δ| Frequency by Size Group** — binned `|Δ|` distribution, % within group.
  4. **|Δ| Cumulative Distribution (CDF)** — climbs to 100; All + 6 size groups.
  5. **Universe / Observation Counts** — CRSP → CIZ common → CCM-linked → matched.
- `bessembinder_corrections.tex` / `.pdf` — earlier single-table report (clipped,
  trfd-recovered numbers). Superseded by the multi-page doc for the latest views.
- `compile.sh` — `bash compile.sh <basename>` builds + opens a PDF (allow-listed
  in `.claude/settings.local.json`, no per-call auth prompt).

## Comparison methodology

- Match Compustat → CRSP on `(permno, date)` via CCM link
  (`crsp_ccmxpf_lnkhist`, `linktype ∈ {LU,LC}`, `linkprim ∈ {P,C}`).
- CRSP screened to **CIZ common stock on a main exchange**:
  `securitytype=EQTY, securitysubtype=COM, sharetype=NS,
  issuertype ∈ {ACOR,CORP}, primaryexch ∈ {A,N,Q}, conditionaltype=RW`.
- Compustat **not** filtered (all matchable rows kept).
- `ret = ri.pct_change()` over `(gvkey, iid)`; `Δ = ret_comp − ret_crsp`.
- **Section-8 gap-guard** on the New panel: null a surviving row's return when
  its immediately-preceding observation was a Section-8-removed row
  (`bessembinder_corrections_log`, `variable` starts with `"8"`). This is a
  validation-side guard, not a pipeline feature.
- Size groups: Compustat NYSE (`exchg=11`) monthly breakpoints
  (1/20/50/80 pct of month-end ME) → mega/large/small/micro/nano + unclassified.

## Key numbers (matched, trfd-recovered)

| Metric (All) | Raw | New |
|---|---|---|
| n | 56,132,957 | 52,725,976 |
| corr (clipped `|ret_comp|≤10`) | 0.953 | 0.976 |
| corr (no clip) | 0.034 | 0.821 |
| `|Δ|=0` | 93.52% | 93.83% |
| `|Δ|<0.01` | 99.28% | 99.31% |
| `|ret_comp|>10` count | 138 | 60 (23 after gap-guard) |

Match quality falls monotonically with size: exact-match 98.7% (Mega) →
79.4% (Nano). All `|Δ|≥10` obs are Nano/Micro/Small (zero in Mega/Large).

## Root-cause findings

- **trfd null recovery.** `trfd` (daily total-return factor) is null for
  never-dividend US securities (~45%). We set `trfd=1` for **never-dividend
  securities only** (not a blanket coalesce), recovering ~18M returns, via
  COALESCE + `BOOL_OR(div/divd/divsp > 0) OVER (gvkey,iid)` in `gen_comp_dsf`'s
  two **daily** SQL views (NA `comp_secd` + Global `comp_g_secd`).
  **Committed** to `bessembinder_correction` as `9ac797d` and pushed.
  NOTE: the **monthly** view (`comp_secm`, `ri_local = prccm/ajexm*trfm`,
  ~line 2121) has the same latent null-for-never-dividend bug and is **not
  fixed** — apply the same pattern to `trfm` if monthly recovery is wanted.
- **prcstd codes are file-specific.** US (`comp_secd`): only `{3,4,null}`,
  with **prcstd=4 = 26.6%** of rows (no-trade closing bid/ask midpoint).
  World (`comp_g_secd`): only `{5,10,null}`, **zero prcstd=4**. So `10` is a
  *native* Global code, not JKP-synthetic (earlier claim corrected).
- **Spike mechanism.** A prcstd=4 (or just very cheap) prior-day price
  ($0.03–$0.30) followed by a real trade → 100–2000× return. In the matched
  CIZ universe, only ~28–45% of `|ret|>10` spikes touch a prcstd=4 leg; the
  rest are real-trade (prcstd=3) low-price moves. So a bid/ask guard alone is a
  poor fix here (nulls ~12% of returns, catches <half the spikes).
- **Section 8 / Filter 8c** (the handoff's penny handling, from commit
  `605f131`, R. Capellini): deletes from first breach forward if `prc < $0.01`
  or `ME < $1M` (exception $0.001 for BRA/IDN/NGA/TUR). The surviving spikes
  have prev price $0.03–$11 (≥ $0.01) and ME ≥ $1M → **21/23 evade 8c**.

## Variables corrected by the algorithm

- **Section 6 (correction):** `prccd, cshoc, ajexdi, qunit, trfd, adrrc` (+ the
  derived `adjprc = prccd/ajexdi`, `adjcsho = cshoc·ajexdi`, then reconstruct
  `prccd`/`cshoc`). Each gets `*_correction_factor` / `*_error_type` / `*_window_*`.
- **Section 8 (deletion, not correction):** filters 8a–8h (volume, AJEX/QUNIT,
  low price/ME, gaps, share/ME jumps, return bounds, initial-obs). Logged with
  `variable` starting `"8"`. Params in `Section8Params` (8e–8h tunable).

## Fix options discussed (none committed)

1. **Higher price floor** (e.g. the $5 used in the MP mispricing factor) — kills
   all 23 survivors; crude, biases small-cap coverage.
2. **Return winsorization vs CRSP** — JKP's production default; caps the spike.
3. **Bid/ask return guard** (null returns touching prcstd=4) — principled but
   nulls ~12% of US returns and only catches ~28–45% of spikes.
4. **Hampel / local median-MAD filter on log-price** — prcstd-agnostic, no hard
   floor, catches both bid/ask and real-trade spikes; needs k,c calibration.
   *Recommended for further testing; not yet run.*

## Cluster artifacts (`ffr7@yale`, `hpc.som.yale.edu`)

- Checkout: `/home/ffr7/jkp-data-bess` (has the trfd fix in `aux_functions.py`).
- Data: `/home/ffr7/bessembinder_validation/data/interim/`
  - `__comp_dsf_uncorrected_trfd.parquet` (Raw, trfd-recovered)
  - `__comp_dsf_bessembinder_trfd.parquet` (New, Section 6+8, trfd-recovered)
  - `__comp_dsf_uncorrected.parquet` / `__comp_dsf_bessembinder.parquet` (pre-trfd)
  - `size_grp_monthly_all.parquet`, `bessembinder_corrections_log.parquet`
- CRSP/raw read-only from `~/jkp-data/data/{interim/crsp_dsf.parquet, raw/raw_tables/}`.
- Analysis scripts in `/home/ffr7/`: `trfdcompare.py` / `trfdcompare4.py`
  (before/after compare, clip, gap-guard, size), `freqba.py` / `freqsize.py`
  (|Δ| freq + CDF by size), `p4sit.py` / `blindspot.py` (prcstd=4 / 8c analysis),
  `outliers2.py` (spike root-cause), `retcount.py`, `prcstd_count.py`.
  sbatch: `trfdcmp4.sbatch` (h100, 32 cpu, 256G).

## Open items / next steps

- trfd never-dividend fix is **committed/pushed** (`9ac797d`); open a PR when
  ready. Consider extending the same fix to the **monthly** `trfm` view.
- If a US Compustat return cleanup is wanted (beyond CRSP-primary), test the
  **Hampel log-price filter** vs the $5 floor vs winsorization on the New panel:
  report spikes remaining, All corr (no clip), and % returns nulled.
- Decide whether 8c's `< $0.01` should be `<=`, and whether prcstd=4 should be
  screened — both close part of the gap but neither alone fixes the matched-set
  spikes.
