# Golden-fixture tests

Two-layer protection for the factor-model builders (currently `gen_ff_data`
and `gen_hxz_data`; the other factor models add their own golden tests as
they land):

- **Unit tests** (`tests/unit/test_gen_*.py`) pin stage-helper contracts on
  synthetic Polars frames so the internals can be refactored freely.
- **Golden tests** (here) pin the end-to-end output parquets against
  reference parquets produced by running the real builders on a fixed,
  fully synthetic input. Any drift in the externally observed output
  surfaces immediately.

## License compliance (synthetic inputs)

All builder inputs are **synthetic** and generated at test time by
`generate_synthetic_wrds.py` (no input data is committed). No real WRDS cell
is ever written:

- Every numeric value is independently sampled from a plausible
  distribution.
- Identifiers live in synthetic bands disjoint from real WRDS allocation:
  `permno >= 9_000_000`, `gvkey >= "900000"`, `world_id >= 9_000_000_000`.
- Schemas mirror the real WRDS tables (column set, dtypes, and null patterns
  measured on WRDS 2000-2015), but the cell values are not derived
  from any obfuscation of WRDS data.

This satisfies the WRDS distribution agreement clause that permits "only
synthetic data or WRDS data obfuscated enough that it cannot be
reverse-engineered to the underlying source."

## What's committed

- `fixtures/ff/`, `fixtures/hxz/` — golden output parquets produced by
  `gen_ff_data` and `gen_hxz_data` against the synthetic inputs.

## Test framing

Golden tests are a **regression guard with numerical tolerance**. They
confirm the builders reproduce a fixed output from the fixed synthetic
input within `rtol=1e-6, atol=1e-10` (see
`_golden_helpers.compare_factor_parquet`). Key columns (`excntry`/`eom`,
`excntry`/`date`, `id`/`eom`) match exactly. They do NOT validate the
factors against real-world returns.

## Workflow

### Synthetic inputs

The `synthetic_wrds_dir` session fixture (`conftest.py`) calls
`write_synthetic_wrds` once into a pytest temp directory. To inspect the
17 parquets:

```bash
uv sync --group test
uv run python -m tests.golden.generate_synthetic_wrds --out /tmp/synthetic_wrds
```

The generator is deterministic (seed = 42; nulls come from a separate RNG).

Nulls are sampled from `synthetic_null_patterns.json`: for each table (and, for
`world_msf` / `world_dsf` / `world_data`, for CRSP-sourced and ex-US rows separately) the
share of rows with each pattern of null columns, measured on the real tables for 2000-2015.
The file holds frequencies only, no values or identifiers; patterns seen in fewer than 100
real rows are dropped. Firm counts (`US_EXCH_COUNTS`, `ROW_FIRMS`) are set so the
CRSP-sourced share of `world_msf` and the North America (US + Canada) share of
`acc_std_ann` match the real tables.

### Regenerate the golden outputs (after intentional builder changes)

```bash
uv run python -m tests.golden.generate_ff_golden
uv run python -m tests.golden.generate_hxz_golden
```

Commit the updated golden parquets in the same change that touches the
builder source. Reviewers see the diff and approve / reject intentional
shifts.

### Run the golden tests

```bash
uv run pytest tests/golden/test_gen_ff_data_golden.py \
              tests/golden/test_gen_hxz_data_golden.py -v
```

The `--regen-golden` flag swaps the golden parquets with the freshly-built
output and skips the assertion (matches the existing
`tests/golden/test_run_portfolio_golden.py` pattern).
