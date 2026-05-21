# Golden-fixture tests

Two-layer protection for the factor-model builders shipped in PR #171
(`gen_ff_data`, `gen_hxz_data`, `gen_mispricing_data`):

- **Unit tests** (`tests/unit/test_gen_*.py`) pin stage-helper contracts on
  synthetic Polars frames so the internals can be refactored freely.
- **Golden tests** (here) pin the end-to-end output parquets against
  reference parquets generated from a real WRDS slice. Any drift in the
  builder's externally observed output surfaces immediately.

## What's committed

- `fixtures/ff/`, `fixtures/hxz/`, `fixtures/mispricing/` — committed
  golden parquets produced by the three builders. ~3 MB total.
- `fixtures/wrds_slices/MANIFEST.json` — generated alongside the slices;
  records the source + row counts + sha256 prefix so reviewers can audit
  reproducibility.

## What's NOT committed

- `fixtures/wrds_slices/*.parquet` — the WRDS input slices are ~2 GB
  (gitignored). They live on the cluster; regenerate locally before
  running the golden tests.

## Workflow

### Regenerate the slices (one-time per source-data refresh)

```bash
uv sync --group test
uv run python -m tests.golden.generate_wrds_slices \
    --source /path/to/jkp-data \         # an existing pipeline data dir
    --start 2018-01-01 --end 2020-12-31 \ # default
    --countries USA                       # default; expand as needed
```

`--source` must contain `raw/raw_tables/` (WRDS pulls) and `interim/`
(world_msf / world_dsf / world_data / market_returns* / raw_data_dfs).

### Regenerate the golden parquets (after intentional output changes)

```bash
uv run python -m tests.golden.generate_ff_golden
uv run python -m tests.golden.generate_hxz_golden
uv run python -m tests.golden.generate_mispricing_golden
```

Commit the updated golden parquets in the same change that touches the
builder source. Reviewers see the diff and approve / reject intentional
shifts.

### Run the golden tests

```bash
uv run pytest tests/golden/test_gen_ff_data_golden.py \
              tests/golden/test_gen_hxz_data_golden.py \
              tests/golden/test_gen_mispricing_data_golden.py -v
```

Each test skips cleanly when its required WRDS slices are missing — CI
without slices simply skips. The `--regen-golden` flag swaps the golden
parquets with the freshly-built output and skips the assertion (matches
the existing `tests/golden/test_run_portfolio_golden.py` pattern).
