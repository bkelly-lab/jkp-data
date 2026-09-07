# Golden-fixture tests

Two-layer protection for the factor-model builders shipped in PR #171
(`gen_ff_data`, `gen_hxz_data`, `gen_mispricing_data`):

- **Unit tests** (`tests/unit/test_gen_*.py`) pin stage-helper contracts on
  synthetic Polars frames so the internals can be refactored freely.
- **Golden tests** (here) pin the end-to-end output parquets against
  reference parquets produced by running the real builders on a fixed,
  fully synthetic input. Any drift in the externally observed output
  surfaces immediately.

## License compliance (synthetic inputs)

All builder inputs are **synthetic** and committed under
`fixtures/synthetic_wrds/`. No real WRDS cell is ever written:

- Every numeric value is independently sampled from a plausible
  distribution.
- Identifiers live in synthetic bands disjoint from real WRDS allocation:
  `permno >= 9_000_000`, `gvkey >= "900000"`, `world_id >= 9_000_000_000`.
- Schemas mirror the real WRDS tables, but the cell values are not derived
  from any obfuscation of WRDS data.

This satisfies the WRDS distribution agreement clause that permits "only
synthetic data or WRDS data obfuscated enough that it cannot be
reverse-engineered to the underlying source."

## What's committed

- `fixtures/synthetic_wrds/` — 16 deterministically generated parquets
  (~200 MB compressed with zstd-22) used as the builder inputs.
- `fixtures/ff/`, `fixtures/hxz/`, `fixtures/mispricing/` — golden output
  parquets produced by the three builders against the synthetic inputs.

## Test framing

Golden tests are a **regression guard with numerical tolerance**. They
confirm the builders reproduce a fixed output from the fixed synthetic
input within `rtol=1e-6, atol=1e-10` (see
`_golden_helpers.compare_factor_parquet`). Key columns (`excntry`/`eom`,
`excntry`/`date`, `id`/`eom`) match exactly. They do NOT validate the
factors against real-world returns.

## Workflow

### Regenerate the synthetic inputs

```bash
uv sync --group test
uv run python -m tests.golden.generate_synthetic_wrds
```

The generator is deterministic (seed = 42). Re-running overwrites
`fixtures/synthetic_wrds/` byte-identically.

### Regenerate the golden outputs (after intentional builder changes)

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

The `--regen-golden` flag swaps the golden parquets with the freshly-built
output and skips the assertion (matches the existing
`tests/golden/test_run_portfolio_golden.py` pattern).
