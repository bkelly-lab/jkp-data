"""Metamorphic tests for the accounting stage (issue #303, step 5).

Each test changes the synthetic Compustat inputs (synthetic_inputs in tests/conftest.py) in a way
whose effect on the output is known, runs the stage on them and compares its annual, quarterly and
combined outputs with the stage's outputs on the unchanged inputs (acc_stage). Changes whose
effects can't mask each other share a stage run, which keeps the tests fast:
    scaled    every monetary item and ME doubled: the levels double, nothing else changes
    extended  copies of two firms under new gvkeys, and a copy in EUR of an AUD firm with the same
              USD values: each copy equals its source and no other firm changes, which catches a
              lag or window computed across firms
    moved     gvkeys in reverse order, every date 12 months later (and the book-equity deferred-tax
              cutoff year one later) and every input table's rows in another order: the outputs
              relabeled and 12 months later, which also catches a pick that depends on the order
              of the input rows
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.generate_acc_chars_golden import KEYS, run_stage

pytestmark = pytest.mark.integration

CHARS = af.acc_chars_list()
# output file -> its characteristics' suffix
FILES = {"achars_world": "", "qchars_world": "_qitem", "acc_chars_world": ""}
COMPUSTAT = ["comp_funda", "comp_fundq", "comp_g_funda", "comp_g_fundq"]
# The persistence regression runs in DuckDB, so allow last-digit differences.
REL_TOL, ABS_TOL = 1e-9, 1e-12
# The compared rows must have a value in at least this share of the characteristics, so a firm
# whose outputs are all null can't pass a comparison. The least-filled comparison (the twin's
# quarterly rows) has 257 of 315.
MIN_FILLED = 0.75

SCALE = 2  # a power of two, so the scaled inputs and every product and ratio of them are exact
NOT_MONEY = ["emp"]  # employees, in thousands
# Characteristics in currency units, which scale with the inputs.
LEVELS = [
    "assets", "sales", "book_equity", "net_income", "enterprise_value", "intrinsic_value",
    "ni_emp", "sale_emp",
]  # fmt: skip
# Characteristics that change otherwise with the scale, and why.
SCALE_DEPENDENT = {
    "o_score": "log(at_x) (char_macros.sas:482)",
    "eq_dur": "missing if lagged be_x or sale_x <= 1 million (char_macros.sas:427-428)",
}

# new gvkey -> the firm it copies (NA in USD, Global in EUR); the new gvkeys sort between the
# random firms (900001-900060) and the personas (900101 up)
COPIES = {"900070": "900010", "900080": "900030"}
TWIN, TWIN_SOURCE, TWIN_CURRENCY = "900090", "900034", "EUR"  # TWIN_SOURCE reports in AUD
# 900107's pre-1982 CAD rates come from a bundled file, whose dates the test does not shift, so
# the moved run is compared without it.
SHIFT_EXCLUDED = ["900107"]
SHUFFLE_SEED = 303

Tables = dict[str, pl.DataFrame]
Outputs = dict[str, pl.DataFrame]


def _run(root: Path, tables: Tables) -> DataPaths:
    """Write the inputs under root/input and run the stage under root/run."""
    inputs = DataPaths(root / "input")
    inputs.raw_tables_dir.mkdir(parents=True)
    inputs.interim_dir.mkdir(parents=True)
    for name, df in tables.items():
        folder = inputs.interim_dir if name == "world_msf" else inputs.raw_tables_dir
        df.write_parquet(folder / f"{name}.parquet")
    paths = DataPaths(root / "run")
    run_stage(paths, inputs)
    return paths


def _read(path: Path, suffix: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return df.rename({c: c.removesuffix(suffix) for c in df.columns})


def _outputs(paths: DataPaths) -> Outputs:
    """The stage's three output files, with the quarterly suffix removed."""
    return {name: _read(paths.interim_dir / f"{name}.parquet", s) for name, s in FILES.items()}


def _differences(want: pl.DataFrame, got: pl.DataFrame) -> dict[str, int]:
    """
    Description:
        Count, per column, the rows where two outputs of the stage differ.
    Steps:
        1) Require the same columns, dtypes and keys, and values in at least MIN_FILLED of the
           characteristics.
        2) Match rows on the keys. Two finite values differ by more than
           ABS_TOL + REL_TOL x |want|; otherwise null, NaN and each infinity equal only
           themselves. Source, and datadate where the file has it, must be equal.
    Output:
        Column -> number of differing rows, for the columns with any.
    """
    assert want.schema == got.schema
    assert want.height > 0
    assert want.select(KEYS).sort(KEYS).equals(got.select(KEYS).sort(KEYS))
    filled = want.select(pl.col(CHARS).is_not_null().any()).row(0)
    assert sum(filled) >= MIN_FILLED * len(CHARS)
    exact = [c for c in ["source", "datadate"] if c in want.columns]

    def differs(c: str) -> pl.Expr:
        a, b = pl.col(c).cast(pl.Float64), pl.col(f"{c}_got").cast(pl.Float64)
        far = (a - b).abs() > ABS_TOL + REL_TOL * a.abs()
        return pl.when(a.is_finite() & b.is_finite()).then(far).otherwise(a.ne_missing(b))

    counts = (
        want.join(got, on=KEYS, suffix="_got")
        .select(
            *(pl.col(c).ne_missing(pl.col(f"{c}_got")).sum() for c in exact),
            *(differs(c).sum().alias(c) for c in CHARS),
        )
        .row(0, named=True)
    )
    return {c: n for c, n in counts.items() if n}


def _money(df: pl.DataFrame) -> list[str]:
    """The table's monetary items: every decimal and float column but NOT_MONEY."""
    return [
        c for c, t in df.schema.items() if (t.is_decimal() or t.is_float()) and c not in NOT_MONEY
    ]


def _with_copies(tables: Tables, copies: dict[str, str]) -> Tables:
    """Add each source firm's rows in the Compustat tables and world_msf again under its new gvkey."""

    def add(df: pl.DataFrame) -> pl.DataFrame:
        rows = [
            df.filter(pl.col("gvkey") == s).with_columns(gvkey=pl.lit(g)) for g, s in copies.items()
        ]
        return pl.concat([df, *rows])

    return tables | {name: add(tables[name]) for name in [*COMPUSTAT, "world_msf"]}


def _with_twin(tables: Tables, fx: pl.DataFrame) -> Tables:
    """
    Description:
        Add TWIN: TWIN_SOURCE's reports in TWIN_CURRENCY, at the same USD values.
    Steps:
        1) Cast the Compustat tables' monetary items to Float64, so the converted items are not
           rounded to the items' decimals.
        2) Copy TWIN_SOURCE's rows that have an FX rate on their datadate as TWIN; multiply its
           items by its currency's rate and divide by TWIN_CURRENCY's (both USD per unit, from
           compustat_fx, on datadate, as standardized_accounting_data converts them).
        3) Copy its world_msf rows, as ME is in USD.
    Output:
        The changed tables.
    """
    out = tables | {"world_msf": _with_copies(tables, {TWIN: TWIN_SOURCE})["world_msf"]}
    for name in COMPUSTAT:
        df = tables[name].with_columns(pl.col(_money(tables[name])).cast(pl.Float64))
        currency = "curcd" if "curcd" in df.columns else "curcdq"
        source_currencies = df.filter(pl.col("gvkey") == TWIN_SOURCE)[currency].unique()
        assert TWIN_CURRENCY not in source_currencies.to_list(), "the twin must change currency"
        rate = fx.select("datadate", pl.col("curcdd").alias(currency), "fx")
        twin_rate = rate.filter(pl.col(currency) == TWIN_CURRENCY).select("datadate", to="fx")
        twin = (
            df.filter(pl.col("gvkey") == TWIN_SOURCE)
            .join(rate, on=["datadate", currency])
            .join(twin_rate, on="datadate")
            .with_columns(
                pl.col(_money(df)) * pl.col("fx") / pl.col("to"),
                gvkey=pl.lit(TWIN),
                **{currency: pl.lit(TWIN_CURRENCY)},
            )
            .drop("fx", "to")
        )
        out[name] = pl.concat([df, twin])
    return out


def _reversed_gvkeys(tables: Tables) -> dict[str, str]:
    """Each gvkey -> the gvkey at its position counted from the end of the sorted gvkeys."""
    gvkeys = (
        pl.concat([tables[name]["gvkey"] for name in [*COMPUSTAT, "world_msf"]])
        .drop_nulls()
        .unique()
        .sort()
        .to_list()
    )
    return dict(zip(gvkeys, reversed(gvkeys), strict=True))


def _relabeled(tables: Tables) -> Tables:
    reverse = _reversed_gvkeys(tables)
    return tables | {
        name: tables[name].with_columns(pl.col("gvkey").replace(reverse))
        for name in [*COMPUSTAT, "world_msf"]
    }


def _later(column: str) -> pl.Expr:
    """The month-end 12 months later."""
    return pl.col(column).dt.offset_by("12mo").dt.month_end()


def _shifted(tables: Tables) -> Tables:
    """Every date 12 months later, fiscal years one later; FX quotes one year later."""
    fiscal_year = {
        "comp_funda": "fyear",
        "comp_g_funda": "fyear",
        "comp_fundq": "fyearq",
        "comp_g_fundq": "fyearq",
    }
    return (
        tables
        | {
            name: tables[name].with_columns(_later("datadate"), pl.col(year) + 1)
            for name, year in fiscal_year.items()
        }
        | {
            "comp_exrt_dly": tables["comp_exrt_dly"].with_columns(
                pl.col("datadate").dt.offset_by("1y")
            ),
            "world_msf": tables["world_msf"].with_columns(_later("eom")),
        }
    )


def _shuffled(tables: Tables) -> Tables:
    """Every input table with its rows in a random (seeded) order."""
    return {
        name: df.sample(fraction=1, shuffle=True, seed=SHUFFLE_SEED) for name, df in tables.items()
    }


@pytest.fixture(scope="module")
def inputs(synthetic_inputs: DataPaths) -> Tables:
    raw = synthetic_inputs.raw_tables_dir
    tables = {
        name: pl.read_parquet(raw / f"{name}.parquet") for name in [*COMPUSTAT, "comp_exrt_dly"]
    }
    msf = pl.read_parquet(synthetic_inputs.interim_dir / "world_msf.parquet")
    return tables | {"world_msf": msf}


@pytest.fixture(scope="module")
def base(acc_stage: DataPaths) -> Outputs:
    return _outputs(acc_stage)


@pytest.fixture(scope="module")
def scaled(tmp_path_factory: pytest.TempPathFactory, inputs: Tables) -> Outputs:
    def times(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        return df.with_columns((pl.col(c) * SCALE).cast(df.schema[c]) for c in columns)

    tables = inputs | {name: times(inputs[name], _money(inputs[name])) for name in COMPUSTAT}
    tables["world_msf"] = times(inputs["world_msf"], ["me_company"])
    return _outputs(_run(tmp_path_factory.mktemp("scaled"), tables))


@pytest.fixture(scope="module")
def extended(
    tmp_path_factory: pytest.TempPathFactory, inputs: Tables, acc_stage: DataPaths
) -> Outputs:
    tables = _with_twin(_with_copies(inputs, COPIES), af.compustat_fx(acc_stage))
    return _outputs(_run(tmp_path_factory.mktemp("extended"), tables))


@pytest.fixture(scope="module")
def moved(tmp_path_factory: pytest.TempPathFactory, inputs: Tables) -> Outputs:
    tables = _shuffled(_shifted(_relabeled(inputs)))
    # The deferred-tax cutoff in be_x is a calendar year, so it moves with the dates.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(af, "BE_DEFERRED_TAX_END_YEAR", af.BE_DEFERRED_TAX_END_YEAR + 1)
        return _outputs(_run(tmp_path_factory.mktemp("moved"), tables))


@pytest.mark.parametrize("name", FILES)
def test_scaling_money_scales_the_levels_only(name: str, base: Outputs, scaled: Outputs) -> None:
    want = base[name].with_columns(pl.col(LEVELS) * SCALE)
    assert _differences(want, scaled[name]).keys() == SCALE_DEPENDENT.keys()


@pytest.mark.parametrize("name", FILES)
def test_adding_firms_changes_no_other_firm(name: str, base: Outputs, extended: Outputs) -> None:
    others = extended[name].filter(~pl.col("gvkey").is_in([*COPIES, TWIN]))
    assert _differences(base[name], others) == {}


@pytest.mark.parametrize("name", FILES)
def test_copied_firms_equal_their_sources(name: str, base: Outputs, extended: Outputs) -> None:
    copies = extended[name].filter(pl.col("gvkey").is_in(list(COPIES)))
    sources = base[name].filter(pl.col("gvkey").is_in(list(COPIES.values())))
    assert _differences(sources, copies.with_columns(pl.col("gvkey").replace(COPIES))) == {}


@pytest.mark.parametrize("name", FILES)
def test_twin_in_another_currency_equals_its_source(name: str, extended: Outputs) -> None:
    source = extended[name].filter(pl.col("gvkey") == TWIN_SOURCE)
    copy = extended[name].filter(pl.col("gvkey") == TWIN).with_columns(gvkey=pl.lit(TWIN_SOURCE))
    assert _differences(source, copy) == {}


@pytest.mark.parametrize("name", FILES)
def test_moving_firms_and_dates_moves_the_outputs(
    name: str, inputs: Tables, base: Outputs, moved: Outputs
) -> None:
    back = moved[name].with_columns(pl.col("gvkey").replace(_reversed_gvkeys(inputs)))
    kept = ~pl.col("gvkey").is_in(SHIFT_EXCLUDED)
    dates = [c for c in ["public_date", "datadate"] if c in base[name].columns]
    want = base[name].filter(kept).with_columns(_later(c) for c in dates)
    assert _differences(want, back.filter(kept)) == {}
