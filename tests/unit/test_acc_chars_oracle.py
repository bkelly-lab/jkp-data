"""Expected values for every accounting characteristic (issue #303, step 5).

acc_char_oracle.py computes the characteristics again in plain Python from accounting_chars.sas,
following Python's rule where the pipeline departs from the SAS on purpose. These tests take the
stage run on the synthetic Compustat inputs (acc_stage in tests/conftest.py) and compare each
characteristic with the oracle in the annual, quarterly and combined outputs, on every persona and
on RANDOM_PER_SOURCE random firms of each source.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.generate_acc_chars_golden import KEYS
from tests.golden.synthetic_compustat import PERSONA_GVKEY
from tests.unit import acc_char_oracle as oracle

pytestmark = pytest.mark.methodology

KEEP = af.acc_chars_list()
# output file -> its characteristics' suffix
FILES = {"achars_world": "", "qchars_world": "_qitem", "acc_chars_world": ""}
# The persistence regression runs in DuckDB, so allow last-digit differences.
REL_TOL, ABS_TOL = 1e-9, 1e-12
# Our choice, for CI time: the oracle runs in plain Python, so it checks every persona but only the
# first RANDOM_PER_SOURCE random firms of each source (NA, Global). The golden fingerprint
# (tests/golden/test_acc_chars_golden.py) covers every firm.
RANDOM_PER_SOURCE = 5
# characteristic -> why the stage differs from the oracle. None today.
KNOWN_DIFFERENCES: dict[str, str] = {}

Rows = dict[tuple[str, dt.date], dict]


@pytest.fixture(scope="module")
def firms(acc_stage: DataPaths) -> list[str]:
    """Every persona and the first RANDOM_PER_SOURCE random firms of each source."""
    std = [acc_stage.interim_dir / f"{name}.parquet" for name in ("acc_std_ann", "acc_std_qtr")]
    gvkeys = pl.concat(pl.read_parquet(p, columns=["gvkey", "source"]) for p in std)
    gvkeys = gvkeys.drop_nulls().unique().sort("gvkey")
    persona = pl.col("gvkey").cast(pl.Int64) >= PERSONA_GVKEY
    first = pl.int_range(pl.len()).over("source") < RANDOM_PER_SOURCE
    random = gvkeys.filter(~persona).filter(first)
    return sorted({*gvkeys.filter(persona)["gvkey"], *random["gvkey"]})


@pytest.fixture(scope="module")
def oracle_runs(acc_stage: DataPaths, firms: list[str]) -> dict[str, Rows]:
    """The oracle's annual and quarterly runs on the firms and their combination, keyed like FILES."""
    i = acc_stage.interim_dir
    fx = {
        (r["curcdd"], r["datadate"]): r["fx"]
        for r in af.compustat_fx(acc_stage).iter_rows(named=True)
    }
    of_firms = pl.col("gvkey").is_in(firms)
    me = oracle.me_company_by_month(pl.read_parquet(i / "world_msf.parquet").filter(of_firms))
    ann, qtr = (
        oracle.stage_characteristics(
            pl.read_parquet(i / f"{name}.parquet").filter(of_firms), fx, me, KEEP
        )
        for name in ("acc_std_ann", "acc_std_qtr")
    )
    return {
        "achars_world": ann,
        "qchars_world": qtr,
        "acc_chars_world": oracle.combine(ann, qtr, KEEP),
    }


@pytest.fixture(scope="module")
def expected(oracle_runs: dict[str, Rows]) -> dict[str, pl.DataFrame]:
    """The oracle's rows as frames: the keys, source and the keep list as Float64."""
    schema = {"gvkey": pl.String, "public_date": pl.Date, "source": pl.String}
    schema |= dict.fromkeys(KEEP, pl.Float64)
    return {
        name: pl.from_dicts(
            [{**row, "gvkey": g, "public_date": d} for (g, d), row in rows.items()], schema=schema
        )
        for name, rows in oracle_runs.items()
    }


@pytest.fixture(scope="module")
def actual(acc_stage: DataPaths, firms: list[str]) -> dict[str, pl.DataFrame]:
    """The stage's three output files on the firms, with the quarterly suffix removed."""
    out = {}
    for name, suffix in FILES.items():
        df = pl.read_parquet(acc_stage.interim_dir / f"{name}.parquet")
        df = df.rename({c: c.removesuffix(suffix) for c in df.columns})
        out[name] = df.filter(pl.col("gvkey").is_in(firms)).select(
            *KEYS, "source", pl.col(KEEP).cast(pl.Float64)
        )
    return out


@pytest.fixture(scope="module")
def compared(
    expected: dict[str, pl.DataFrame], actual: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    """Per output file, the oracle's rows joined to the stage's on the keys (the stage's columns
    end in _got)."""
    return {name: expected[name].join(actual[name], on=KEYS, suffix="_got") for name in FILES}


def _same(want: pl.Expr, got: pl.Expr) -> pl.Expr:
    """
    Description:
        math.isclose(want, got, rel_tol=REL_TOL, abs_tol=ABS_TOL) as an expression, for SAS
        missing values: null is missing, and a NaN in the stage's output is not, so it never
        matches.
    Steps:
        1) Null matches only null; NaN matches nothing; an infinity matches only itself.
        2) Otherwise |want - got| <= max(REL_TOL x max(|want|, |got|), ABS_TOL).
    Output:
        Boolean expression, never null.
    """
    tol = pl.max_horizontal(REL_TOL * pl.max_horizontal(want.abs(), got.abs()), pl.lit(ABS_TOL))
    return (
        pl.when(want.is_null() | got.is_null())
        .then(want.is_null() & got.is_null())
        .when(want.is_nan() | got.is_nan())
        .then(False)
        .when(want.is_infinite() | got.is_infinite())
        .then(want == got)
        .otherwise((want - got).abs() <= tol)
    )


def test_oracle_covers_every_characteristic(oracle_runs: dict[str, Rows]) -> None:
    """stage_characteristics raises on a keep-list name no oracle section produces."""
    assert set(KEEP) <= set(next(iter(oracle_runs["acc_chars_world"].values())))


@pytest.mark.parametrize("name", FILES)
def test_public_dates_and_source_match_oracle(
    name: str,
    expected: dict[str, pl.DataFrame],
    actual: dict[str, pl.DataFrame],
    compared: dict[str, pl.DataFrame],
) -> None:
    assert expected[name].select(KEYS).sort(KEYS).equals(actual[name].select(KEYS).sort(KEYS))
    sources = compared[name].filter(pl.col("source").ne_missing(pl.col("source_got")))
    assert sources.height == 0, sources.select(*KEYS, "source", "source_got").head()


@pytest.mark.parametrize(
    "char",
    [
        pytest.param(c, marks=pytest.mark.xfail(reason=KNOWN_DIFFERENCES[c], strict=True))
        if c in KNOWN_DIFFERENCES
        else c
        for c in KEEP
    ],
)
def test_characteristic_matches_oracle(char: str, compared: dict[str, pl.DataFrame]) -> None:
    """Each output file's values of one characteristic equal the oracle's, nulls included."""
    report = {}
    for name, df in compared.items():
        bad = df.filter(~_same(pl.col(char), pl.col(f"{char}_got"))).sort(KEYS)
        if bad.height:
            report[name] = (bad.height, bad.select(*KEYS, char, f"{char}_got").head(3).rows())
    assert not report, report
