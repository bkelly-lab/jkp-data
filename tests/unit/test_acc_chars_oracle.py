"""Expected values for every accounting characteristic (issue #303, step 5).

acc_char_oracle.py computes the characteristics again in plain Python from accounting_chars.sas.
These tests run the stage on the synthetic Compustat inputs (tests/golden/fixtures/
synthetic_compustat/) and compare each characteristic with the oracle in the annual, quarterly and
combined outputs.
"""

from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from jkp.data import aux_functions as af
from jkp.data.paths import DataPaths
from tests.golden.generate_acc_chars_golden import run_stage
from tests.unit import acc_char_oracle as oracle

pytestmark = pytest.mark.methodology

KEEP = af.acc_chars_list()
# output file -> its characteristics' suffix
FILES = {"achars_world": "", "qchars_world": "_qitem", "acc_chars_world": ""}
# The persistence regression runs in DuckDB, so allow last-digit differences.
REL_TOL, ABS_TOL = 1e-9, 1e-12
# characteristic -> why the stage deliberately differs from the SAS. None today.
KNOWN_DIFFERENCES: dict[str, str] = {}

Frame = dict[tuple[str, dt.date], dict]


@pytest.fixture(scope="module")
def stage(tmp_path_factory: pytest.TempPathFactory) -> DataPaths:
    paths = DataPaths(tmp_path_factory.mktemp("acc_stage"))
    run_stage(paths)
    return paths


@pytest.fixture(scope="module")
def expected(stage: DataPaths) -> dict[str, Frame]:
    """The oracle's annual and quarterly runs and their combination, keyed like FILES."""
    i = stage.interim_dir
    fx = {
        (r["curcdd"], r["datadate"]): r["fx"] for r in af.compustat_fx(stage).iter_rows(named=True)
    }
    me = oracle.me_company_by_month(pl.read_parquet(i / "world_msf.parquet"))
    ann, qtr = (
        oracle.stage_characteristics(pl.read_parquet(i / f"{name}.parquet"), fx, me, KEEP)
        for name in ("acc_std_ann", "acc_std_qtr")
    )
    return {
        "achars_world": ann,
        "qchars_world": qtr,
        "acc_chars_world": oracle.combine(ann, qtr, KEEP),
    }


@pytest.fixture(scope="module")
def actual(stage: DataPaths) -> dict[str, Frame]:
    """The stage's three output files, with the quarterly suffix removed."""
    out = {}
    for name, suffix in FILES.items():
        df = pl.read_parquet(stage.interim_dir / f"{name}.parquet")
        if suffix:
            df = df.rename({c: c.removesuffix(suffix) for c in df.columns if c.endswith(suffix)})
        out[name] = {(r["gvkey"], r["public_date"]): r for r in df.iter_rows(named=True)}
    return out


def _same(want: float | None, got: float | None) -> bool:
    """SAS missing is null; a NaN in the stage's output is not missing, so it never matches."""
    if want is None or got is None:
        return want is None and got is None
    return not math.isnan(got) and math.isclose(want, got, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def test_oracle_covers_every_characteristic(expected: dict[str, Frame]) -> None:
    """stage_characteristics raises on a keep-list name no oracle section produces."""
    assert set(KEEP) <= set(next(iter(expected["acc_chars_world"].values())))


@pytest.mark.parametrize("name", FILES)
def test_public_dates_and_source_match_oracle(
    name: str, expected: dict[str, Frame], actual: dict[str, Frame]
) -> None:
    assert expected[name].keys() == actual[name].keys()
    sources = [k for k, row in expected[name].items() if row["source"] != actual[name][k]["source"]]
    assert sources == []


@pytest.mark.parametrize(
    "char",
    [
        pytest.param(c, marks=pytest.mark.xfail(reason=KNOWN_DIFFERENCES[c], strict=True))
        if c in KNOWN_DIFFERENCES
        else c
        for c in KEEP
    ],
)
def test_characteristic_matches_oracle(
    char: str, expected: dict[str, Frame], actual: dict[str, Frame]
) -> None:
    """Each output file's values of one characteristic equal the oracle's, nulls included."""
    report = {}
    for name in FILES:
        want, got = expected[name], actual[name]
        bad = [
            k for k in sorted(want.keys() & got.keys()) if not _same(want[k][char], got[k][char])
        ]
        if bad:
            report[name] = (len(bad), [(*k, want[k][char], got[k][char]) for k in bad[:3]])
    assert not report, report
