"""
Mutation check of the accounting-stage tests (issue #303, step 6). Not part of CI.

Each mutation replaces one snippet of src/jkp/data/aux_functions.py with a plausible bug. For each
one the script copies src/, tests/ and pyproject.toml to a temporary folder, applies the mutation
there and runs the accounting-stage tests on the copy. A mutation is killed when a test fails,
crashed when only an exception stops a test, and survived when no test notices it. A crash counts
as caught, since CI fails on it too, but is reported apart. The working tree is never changed. The
unmutated copy runs first and must pass.

Run on Yale from the repository root (every mutation runs the stage several times):
    srun --cpus-per-task=8 --mem=32G .venv/bin/python scripts/mutation_check_acc_chars.py
or split across a Slurm array, each task checking the baseline and every tenth mutation:
    sbatch --array=0-9 --cpus-per-task=8 --mem=16G --wrap \
        '.venv/bin/python scripts/mutation_check_acc_chars.py --shard $SLURM_ARRAY_TASK_ID/10'
Options: -k NAME runs the mutations whose name contains NAME; --shard K/N runs every Nth mutation
from the Kth; --check only checks that every snippet occurs exactly once and that every mutated
file still compiles.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = Path("src/jkp/data/aux_functions.py")
TESTS = [
    "tests/unit/test_acc*.py",
    "tests/unit/test_earnings_persistence.py",
    "tests/unit/test_expressions.py",
    "tests/unit/test_scaling_ratios.py",
    "tests/property/test_acc_*.py",
    "tests/golden/test_acc_chars_golden.py",
    "tests/integration/test_accounting_stage_pipeline.py",
]

# (name, snippet of TARGET, mutated snippet)
MUTATIONS: list[tuple[str, str, str]] = [
    # source, format and fiscal-calendar choice
    (
        "prefer_global annual: NA row wins over Global",
        'prefer_global(__gfunda, __funda, ["gvkey", "datadate"])',
        'prefer_global(__funda, __gfunda, ["gvkey", "datadate"])',
    ),
    (
        "prefer_global quarterly: key is (gvkey, datadate) not the fiscal period",
        'prefer_global(__gfundq, __fundq, ["gvkey", "fyr", "fyearq", "fqtr"])',
        'prefer_global(__gfundq, __fundq, ["gvkey", "datadate"])',
    ),
    (
        "apply_indfmt_filter: FS-only keys dropped",
        'return df.filter(is_indl | ~is_indl.any().over(keys)).drop("indfmt")',
        'return df.filter(is_indl).drop("indfmt")',
    ),
    (
        "global quarterly indfmt chosen across calendars (no fyr in key)",
        '.pipe(apply_indfmt_filter, ["gvkey", "datadate", "fyr"])',
        '.pipe(apply_indfmt_filter, ["gvkey", "datadate"])',
    ),
    (
        "one_row_per_date: most recent calendar run wins",
        "descending=[False, False, True, False, False, False],",
        "descending=[False, False, False, False, False, False],",
    ),
    (
        "one_row_per_date: lowest fyr wins the tie",
        "descending=[False, False, True, False, False, False],",
        "descending=[False, False, True, True, False, False],",
    ),
    (
        "one_row_per_date: every row starts a new run",
        "> CALENDAR_RUN_MAX_GAP_DAYS).fill_null(",
        "> 0).fill_null(",
    ),
    # quarterize and cumulate_4q
    (
        "quarterize: running count per (gvkey, fyr) instead of per fiscal year",
        'col("gvkey").cum_count().over(["gvkey", "fyr", "fyearq"])',
        'col("gvkey").cum_count().over(["gvkey", "fyr"])',
    ),
    (
        "quarterize: first obs of a year at fqtr 2 not deleted",
        'c1 = (col("fqtr") != 1)',
        'c1 = (col("fqtr") > 2)',
    ),
    (
        "quarterize: rows without fqtr kept",
        'df.filter(col("fqtr").is_not_null())',
        "df.filter(pl.lit(True))",
    ),
    (
        "cumulate_4q: no fyr continuity check",
        '(pl.col("fyr") == pl.col("fyr").shift(3))',
        "pl.lit(True)",
    ),
    (
        "cumulate_4q: year-to-date fallback at fqtr 3",
        'pl.when(pl.col("fqtr") == 4).then(pl.col(f"{var[:-1]}y")),',
        'pl.when(pl.col("fqtr") == 3).then(pl.col(f"{var[:-1]}y")),',
    ),
    (
        "ttm: lag 2 replaced by lag 1",
        "col(var).shift(2) + col(var).shift(3)",
        "col(var).shift(1) + col(var).shift(3)",
    ),
    (
        "quarterly: y_q preferred over reported q",
        'pl.coalesce([f"{var[:-1]}q", f"{var[:-1]}y_q"]).alias(f"{var[:-1]}q")',
        'pl.coalesce([f"{var[:-1]}y_q", f"{var[:-1]}q"]).alias(f"{var[:-1]}q")',
    ),
    # FX conversion
    (
        "FX: divide instead of multiply",
        '[(col(var) * col("fx")).alias(var) for var in vars] + [pl.lit("USD").alias(fx_var)]',
        '[(col(var) / col("fx")).alias(var) for var in vars] + [pl.lit("USD").alias(fx_var)]',
    ),
    (
        "FX: quarterly year-to-date vars not converted",
        '__compq = add_fx_and_convert_vars(qname, fx, qvars, "quarterly")',
        '__compq = add_fx_and_convert_vars(qname, fx, qvars_q, "quarterly")',
    ),
    (
        "convert_raw_vars_to_usd: net_income not converted",
        'for i in ["assets", "sales", "book_equity", "net_income"]',
        'for i in ["assets", "sales", "book_equity"]',
    ),
    # public-date expansion and ME join
    (
        "public-date lag 4 -> 3 months",
        '.with_columns(start_date=col("datadate").dt.offset_by(f"{lag_to_pub}mo").dt.month_end())',
        '.with_columns(start_date=col("datadate").dt.offset_by(f"{lag_to_pub - 1}mo").dt.month_end())',
    ),
    (
        "public-date max lag 18 -> 17 months",
        '(col("datadate").dt.offset_by(f"{max_lag}mo").dt.month_end()),',
        '(col("datadate").dt.offset_by(f"{max_lag - 1}mo").dt.month_end()),',
    ),
    (
        "public-date next start taken across gvkeys",
        'next_start_date=col("start_date").shift(-1).over(["gvkey"])',
        'next_start_date=col("start_date").shift(-1)',
    ),
    (
        "ME joined at the previous month-end",
        'left_on=["gvkey", "public_date"],',
        'left_on=["gvkey", col("public_date").dt.offset_by("-1mo").dt.month_end()],',
    ),
    # history gates and lags
    (
        "capex_abn gate count > 36 -> >= 36",
        'c2 = col("count") > 36',
        'c2 = col("count") >= 36',
    ),
    # ni_inc8q's count >= 33 gate is not mutated: eight increases need four-quarter NI sums at
    # twelve dates three months apart, and a firm has those 39 months after its first report at
    # the earliest, so no gate up to 40 months changes the output.
    (
        "ni_inc8q needs 8 non-null increases -> 7",
        '(col("n_ni_inc") == 8)',
        '(col("n_ni_inc") >= 7)',
    ),
    (
        "standardized surprise gate count > 36 -> >= 36",
        'pl.when(col("count") > (12 + qtrs * 3))',
        'pl.when(col("count") >= (12 + qtrs * 3))',
    ),
    (
        "standardized surprise mean lag 3 -> 1",
        'col("__chg_mean").shift(3)',
        'col("__chg_mean").shift(1)',
    ),
    (
        "volq gate count > 45 -> >= 45",
        'pl.when((col("count") > ((qtrs - 1) * 3)) & (col("__n") >= qtrs_min))',
        'pl.when((col("count") >= ((qtrs - 1) * 3)) & (col("__n") >= qtrs_min))',
    ),
    (
        "roe_be_std window 5 -> 4 years",
        "t1_col = [16, 16, 20, 5]",
        "t1_col = [16, 16, 20, 4]",
    ),
    (
        "chg_to_lagassets scaled by current assets",
        'name_gr_exp = (col(var_gr) - col(var_gr).shift(12)) / col("at_x").shift(12)',
        'name_gr_exp = (col(var_gr) - col(var_gr).shift(12)) / col("at_x")',
    ),
    (
        "aliq_mat without same-gvkey check on the lag",
        'pl.when((col("gvkey") == col("gvkey").shift(12)) & (col("mat").shift(12) != 0))',
        'pl.when(col("mat").shift(12) != 0)',
    ),
    (
        "nonpositive mat kept",
        'mat=pl.when(col("mat") > 0).then(col("mat")).otherwise(fl_none()),',
        'mat=col("mat"),',
    ),
    (
        "emp_gr1 computed on the quarterly file instead of the annual",
        'if Path(path).name == "acc_std_qtr.parquet":',
        'if Path(path).name == "acc_std_ann.parquet":',
    ),
    # safe_div modes, sum_sas/sub_sas, coalesce order in add_helper_vars
    (
        "safe_div mode 3: den != 0 instead of den > 0",
        "pl.when(col(den) > 0).then(col(num) / col(den)).otherwise(fl_none()).alias(name)",
        "pl.when(col(den) != 0).then(col(num) / col(den)).otherwise(fl_none()).alias(name)",
    ),
    (
        "safe_div mode 5: fx not applied",
        '.then(col(num) * col("fx") / col(den))',
        ".then(col(num) / col(den))",
    ),
    (
        "sum_sas: null propagates instead of null-as-zero",
        ".then(pl.coalesce([col1, 0.0]) + pl.coalesce([col2, 0.0]))",
        ".then(col(col1) + col(col2))",
    ),
    (
        "sub_sas: null propagates instead of null-as-zero",
        ".then(pl.coalesce([col1, 0.0]) - pl.coalesce([col2, 0.0]))",
        ".then(col(col1) - col(col2))",
    ),
    (
        "sale_x coalesce order revt before sale",
        'sale_x=pl.coalesce(["sale", "revt"]),',
        'sale_x=pl.coalesce(["revt", "sale"]),',
    ),
    (
        "txditc_x coalesce order txdb+itcb before txditc",
        'txditc_x=pl.coalesce(["txditc", sum_sas("txdb", "itcb")]),',
        'txditc_x=pl.coalesce([sum_sas("txdb", "itcb"), "txditc"]),',
    ),
    (
        "be_x adds preferred stock instead of subtracting",
        '- pl.coalesce([col("pstk_x"), pl.lit(0)])',
        '+ pl.coalesce([col("pstk_x"), pl.lit(0)])',
    ),
    # earnings persistence
    (
        "earnings persistence min obs 5 -> 4",
        "compute_earnings_persistence(paths, data_path, 5, 5)",
        "compute_earnings_persistence(paths, data_path, 5, 4)",
    ),
    (
        "ni_ivol uses population std",
        "stddev_samp(resid ORDER BY datadate) END AS ni_ivol,",
        "stddev_pop(resid ORDER BY datadate) END AS ni_ivol,",
    ),
    # combine_ann_qtr_chars precedence
    (
        "combine: quarterly wins ties on datadate",
        "q.datadate > a.datadate)",
        "q.datadate >= a.datadate)",
    ),
    (
        "combine: quarterly-only months dropped (LEFT JOIN)",
        "FULL OUTER JOIN read_parquet('{qtr_df_path.as_posix()}') AS q",
        "LEFT JOIN read_parquet('{qtr_df_path.as_posix()}') AS q",
    ),
    # formula constants and signs
    (
        "ohlson: working-capital term sign flipped",
        '- 1.43 * col("__o_wc")',
        '+ 1.43 * col("__o_wc")',
    ),
    (
        "altman: ME scaled by assets instead of liabilities",
        'safe_div("me_fiscal", "lt", "__z_me", 3)',
        'safe_div("me_fiscal", "at_x", "__z_me", 3)',
    ),
    (
        "kz: dividend term sign flipped",
        '- 39.368 * col("__kz_dv")',
        '+ 39.368 * col("__kz_dv")',
    ),
    (
        "pitroski: accruals sign flipped",
        '__f_acc=col("__f_croa") - col("__f_roa"),',
        '__f_acc=col("__f_roa") - col("__f_croa"),',
    ),
    (
        "eq_dur: terminal constant (1+r)/r -> 1/r",
        "ed_constant=(pl.lit(horizon) + ((1 + r) / r)),",
        "ed_constant=(pl.lit(horizon) + (1 / r)),",
    ),
]


def _copy(root: Path) -> dict[str, str]:
    """Copy the code and tests under root; return the environment that imports that copy."""
    for part in ("src", "tests"):
        shutil.copytree(REPO / part, root / part, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(REPO / "pyproject.toml", root)
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}


def _pytest(root: Path, env: dict[str, str], workers: int) -> tuple[int, str]:
    """pytest's exit code and its last output line, e.g. '1 failed, 144 passed in 28s'."""
    tests = sorted(str(p.relative_to(root)) for pattern in TESTS for p in root.glob(pattern))
    args = ["-x", "-q", "-p", "no:cacheprovider", "--no-cov", "-n", str(workers), *tests]
    command = [sys.executable, "-m", "pytest", *args]
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    return result.returncode, lines[-1].strip("= ") if lines else ""


def _status(code: int, summary: str) -> str:
    """
    Description:
        Classify one run. With xdist, -x exits with code 2 (interrupted) after the first
        failure, so the summary line, not the exit code, tells a failure from other stops.
    Steps:
        1) Exit 0: every test passed, so the mutation survived.
        2) A failed test: killed. Only an error (an exception in a fixture, such as the stage
           raising): crashed. Anything else: the exit code.
    Output:
        "SURVIVED", "killed", "crashed" or "exit <code>".
    """
    if code == 0:
        return "SURVIVED"
    if " failed" in f" {summary}":
        return "killed"
    return "crashed" if " error" in f" {summary}" else f"exit {code}"


def run(old: str, new: str, workers: int) -> tuple[int, str, float]:
    """
    Description:
        Run the accounting tests on a copy of the code with old replaced by new.
    Steps:
        1) Copy src/, tests/ and pyproject.toml to a temporary folder and apply the mutation.
        2) Check that Python imports aux_functions from the copy.
        3) Run TESTS with pytest, stopping at the first failure.
    Output:
        (pytest's exit code, its summary line, seconds taken).
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = _copy(root)
        target = root / TARGET
        target.write_text(target.read_text().replace(old, new, 1))
        probe = [sys.executable, "-c", "import jkp.data.aux_functions as a; print(a.__file__)"]
        imported = subprocess.run(probe, env=env, capture_output=True, text=True, check=True)
        assert Path(imported.stdout.strip()) == target, imported.stdout
        start = time.monotonic()
        code, summary = _pytest(root, env, workers)
        return code, summary, time.monotonic() - start


def check_snippets() -> list[str]:
    """Names of the mutations whose snippet does not occur exactly once, does not change it, or
    leaves a file that doesn't compile (a syntax error would crash every test)."""
    text = (REPO / TARGET).read_text()
    bad = []
    for name, old, new in MUTATIONS:
        if text.count(old) != 1 or old == new:
            bad.append(name)
            continue
        try:
            compile(text.replace(old, new, 1), str(TARGET), "exec")
        except SyntaxError:
            bad.append(name)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("-k", default="", help="run the mutations whose name contains this")
    parser.add_argument("--check", action="store_true", help="only check the snippets")
    parser.add_argument("--workers", type=int, default=4, help="pytest-xdist workers")
    parser.add_argument("--shard", default="0/1", help="K/N: run every Nth mutation from the Kth")
    args = parser.parse_args()
    shard, shards = map(int, args.shard.split("/"))
    if bad := check_snippets():
        sys.exit(f"mutations with a missing, repeated or non-compiling snippet: {bad}")
    if args.check:
        print(f"{len(MUTATIONS)} snippets each occur once and compile")
        return
    code, summary, seconds = run("", "", args.workers)
    print(f"baseline  {seconds:4.0f}s  {summary}", flush=True)
    if code != 0:
        sys.exit("the unmutated tests fail")
    survivors, crashed, killed = [], [], 0
    for i, (name, old, new) in enumerate(MUTATIONS):
        if args.k not in name or i % shards != shard:
            continue
        code, summary, seconds = run(old, new, args.workers)
        status = _status(code, summary)
        print(f"{status:8}  {seconds:4.0f}s  {name}  [{summary}]", flush=True)
        if status == "killed":
            killed += 1
        elif status == "crashed":
            crashed.append(name)
        else:
            survivors.append(name)
    print(f"{killed} killed by a failed test, {len(crashed)} crashed: {crashed}")
    print(f"{len(survivors)} not killed: {survivors}")
    sys.exit(1 if survivors else 0)


if __name__ == "__main__":
    main()
