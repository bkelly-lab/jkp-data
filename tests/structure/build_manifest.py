"""
Description:
    Build the repository's inventory manifest by static inspection.

    The manifest records what the repo is supposed to contain: pipeline functions
    and their signatures, module import edges, run_pipeline's stage order, the
    rolling/factor dispatch tables, the test inventory, tracked non-code
    artifacts, CHANGELOG_DATA.md entries, and the canonical characteristic names.

    Comparing the manifest against tests/structure/pipeline_structure.txt turns a
    silent deletion into a red test and a reviewable diff line. See issue #254.

Steps:
    1) fn:       every top-level def under src/jkp/data/, with signature.
    2) imp:      relative-import edges between src/jkp/data/ modules.
    3) stage:    run_pipeline's ordered call sequence.
    4) dispatch: the process_window and generate_factor_models lookup tables.
    5) test:     every test function under tests/.
    6) file:     every tracked non-code artifact, by path.
    7) chlog:    every section header in CHANGELOG_DATA.md.
    8) char:     PORTFOLIO_CHARS and acc_chars_list() names.
Output:
    A sorted list of manifest lines; --regen writes them to
    tests/structure/pipeline_structure.txt.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

MANIFEST_PATH = Path("tests/structure/pipeline_structure.txt")

PKG_DIR = "src/jkp/data"

# Entry points. Nothing outside these can pull a module into the pipeline.
ROOT_MODULES = frozenset({"cli", "main", "portfolio", "__init__"})

# Extensions that are code or code-adjacent config. Everything else tracked is an
# artifact whose deletion the `file` section is meant to catch.
CODE_SUFFIXES = frozenset(
    {".py", ".yml", ".yaml", ".toml", ".lock", ".cff", ".typed", ".gitignore"}
)


def tracked_files(repo_root: Path) -> list[str]:
    """
    Description:
        List repo-relative paths of all git-tracked files.
    Steps:
        1) Shell out to `git ls-files`.
    Output:
        Sorted list of repo-relative paths. Uses git rather than a filesystem
        glob so untracked Finder-style " 2.py" duplicates are not counted.
    """
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line)


def _parse(repo_root: Path, rel_path: str) -> ast.Module:
    return ast.parse((repo_root / rel_path).read_text(encoding="utf-8"), filename=rel_path)


def _module_name(rel_path: str) -> str:
    return rel_path.rsplit("/", 1)[-1].removesuffix(".py")


def function_lines(repo_root: Path, paths: list[str]) -> list[str]:
    """
    Description:
        Emit an `fn` line per top-level def under src/jkp/data/, with signature.
    Steps:
        1) Walk each module's AST for function definitions.
        2) Render the argument list via ast.unparse.
    Output:
        List of "fn <module>.<name>(<signature>)" lines. Signatures are included
        so that dropping a parameter — as PR #247 did to gen_comp_dsf — shows up.
    """
    lines = []
    for rel in paths:
        if not (rel.startswith(f"{PKG_DIR}/") and rel.endswith(".py")):
            continue
        mod = _module_name(rel)
        for node in ast.walk(_parse(repo_root, rel)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                lines.append(f"fn {mod}.{node.name}({ast.unparse(node.args)})")
    return lines


def import_edges(repo_root: Path, paths: list[str]) -> list[str]:
    """
    Description:
        Emit an `imp` line per relative-import edge between package modules.
    Steps:
        1) Walk each module for ImportFrom nodes with a relative level.
    Output:
        List of "imp <module> -> <imported>" lines. ast.walk (not iteration over
        module.body) is required: portfolio.py imports inside run_portfolio to
        bypass its module __getattr__, and those edges are real.
    """
    lines = []
    for rel in paths:
        if not (rel.startswith(f"{PKG_DIR}/") and rel.endswith(".py")):
            continue
        mod = _module_name(rel)
        for node in ast.walk(_parse(repo_root, rel)):
            if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
                lines.append(f"imp {mod} -> {node.module.split('.')[0]}")
    return lines


def reachable_modules(edges: list[str], present: set[str]) -> set[str]:
    """
    Description:
        Resolve which package modules are reachable from the entry points.
    Steps:
        1) Build an adjacency map from `imp` lines.
        2) Walk outward from ROOT_MODULES.
    Output:
        Set of reachable module names. A module outside this set is orphaned —
        still present and still passing its own unit tests, but never called.
    """
    graph: dict[str, set[str]] = {}
    for line in edges:
        src, dst = line.removeprefix("imp ").split(" -> ")
        graph.setdefault(src, set()).add(dst)

    seen: set[str] = set()
    stack = sorted(ROOT_MODULES & present)
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        stack.extend(d for d in graph.get(mod, ()) if d in present and d not in seen)
    return seen


def stage_lines(repo_root: Path) -> list[str]:
    """
    Description:
        Emit a `stage` line per call in run_pipeline, in order.
    Steps:
        1) Locate run_pipeline in main.py.
        2) Record each top-level expression-statement call, numbered.
    Output:
        List of "stage <nnn> <name>" lines. The index pins order, so a stage
        being dropped or resequenced is visible.
    """
    tree = _parse(repo_root, f"{PKG_DIR}/main.py")
    lines = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "run_pipeline"):
            continue
        idx = 0
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name:
                    lines.append(f"stage {idx:03d} {name}")
                    idx += 1
    return lines


def dispatch_lines(repo_root: Path) -> list[str]:
    """
    Description:
        Emit a `dispatch` line per entry of the indirect-dispatch tables.
    Steps:
        1) Find process_window's `funcs` dict in aux_functions.py.
        2) Find generate_factor_models' generators dict in main.py.
    Output:
        List of "dispatch <table> <key> -> <function>" lines. These dicts are the
        real wiring for rolling stats and factor models; a name reached only
        through them is invisible to a plain call-graph read.
    """
    lines = []
    specs = [
        (f"{PKG_DIR}/aux_functions.py", "process_window", "process_window.funcs"),
        (f"{PKG_DIR}/main.py", "generate_factor_models", "generate_factor_models"),
    ]
    for rel, fn_name, label in specs:
        tree = _parse(repo_root, rel)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.FunctionDef) and node.name == fn_name):
                continue
            for dct in ast.walk(node):
                if not isinstance(dct, ast.Dict):
                    continue
                for key, val in zip(dct.keys, dct.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(val, ast.Name):
                        lines.append(f"dispatch {label} {key.value} -> {val.id}")
    return lines


def test_lines(repo_root: Path, paths: list[str]) -> list[str]:
    """
    Description:
        Emit a `test` line per test function under tests/.
    Steps:
        1) Walk each tests/*.py module for defs named test_*.
    Output:
        List of "test <name>" lines. This is what catches a test being deleted
        alongside the code it guards — the PR #247 test_nulls_return_above_10x
        case, which no test suite can catch on its own.
    """
    lines = []
    for rel in paths:
        if not (rel.startswith("tests/") and rel.endswith(".py")):
            continue
        for node in ast.walk(_parse(repo_root, rel)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                lines.append(f"test {node.name}")
    return lines


def artifact_lines(paths: list[str]) -> list[str]:
    """
    Description:
        Emit a `file` line per tracked non-code artifact.
    Steps:
        1) Filter tracked paths to those without a code suffix.
        2) Skip the manifest itself, which would never reach a fixed point.
    Output:
        List of "file <path>" lines. Names only, no content hash: the goal is to
        catch a drop, not to police edits, and hashing would fire on every doc
        re-render. Covers the golden fixtures, which stage_synthetic_slices skips
        silently when absent.
    """
    manifest = MANIFEST_PATH.as_posix()
    return [f"file {p}" for p in paths if Path(p).suffix not in CODE_SUFFIXES and p != manifest]


def changelog_lines(repo_root: Path) -> list[str]:
    """
    Description:
        Emit a `chlog` line per section header in CHANGELOG_DATA.md.
    Steps:
        1) Take each line starting with '## ', stripped.
    Output:
        List of "chlog <header>" lines. Header text is captured verbatim and not
        parsed: the format is not uniform — the oldest entries are MM-DD-YYYY
        (02-19-2021 has no month 19), some carry trailing space or [tags].
    """
    text = (repo_root / "CHANGELOG_DATA.md").read_text(encoding="utf-8")
    return [f"chlog {ln[3:].strip()}" for ln in text.splitlines() if ln.startswith("## ")]


def _literal_list(tree: ast.Module, name: str) -> list[str] | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


def char_lines(repo_root: Path) -> list[str]:
    """
    Description:
        Emit a `char` line per canonical characteristic name.
    Steps:
        1) Read PORTFOLIO_CHARS from config.py.
        2) Read acc_chars_list()'s literal from aux_functions.py.
    Output:
        List of "char <kind> <name>" lines. Both are pure literals, so they are
        read via ast.literal_eval without importing jkp.data. Names are pinned,
        never counts: a count tells a reviewer nothing about which one vanished.
    """
    lines = []
    cfg = _parse(repo_root, f"{PKG_DIR}/config.py")
    for name in _literal_list(cfg, "PORTFOLIO_CHARS") or []:
        lines.append(f"char portfolio {name}")

    aux = _parse(repo_root, f"{PKG_DIR}/aux_functions.py")
    for node in ast.walk(aux):
        if isinstance(node, ast.FunctionDef) and node.name == "acc_chars_list":
            for name in (
                _literal_list(ast.Module(body=node.body, type_ignores=[]), "acc_chars") or []
            ):
                lines.append(f"char accounting {name}")
    return lines


def build_manifest(repo_root: Path) -> list[str]:
    """
    Description:
        Derive the repository's full inventory manifest.
    Steps:
        1) Enumerate tracked files.
        2) Collect every section.
    Output:
        Sorted, de-duplicated list of manifest lines.
    """
    paths = tracked_files(repo_root)
    lines = [
        *function_lines(repo_root, paths),
        *import_edges(repo_root, paths),
        *stage_lines(repo_root),
        *dispatch_lines(repo_root),
        *test_lines(repo_root, paths),
        *artifact_lines(paths),
        *changelog_lines(repo_root),
        *char_lines(repo_root),
    ]
    return sorted(set(lines))


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(out)


def main() -> int:
    root = repo_root()
    lines = build_manifest(root)
    if "--regen" in sys.argv:
        target = root / MANIFEST_PATH
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {len(lines)} lines to {MANIFEST_PATH}")
    else:
        print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
