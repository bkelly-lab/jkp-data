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
import functools
import subprocess
import sys
from pathlib import Path

MANIFEST_PATH = Path("tests/structure/pipeline_structure.txt")

PKG_DIR = "src/jkp/data"

# Entry points. Nothing outside these can pull a module into the pipeline.
ROOT_MODULES = frozenset({"cli", "main", "portfolio", "__init__"})


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


@functools.cache
def _parse(repo_root: Path, rel_path: str) -> ast.Module:
    """Parse a tracked module. Cached: aux_functions.py alone is >10k lines and
    several sections read it."""
    return ast.parse((repo_root / rel_path).read_text(encoding="utf-8"), filename=rel_path)


def _qualified_defs(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """
    Description:
        List every function definition with its dotted scope path.
    Steps:
        1) Descend every child node, tracking enclosing class and function names.
    Output:
        List of (dotted_name, node). Nesting is part of the name — two helpers
        called `_inner` in different parents stay distinct, so neither can be
        deleted behind the other under build_manifest's de-duplication.

        Descent passes through non-scoping nodes (if/try/for/with) rather than
        only class and function bodies: aux_functions.concat_one is defined
        inside an `if` block, and stopping at scopes would drop it silently.
    """
    found: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                found.append((f"{prefix}{child.name}", child))
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return found


def _module_name(rel_path: str) -> str:
    return rel_path.rsplit("/", 1)[-1].removesuffix(".py")


def function_lines(repo_root: Path, paths: list[str]) -> list[str]:
    """
    Description:
        Emit an `fn` line per function under src/jkp/data/, with signature.
    Steps:
        1) Collect each module's defs with their dotted scope path.
        2) Render the argument list via ast.unparse.
    Output:
        List of "fn <module>.<scope>.<name>(<signature>)" lines, covering methods
        and nested helpers as well as top-level defs. Signatures are included so
        that dropping a parameter — as PR #247 did to gen_comp_dsf — shows up.
    """
    lines = []
    for rel in paths:
        if not (rel.startswith(f"{PKG_DIR}/") and rel.endswith(".py")):
            continue
        mod = _module_name(rel)
        for dotted, node in _qualified_defs(_parse(repo_root, rel)):
            lines.append(f"fn {mod}.{dotted}({ast.unparse(node.args)})")
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
            if not (isinstance(node, ast.ImportFrom) and node.level > 0):
                continue
            if node.module:
                lines.append(f"imp {mod} -> {node.module.split('.')[0]}")
            else:
                # `from . import aux_functions` (portfolio.py:52) carries the
                # module in the alias, not in node.module.
                for alias in node.names:
                    lines.append(f"imp {mod} -> {alias.name.split('.')[0]}")
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
        being dropped or resequenced is visible. Calls are sorted by source line:
        ast.walk is breadth-first, which would bury a call nested in a loop —
        roll_apply_daily runs mid-pipeline — behind every top-level statement.
    """
    tree = _parse(repo_root, f"{PKG_DIR}/main.py")
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "run_pipeline"):
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func = stmt.value.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name:
                    hits.append((stmt.lineno, name))
    return [f"stage {idx:03d} {name}" for idx, (_, name) in enumerate(sorted(hits))]


def dispatch_lines(repo_root: Path) -> list[str]:
    """
    Description:
        Emit a `dispatch` line per entry of the indirect-dispatch tables.
    Steps:
        1) Locate the named function, raising if it has moved or been renamed.
        2) Read the named dict assigned inside it.
    Output:
        List of "dispatch <fn>.<var> <key> -> <target>" lines. process_window's
        funcs dict is the real wiring for rolling stats — a handler reached only
        through it is invisible to a plain call-graph read.

        A missing target raises rather than emitting nothing: a dispatch spec
        that silently no-ops advertises coverage it does not provide.
    """
    lines = []
    specs = [(f"{PKG_DIR}/aux_functions.py", "process_window", "funcs")]
    for rel, fn_name, var_name in specs:
        tree = _parse(repo_root, rel)
        fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn_name),
            None,
        )
        if fn is None:
            raise LookupError(f"{rel}: dispatch target {fn_name}() not found — update the spec")

        dct = next(
            (
                n.value
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                and isinstance(n.value, ast.Dict)
                and any(getattr(t, "id", None) == var_name for t in n.targets)
            ),
            None,
        )
        if dct is None:
            raise LookupError(f"{rel}: {fn_name}() has no dict named {var_name!r}")

        for key, val in zip(dct.keys, dct.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(val, ast.Name):
                lines.append(f"dispatch {fn_name}.{var_name} {key.value} -> {val.id}")
    return lines


def test_lines(repo_root: Path, paths: list[str]) -> list[str]:
    """
    Description:
        Emit a `test` line per test function under tests/.
    Steps:
        1) Walk each tests/*.py module for defs named test_*.
    Output:
        List of "test <path>::<scope>::<name>" lines, in pytest node-id form.
        This is what catches a test being deleted alongside the code it guards —
        the PR #247 test_nulls_return_above_10x case, which no test suite can
        catch on its own.

        The path and class qualify the name because test names are not unique:
        34 definitions share a name with another, two of them within a single
        file. A bare name would let build_manifest's de-duplication collapse
        them, so deleting one would leave the manifest byte-identical.
    """
    lines = []
    for rel in paths:
        if not (rel.startswith("tests/") and rel.endswith(".py")):
            continue
        for dotted, _ in _qualified_defs(_parse(repo_root, rel)):
            if dotted.rsplit(".", 1)[-1].startswith("test_"):
                lines.append(f"test {rel}::{dotted.replace('.', '::')}")
    return lines


def artifact_lines(paths: list[str]) -> list[str]:
    """
    Description:
        Emit a `file` line per tracked file.
    Steps:
        1) Take every tracked path except the manifest itself.
    Output:
        List of "file <path>" lines. Every tracked file is listed, rather than
        only those failing a suffix blocklist: a blocklist left tracked .py
        outside src/jkp/data/ and tests/ (scripts/, documentation/) covered by no
        section at all, and silently deletable. Listing everything means a file
        cannot fall between sections. The overlap with `fn`/`test` lines for
        Python files is deliberate — they pin contents, this pins existence.

        Names only, no content hash: the goal is catching a drop, not policing
        edits, and hashing would fire on every doc re-render. Covers the golden
        fixtures, which stage_synthetic_slices skips silently when absent.

        The manifest excludes itself: listing it could never reach a fixed point.
    """
    manifest = MANIFEST_PATH.as_posix()
    return [f"file {p}" for p in paths if p != manifest]


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
