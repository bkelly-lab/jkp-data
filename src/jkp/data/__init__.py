"""JKP Factor Data generation pipeline."""

# Use underscore-prefixed aliases so the `importlib.metadata` helpers don't
# leak into `dir(jkp.data)` / tab-completion as if they were part of our API.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version
from typing import Any as _Any  # underscore-aliased so it doesn't leak into dir()

# IMPORTANT: `__version__` must be assigned BEFORE `_LAZY_ATTRS` / `__getattr__`
# below. Submodules loaded by the lazy resolver (notably `.cli`) do
# `from . import __version__` at import time; if that name isn't already in
# this module's `__dict__` when the lazy import fires, those submodules will
# observe a partially-initialized package and fail. Keep this assignment at
# the top of the file.
try:
    __version__ = _version("jkp-data")
except _PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"


# Public API surface for the v0.x line. Adding a name here commits to keeping
# it importable from `jkp.data` for the duration of v0.x — bumping these names
# requires a major version bump.
#
# We deliberately keep this list small. The five names below cover:
#   - the package version string (Python convention),
#   - the Typer CLI app object (so future plug-in packages can compose it via
#     `add_typer`),
#   - the `DataPaths` dataclass (extension packages need it to integrate with
#     the pipeline's directory layout),
#   - the two top-level pipeline entry points (so users can drive the pipeline
#     from notebooks / scripts without going through the CLI).
#
# Other internals (per-characteristic helpers in `aux_functions`, output
# writers, credential plumbing) remain private until we have a clearer reason
# to commit them as public — adding a name here is a one-way door for v0.x.
__all__ = [
    "__version__",
    "app",
    "DataPaths",
    "run_pipeline",
    "run_portfolio",
]


# We expose the names above via lazy module-level `__getattr__` (see Python's
# "Customizing module attribute access" in the data model docs) rather than
# eagerly re-exporting them at import time. Eager `from .main import
# run_pipeline` would force `import polars`, `import duckdb`, and `import
# ibis` — a few hundred milliseconds of work — on every `import jkp.data`,
# even for callers that only want `jkp.data.__version__`. With `__getattr__`
# below, the heavy modules are imported only on first access of the name
# they back, and `__version__` lookups stay cheap.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # name -> (submodule, attribute-on-submodule)
    "app": (".cli", "app"),
    "DataPaths": (".paths", "DataPaths"),
    "run_pipeline": (".main", "run_pipeline"),
    "run_portfolio": (".portfolio", "run_portfolio"),
}


def __getattr__(name: str) -> _Any:
    """Resolve `__all__` names lazily via module-level `__getattr__`."""
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name, attr_name = target
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    # Cache on the module so future lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Advertise the public API plus standard module dunders, nothing else.

    `dir()` returns:
      - every name in `__all__` (drives tab-completion for the public API,
        even names not yet resolved by the lazy `__getattr__`),
      - standard module dunders like `__file__` / `__doc__` / `__spec__`
        (tooling depends on them being visible).

    It does NOT return:
      - submodule names (`cli`, `main`, `paths`, …) that get bound on the
        package as a side-effect of imports — those are implementation
        detail, not public API,
      - underscore-private internals (`_LAZY_ATTRS`),
      - aliased helpers (`_Any`, `_version`, `_PackageNotFoundError`).

    Note: the standard CPython REPL's tab-completion may still surface
    submodules via package-path (`__path__`) introspection — that is
    Python's package machinery and is not driven by `__dir__`.
    """
    standard_dunders = {n for n in globals() if n.startswith("__") and n.endswith("__")}
    return sorted(set(__all__) | standard_dunders)
