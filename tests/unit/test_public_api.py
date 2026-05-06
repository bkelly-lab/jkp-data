"""Tests for the public API surface declared by ``jkp.data.__all__``.

These tests pin the v0.x public API: every name in ``__all__`` must resolve to
something callable/usable, ``from jkp.data import *`` must produce exactly the
intended set of names, and importing the package must NOT eagerly import the
heavy modules (`polars`, `duckdb`, `ibis`) — that's the point of the
module-level ``__getattr__`` in ``__init__.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _clean_env() -> dict[str, str]:
    """Return an environment without coverage subprocess hooks.

    Both pytest-cov and coverage.py have mechanisms to auto-activate coverage
    in subprocesses, which would force an eager import of the measured package
    and defeat the lazy-import assertions below:

      - Older pytest-cov (2.x/3.x) sets ``COV_CORE_*``.
      - Modern coverage.py uses ``COVERAGE_PROCESS_START`` plus a ``.pth`` file
        installed in site-packages; the ``.pth`` only activates when that env
        var is set, so stripping it suppresses subprocess coverage.

    Strip both prefixes so the subprocess Python sees no coverage hooks
    regardless of which tooling configuration the project is using.
    """
    return {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("COV_") or k.startswith("COVERAGE_"))
    }


# Names in __all__ that the package should expose. Updating this list
# alongside `jkp.data.__all__` is intentional — adding a public name should
# require updating both places.
EXPECTED_PUBLIC_API = {
    "__version__",
    "app",
    "DataPaths",
    "run_pipeline",
    "run_portfolio",
}


@pytest.mark.unit
class TestPublicApiSurface:
    """The advertised public API matches `__all__` and resolves correctly."""

    def test_all_matches_expected(self) -> None:
        """`__all__` must be exactly the documented public surface."""
        import jkp.data

        assert set(jkp.data.__all__) == EXPECTED_PUBLIC_API

    def test_each_name_resolves(self) -> None:
        """Every name in `__all__` must resolve to a non-None object."""
        import jkp.data

        for name in jkp.data.__all__:
            value = getattr(jkp.data, name)
            assert value is not None, f"jkp.data.{name} resolved to None"

    def test_star_import_exposes_exactly_all(self) -> None:
        """`from jkp.data import *` exposes exactly the names in __all__.

        We use a fresh subprocess so the global namespace from this test file
        doesn't pollute the assertion. Filter out Python-injected dunders
        (`__builtins__`, `__annotations__`, etc.) that always appear in
        module scope but are not part of our public API; keep `__version__`
        since that one IS in `__all__`.
        """
        cmd = [
            sys.executable,
            "-c",
            "from jkp.data import *\n"
            "import jkp.data\n"
            "names = {n for n in globals() if not n.startswith('__') or n == '__version__'}\n"
            "names.discard('jkp')\n"
            f"expected = set({sorted(EXPECTED_PUBLIC_API)!r})\n"
            "assert names == expected, (\n"
            "    f'star-import mismatch: extras={sorted(names - expected)!r}, '\n"
            "    f'missing={sorted(expected - names)!r}'\n"
            ")\n",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=_clean_env())
        assert result.returncode == 0, (
            f"star-import test failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )

    def test_unknown_attribute_raises(self) -> None:
        """Accessing a name not in `__all__` raises AttributeError as usual."""
        import jkp.data

        with pytest.raises(AttributeError, match="not_a_real_name"):
            jkp.data.not_a_real_name  # noqa: B018

    def test_dir_advertises_public_api_and_hides_internals(self) -> None:
        """`dir(jkp.data)` advertises the public API plus standard dunders only.

        Specifically:
          - every `__all__` name must appear (drives tab-completion for the
            documented public API),
          - standard module dunders like `__file__` / `__doc__` / `__spec__`
            must remain visible (tooling depends on them),
          - submodules (`cli`, `main`, `paths`, `aux_functions`, …) and
            aliased helpers (`_Any`, `_LAZY_ATTRS`) must NOT leak — those
            are implementation detail, not part of the v0.x API contract.
        """
        import jkp.data

        names = set(dir(jkp.data))

        # All public names are advertised.
        assert EXPECTED_PUBLIC_API.issubset(names), (
            f"public API not fully advertised: missing {EXPECTED_PUBLIC_API - names}"
        )
        # Standard module dunders must remain visible.
        for dunder in ("__file__", "__doc__", "__spec__", "__name__", "__package__"):
            assert dunder in names, f"standard module attribute {dunder!r} missing from dir()"
        # No single-underscore-prefixed private names should leak.
        leaked_private = [n for n in names if n.startswith("_") and not n.startswith("__")]
        assert not leaked_private, f"private names leaked into dir(): {leaked_private}"
        # No submodules should leak (they may be bound on the package as a
        # side-effect of the lazy resolver, but they are not public API).
        forbidden_submodules = {
            "aux_functions",
            "cli",
            "config",
            "main",
            "output_writer",
            "paths",
            "portfolio",
            "wrds_credentials",
        }
        leaked_submodules = forbidden_submodules & names
        assert not leaked_submodules, f"submodules leaked into dir(): {sorted(leaked_submodules)}"


@pytest.mark.unit
class TestLazyImportBehavior:
    """Importing `jkp.data` should not eagerly import heavy submodules.

    The point of the module-level ``__getattr__`` in `__init__.py` is to
    keep ``import jkp.data`` cheap. If someone removes the lazy-loader and
    goes back to eager re-exports, this test catches it.
    """

    def _module_imported_after(self, code: str) -> bool:
        """Run `code` in a fresh interpreter and report whether `polars` ended up imported."""
        cmd = [
            sys.executable,
            "-c",
            code + "\nimport sys; print('IMPORTED' if 'polars' in sys.modules else 'ABSENT')",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=_clean_env())
        return "IMPORTED" in result.stdout.split()

    def test_bare_import_does_not_pull_polars(self) -> None:
        """`import jkp.data` alone should not import polars."""
        assert not self._module_imported_after("import jkp.data"), (
            "polars was imported just by `import jkp.data` — the "
            "module-level `__getattr__` in `__init__.py` may have been "
            "bypassed by a new eager re-export"
        )

    def test_version_lookup_does_not_pull_polars(self) -> None:
        """Reading `__version__` should not require importing polars."""
        assert not self._module_imported_after("import jkp.data\nv = jkp.data.__version__"), (
            "polars was imported when reading __version__"
        )

    def test_run_pipeline_access_does_pull_polars(self) -> None:
        """Sanity check: accessing `run_pipeline` SHOULD trigger the heavy import.

        This is the inverse of the above tests — it confirms the lazy resolver
        actually fires when a heavy name is touched. Without this, a buggy
        `__getattr__` that silently returned `None` could pass the negative
        tests.
        """
        assert self._module_imported_after(
            "import jkp.data\nfn = jkp.data.run_pipeline\nassert callable(fn)"
        ), (
            "polars was NOT imported after touching jkp.data.run_pipeline — "
            "the lazy resolver may be returning a bogus value"
        )
