"""JKP Factor Data generation pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jkp-data")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"
