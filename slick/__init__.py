from __future__ import annotations

try:  # Python 3.8+
    from importlib.metadata import PackageNotFoundError, version
except Exception:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore

try:
    __version__ = version("slick-ai")
except PackageNotFoundError:  # not installed, e.g., running from source without poetry install
    __version__ = "0.0.0"

__all__ = ["__version__"]
