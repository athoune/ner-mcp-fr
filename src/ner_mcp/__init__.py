"""Single source of truth for the version: the installed package metadata.

Keeping a literal here as well as in `pyproject.toml` guarantees they drift — it happened once:
the package was bumped to 0.2.0 while `--version` kept answering 0.1.0. The metadata is written
at build time from `pyproject.toml`, so reading it back cannot disagree.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ner-mcp-fr")
except PackageNotFoundError:  # pragma: no cover - running from a tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
