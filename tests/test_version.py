"""The version printed by the CLI must be the version that was published."""

from __future__ import annotations

from importlib.metadata import version

import ner_mcp


def test_version_comes_from_the_package_metadata() -> None:
    # A literal in __init__.py silently drifts from pyproject.toml; this is the guard.
    assert ner_mcp.__version__ == version("ner-mcp-fr")


def test_version_is_not_a_placeholder() -> None:
    assert ner_mcp.__version__ != "0.0.0+unknown"
    assert ner_mcp.__version__.count(".") >= 2
