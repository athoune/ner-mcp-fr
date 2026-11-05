"""Shared fixtures. No model is loaded in the unit test suite."""

from __future__ import annotations

import pytest

from ner_mcp.schemas import Entity, ExtractionResult

SENTENCE = "Sylvain Boyer est à Paris."


@pytest.fixture
def sample_result() -> ExtractionResult:
    return ExtractionResult(
        model="CATIE-AQ/NERmemberta-4entities",
        profile="fast",
        device="cpu",
        entities=[
            Entity(text="Sylvain Boyer", label="PER", start=0, end=13, score=0.9991),
            Entity(text="Paris", label="LOC", start=20, end=25, score=0.9987),
        ],
        text_length=len(SENTENCE),
        chunks=1,
        truncated=False,
        dropped_by_min_score=2,
        elapsed_ms=42,
    )


@pytest.fixture
def empty_result() -> ExtractionResult:
    return ExtractionResult(
        model="CATIE-AQ/NERmemberta-4entities",
        profile="fast",
        device="cpu",
        entities=[],
        text_length=len(SENTENCE),
        chunks=1,
        truncated=False,
        dropped_by_min_score=0,
        elapsed_ms=7,
    )
