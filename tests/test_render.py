"""Every output format is testable from a frozen result: no model, no torch."""

from __future__ import annotations

import json

from ner_mcp.render import (
    render_highlight,
    render_json,
    render_jsonl,
    render_markdown,
    render_stats,
    render_text,
)
from ner_mcp.schemas import ExtractionResult

SENTENCE = "Sylvain Boyer est à Paris."


def test_json_is_the_wire_schema(sample_result: ExtractionResult) -> None:
    payload = json.loads(render_json(sample_result))
    assert payload["model"] == "CATIE-AQ/NERmemberta-4entities"
    assert [entity["label"] for entity in payload["entities"]] == ["PER", "LOC"]
    assert payload["dropped_by_min_score"] == 2


def test_jsonl_is_one_entity_per_line(sample_result: ExtractionResult) -> None:
    lines = render_jsonl(sample_result).splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "Sylvain Boyer"


def test_text_table_has_a_header_and_aligned_rows(sample_result: ExtractionResult) -> None:
    output = render_text(sample_result)
    lines = output.splitlines()
    assert lines[0].startswith("LABEL")
    assert "SCORE" in lines[0] and "SPAN" in lines[0]
    assert "PER" in lines[2] and "'Sylvain Boyer'" in lines[2]
    assert "[20,25)" in output


def test_text_format_handles_no_entities(empty_result: ExtractionResult) -> None:
    assert render_text(empty_result) == "no entities found"


def test_highlight_without_colour_marks_labels_inline(sample_result: ExtractionResult) -> None:
    output = render_highlight(sample_result, SENTENCE, color=False, legend=False)
    assert output == "Sylvain Boyer (PER) est à Paris (LOC)."


def test_highlight_with_colour_wraps_spans_and_has_a_legend(
    sample_result: ExtractionResult,
) -> None:
    output = render_highlight(sample_result, SENTENCE, color=True, legend=True)
    assert "\x1b[33mSylvain Boyer\x1b[0m" in output
    assert "\x1b[32mParis\x1b[0m" in output
    assert "personne (1)" in output
    assert "lieu (1)" in output
    assert "\x1b[33mPER\x1b[0m" in output


def test_highlight_skips_overlapping_entities_instead_of_corrupting_text(
    sample_result: ExtractionResult,
) -> None:
    overlapping = sample_result.model_copy(deep=True)
    overlapping.entities[1].start = 5  # inside "Sylvain"
    overlapping.entities[1].end = 9
    output = render_highlight(overlapping, SENTENCE, color=False, legend=False)
    assert output == "Sylvain Boyer (PER) est à Paris."


def test_markdown_has_a_table_and_an_annotated_paragraph(sample_result: ExtractionResult) -> None:
    output = render_markdown(sample_result, SENTENCE)
    assert "| Label | Score | Span | Text |" in output
    assert "**Sylvain Boyer** (PER)" in output


def test_stats_line_carries_diagnostics(sample_result: ExtractionResult) -> None:
    line = render_stats(sample_result)
    assert "chunks=1" in line
    assert "dropped=2" in line
    assert "elapsed=42ms" in line
