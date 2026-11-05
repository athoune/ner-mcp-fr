"""Rendering of extraction results.

Kept separate from the CLI so every format is testable against a frozen `ExtractionResult`,
without loading a model.
"""

from __future__ import annotations

from .schemas import ExtractionResult

# ANSI SGR codes, one colour per label.
LABEL_COLORS: dict[str, str] = {"PER": "33", "LOC": "32", "ORG": "34", "MISC": "35"}
LABEL_GLOSS: dict[str, str] = {
    "PER": "personne",
    "LOC": "lieu",
    "ORG": "organisation",
    "MISC": "divers",
}


def render_json(result: ExtractionResult) -> str:
    """The wire schema (§9 of SPECS.md), stable and complete."""
    return result.model_dump_json(indent=2)


def render_jsonl(result: ExtractionResult) -> str:
    """One entity per line, for `jq` and friends. Metadata belongs on stderr."""
    return "\n".join(entity.model_dump_json() for entity in result.entities)


def render_text(result: ExtractionResult) -> str:
    """Aligned human-readable table."""
    if not result.entities:
        return "no entities found"
    rows = [
        (entity.label, f"{entity.score:.4f}", f"[{entity.start},{entity.end})", repr(entity.text))
        for entity in result.entities
    ]
    header = ("LABEL", "SCORE", "SPAN", "TEXT")
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]
    lines = []
    for index, row in enumerate([header, *rows]):
        cells = [cell.ljust(widths[column]) for column, cell in enumerate(row)]
        lines.append("  ".join(cells).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def render_highlight(
    result: ExtractionResult,
    source: str,
    *,
    color: bool = True,
    legend: bool = True,
) -> str:
    """The original text with entities marked in place: the fastest way to eyeball boundaries."""
    pieces: list[str] = []
    cursor = 0
    for entity in sorted(result.entities, key=lambda e: (e.start, e.end)):
        if entity.start < cursor or entity.end > len(source):
            continue  # defensive: never emit misaligned markup
        pieces.append(source[cursor : entity.start])
        span = source[entity.start : entity.end]
        if color:
            pieces.append(f"\x1b[{LABEL_COLORS[entity.label]}m{span}\x1b[0m")
        else:
            pieces.append(f"{span} ({entity.label})")
        cursor = entity.end
    pieces.append(source[cursor:])
    body = "".join(pieces)

    if not legend or not result.entities:
        return body

    present = [label for label in LABEL_COLORS if any(e.label == label for e in result.entities)]
    lines = [body.rstrip("\n"), ""]
    for label in present:
        count = sum(1 for e in result.entities if e.label == label)
        tag = f"\x1b[{LABEL_COLORS[label]}m{label}\x1b[0m" if color else label
        lines.append(f"  {tag}  {LABEL_GLOSS[label]} ({count})")
    return "\n".join(lines)


def render_markdown(result: ExtractionResult, source: str | None = None) -> str:
    """Table plus an annotated paragraph, ready to paste into an issue."""
    lines: list[str] = []
    if result.entities:
        lines.extend(["| Label | Score | Span | Text |", "|---|---|---|---|"])
        for entity in result.entities:
            lines.append(
                f"| {entity.label} | {entity.score:.4f} | [{entity.start},{entity.end}) "
                f"| `{entity.text}` |"
            )
    else:
        lines.append("_No entities found._")

    if source is not None and result.entities:
        pieces: list[str] = []
        cursor = 0
        for entity in sorted(result.entities, key=lambda e: (e.start, e.end)):
            if entity.start < cursor:
                continue
            pieces.append(source[cursor : entity.start])
            pieces.append(f"**{source[entity.start:entity.end]}** ({entity.label})")
            cursor = entity.end
        pieces.append(source[cursor:])
        lines.extend(["", "".join(pieces)])

    return "\n".join(lines)


def render_stats(result: ExtractionResult) -> str:
    """One diagnostic line, meant for stderr."""
    return (
        f"model={result.model} profile={result.profile} device={result.device} "
        f"chunks={result.chunks} chars={result.text_length} "
        f"entities={len(result.entities)} dropped={result.dropped_by_min_score} "
        f"truncated={str(result.truncated).lower()} elapsed={result.elapsed_ms}ms"
    )
