"""Sentence-aware windowing so long texts are never silently truncated.

`max_position_embeddings` is 512 or 1024 tokens depending on the model: less than a newspaper
paragraph. Splitting is therefore mandatory, and the only hard requirement is that every window
is an exact substring of the original text, so offsets can be rebased by addition.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

# A boundary is a run of whitespace after sentence punctuation, or a blank line.
_BREAK = re.compile(r"(?<=[.!?…])\s+|\n{2,}")

# Safety margin, in tokens: a sentence tokenized alone costs slightly fewer tokens than the same
# sentence in context (BPE merges across boundaries), and special tokens need room.
DEFAULT_MARGIN = 32


@dataclass(frozen=True)
class Window:
    """A contiguous slice of the input, with absolute character offsets."""

    start: int
    end: int
    text: str

    @property
    def length(self) -> int:
        return self.end - self.start


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Split `text` into contiguous spans, keeping punctuation and trailing space inside."""
    bounds = [0]
    bounds.extend(match.end() for match in _BREAK.finditer(text))
    bounds.append(len(text))
    return [(a, b) for a, b in pairwise(bounds) if b > a]


def split_windows(
    text: str,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    *,
    margin: int = DEFAULT_MARGIN,
    overlap_sentences: int = 1,
) -> list[Window]:
    """Cover `text` with windows that each fit the model context.

    Windows overlap by `overlap_sentences` so an entity sitting on a boundary is seen whole by
    at least one window; callers deduplicate afterwards.
    """
    if not text:
        return []

    budget = max(max_tokens - margin, 8)

    # Fast path: the common case is a text that fits in one window.
    if count_tokens(text) <= budget:
        return [Window(0, len(text), text)]

    spans = sentence_spans(text)
    windows: list[Window] = []
    index = 0
    while index < len(spans):
        first = spans[index][0]
        tokens = 0
        cursor = index
        while cursor < len(spans):
            n_tokens = count_tokens(text[spans[cursor][0] : spans[cursor][1]])
            if tokens + n_tokens > budget:
                break
            tokens += n_tokens
            cursor += 1

        if cursor == index:
            # One sentence alone exceeds the context: split it, there is no gentler cut.
            windows.extend(
                _hard_split(text, spans[index][0], spans[index][1], count_tokens, budget)
            )
            index += 1
            continue

        end = spans[cursor - 1][1]
        windows.append(Window(first, end, text[first:end]))
        if cursor >= len(spans):
            break
        # Guarantee progress: when a window holds a single sentence there is nothing to
        # overlap with, and re-using it would loop forever.
        index = max(index + 1, cursor - overlap_sentences)

    return windows


def _hard_split(
    text: str,
    start: int,
    end: int,
    count_tokens: Callable[[str], int],
    budget: int,
) -> list[Window]:
    """Cut a single oversized span, preferring whitespace boundaries."""
    windows: list[Window] = []
    cursor = start
    while cursor < end:
        lo, hi = cursor + 1, end
        cut = cursor + 1
        while lo <= hi:  # binary search the furthest prefix that fits
            mid = (lo + hi) // 2
            if count_tokens(text[cursor:mid]) <= budget:
                cut = mid
                lo = mid + 1
            else:
                hi = mid - 1

        if cut < end:
            space = text.rfind(" ", cursor + 1, cut)
            if space > cursor:
                cut = space

        windows.append(Window(cursor, cut, text[cursor:cut]))
        cursor = cut

    return windows
