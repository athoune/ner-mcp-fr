"""Windowing must cover the text exactly and never truncate silently."""

from __future__ import annotations

from itertools import pairwise

from ner_mcp.chunking import sentence_spans, split_windows


def count_words(text: str) -> int:
    """Cheap stand-in for a tokenizer: one token per word."""
    return len(text.split())


def test_sentence_spans_cover_the_text() -> None:
    text = "Première phrase. Deuxième phrase !\n\nNouveau paragraphe."
    spans = sentence_spans(text)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    assert all(text[a:b].strip() for a, b in spans)
    for (_, end), (start, _) in pairwise(spans):
        assert end == start


def test_short_text_is_a_single_window() -> None:
    text = "Sylvain Boyer est à Paris."
    windows = split_windows(text, count_words, max_tokens=512)
    assert len(windows) == 1
    assert windows[0].start == 0
    assert windows[0].end == len(text)
    assert windows[0].text == text


def test_long_text_is_split_and_every_window_is_an_exact_slice() -> None:
    sentence = "Le dévoilement du logo officiel des JO a eu lieu à Paris en 1924. "
    text = sentence * 40
    windows = split_windows(text, count_words, max_tokens=40)

    assert len(windows) > 1
    for window in windows:
        assert text[window.start : window.end] == window.text
        assert count_words(window.text) <= 40

    # Coverage: no character between the first and last entity is left unseen.
    assert windows[0].start == 0
    assert windows[-1].end == len(text)
    for previous, following in pairwise(windows):
        assert following.start <= previous.end, "windows must overlap or touch, never gap"


def test_overlap_makes_boundary_entities_visible_whole() -> None:
    sentences = [f"Phrase numéro {index} sur Paris." for index in range(20)]
    text = " ".join(sentences)
    windows = split_windows(text, count_words, max_tokens=60)
    overlapping = sum(1 for a, b in pairwise(windows) if b.start < a.end)
    assert overlapping >= 1


def test_a_single_oversized_sentence_is_hard_split_on_whitespace() -> None:
    text = " ".join(f"mot{index}" for index in range(200))
    windows = split_windows(text, count_words, max_tokens=12)

    assert len(windows) > 1
    assert "".join(window.text for window in windows) == text
    # Every window boundary lands between words, never inside one.
    for window in windows[:-1]:
        assert text[window.end - 1 : window.end + 1] == " m" or text[window.end] == " "


def test_empty_text_has_no_window() -> None:
    assert split_windows("", count_words, max_tokens=512) == []
