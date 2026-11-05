"""Aggregation is the most important contract: text[start:end] must always hold."""

from __future__ import annotations

from ner_mcp.engine import _strip_punctuation, aggregate_entities
from ner_mcp.schemas import Entity

ID2LABEL = {0: "O", 1: "PER", 2: "ORG", 3: "LOC", 4: "MISC"}

TEXT = "Sylvain Boyer est à Paris."


def _tokens() -> tuple[list[tuple[int, int]], list[int], list[int], list[float]]:
    """Word-level predictions: (offsets, word ids, label ids, scores)."""
    words = [
        ("Sylvain", 1, 0.99),
        ("Boyer", 1, 0.98),
        ("est", 0, 0.99),
        ("à", 0, 0.99),
        ("Paris", 3, 0.97),
        (".", 0, 0.99),
    ]
    offsets: list[tuple[int, int]] = []
    word_ids: list[int] = []
    label_ids: list[int] = []
    scores: list[float] = []
    cursor = 0
    for index, (word, label, score) in enumerate(words):
        start = TEXT.index(word, cursor)
        offsets.append((start, start + len(word)))
        word_ids.append(index)
        label_ids.append(label)
        scores.append(score)
        cursor = start + len(word)
    return offsets, word_ids, label_ids, scores


def test_adjacent_same_label_words_are_merged() -> None:
    offsets, word_ids, label_ids, scores = _tokens()
    entities = aggregate_entities(
        text=TEXT,
        offsets=offsets,
        word_ids=word_ids,
        label_ids=label_ids,
        scores=scores,
        id2label=ID2LABEL,
    )
    assert [(e.label, e.text) for e in entities] == [("PER", "Sylvain Boyer"), ("LOC", "Paris")]


def test_merging_can_be_disabled() -> None:
    offsets, word_ids, label_ids, scores = _tokens()
    entities = aggregate_entities(
        text=TEXT,
        offsets=offsets,
        word_ids=word_ids,
        label_ids=label_ids,
        scores=scores,
        id2label=ID2LABEL,
        merge_adjacent=False,
    )
    assert [e.text for e in entities] == ["Sylvain", "Boyer", "Paris"]


def test_offsets_exclude_surrounding_whitespace() -> None:
    # SentencePiece sometimes reports a leading space inside the token span.
    text = " Paris"
    entities = aggregate_entities(
        text=text,
        offsets=[(0, 6)],
        word_ids=[0],
        label_ids=[3],
        scores=[0.9],
        id2label=ID2LABEL,
    )
    assert len(entities) == 1
    assert (entities[0].start, entities[0].end) == (1, 6)
    assert entities[0].text == "Paris"


def test_o_and_unknown_labels_are_dropped() -> None:
    entities = aggregate_entities(
        text=TEXT,
        offsets=[(0, 7), (8, 13)],
        word_ids=[0, 1],
        label_ids=[0, 9],
        scores=[0.99, 0.99],
        id2label=ID2LABEL,
    )
    assert entities == []


def test_offsets_invariant_holds_for_every_entity() -> None:
    offsets, word_ids, label_ids, scores = _tokens()
    entities = aggregate_entities(
        text=TEXT,
        offsets=offsets,
        word_ids=word_ids,
        label_ids=label_ids,
        scores=scores,
        id2label=ID2LABEL,
    )
    for entity in entities:
        assert TEXT[entity.start : entity.end] == entity.text


def test_special_tokens_with_no_word_id_are_ignored() -> None:
    text = "Paris"
    entities = aggregate_entities(
        text=text,
        offsets=[(0, 0), (0, 5), (0, 0)],
        word_ids=[None, 0, None],
        label_ids=[3, 3, 3],
        scores=[0.5, 0.9, 0.5],
        id2label=ID2LABEL,
    )
    assert [(e.text, e.start, e.end) for e in entities] == [("Paris", 0, 5)]


def _entity(text: str, word: str, label: str = "MISC") -> Entity:
    start = text.index(word)
    return Entity(text=word, label=label, start=start, end=start + len(word), score=0.9)


def test_trailing_punctuation_is_stripped() -> None:
    text = "Le Grand Rex."
    entity = _strip_punctuation(text, _entity(text, "Grand Rex.", "LOC"))
    assert entity is not None
    assert (entity.start, entity.end, entity.text) == (3, 12, "Grand Rex")


def test_leading_and_trailing_quotes_are_stripped() -> None:
    text = "Il a dit « Paris » hier."
    entity = _strip_punctuation(text, _entity(text, "« Paris »", "LOC"))
    assert entity is not None
    assert entity.text == "Paris"


def test_clean_span_is_returned_unchanged() -> None:
    text = "Paris"
    original = _entity(text, "Paris", "LOC")
    assert _strip_punctuation(text, original) is original


def test_punctuation_only_span_disappears() -> None:
    text = "Fin."
    assert _strip_punctuation(text, _entity(text, "Fin.")) is not None
    assert _strip_punctuation(text, _entity(text, ".")) is None


def test_stripped_entity_still_satisfies_the_offset_invariant() -> None:
    text = "Le Grand Rex. Puis Paris."
    for word, label in [("Grand Rex.", "LOC"), ("Paris.", "LOC")]:
        entity = _strip_punctuation(text, _entity(text, word, label))
        assert entity is not None
        assert text[entity.start : entity.end] == entity.text

def test_markdown_emphasis_is_stripped() -> None:
    text = "phrase adressée **à Ebenezer** (concerne"
    start = text.index("Ebenezer")
    entity = _strip_punctuation(
        text, Entity(text="Ebenezer**", label="PER", start=start, end=start + 10, score=0.7)
    )
    assert entity is not None
    assert entity.text == "Ebenezer"


def test_trailing_emphasis_then_parenthesis_is_fully_stripped() -> None:
    text = "la Salerie)** et"
    entity = _strip_punctuation(
        text, Entity(text="la Salerie)**", label="LOC", start=0, end=13, score=0.8)
    )
    assert entity is not None
    assert entity.text == "la Salerie"


def test_leading_emphasis_is_stripped() -> None:
    text = "**Paris**"
    entity = _strip_punctuation(
        text, Entity(text="**Paris", label="LOC", start=0, end=7, score=0.9)
    )
    assert entity is not None
    assert entity.text == "Paris"
    assert text[entity.start : entity.end] == entity.text


def test_markdown_table_pipes_are_stripped() -> None:
    text = "| Paris | France |"
    entity = _strip_punctuation(
        text, Entity(text="| Paris |", label="LOC", start=0, end=9, score=0.9)
    )
    assert entity is not None
    assert entity.text == "Paris"


def test_span_made_only_of_decoration_disappears() -> None:
    text = "**"
    entity = _strip_punctuation(text, Entity(text="**", label="MISC", start=0, end=2, score=0.6))
    assert entity is None
