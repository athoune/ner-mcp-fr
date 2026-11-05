"""French NER engine: ONNX graph, our aggregation, no torch.

This module knows nothing about MCP or argparse: it takes text in, gives schemas out. Both the
MCP server and the CLI call it, so what you verify on the command line is exactly what an agent
receives.

Design notes:

* Token aggregation is re-implemented instead of using
  ``pipeline(aggregation_strategy="simple")`` because we must guarantee two things the pipeline
  does not: character offsets that validate against the original text, and windowing for texts
  longer than the model context.
* Inference runs on onnxruntime. `torch` is only needed to export the graph, never to use it,
  which keeps the install at ~150 MB instead of ~780 MB.
* Model loading is lazy: ``tools/list`` and ``ner-mcp-fr info`` work without downloading 210 MB.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .artifacts import ArtifactError
from .backend import BackendError, OnnxBackend
from .chunking import Window, split_windows
from .config import ATTRIBUTION, LABELS, PROJECT_URL, Settings
from .schemas import BatchItem, BatchResult, Entity, ExtractionResult, ModelInfo

LOGGER = logging.getLogger("ner_mcp.engine")

# The model absorbs sentence punctuation and Markdown emphasis into entity spans: "Grand Rex.",
# "Ebenezer**", "Saint-Asaph)**". Stripping runs at both edges, after merging (stripping earlier
# would break the merge of adjacent words across a glued punctuation mark), and the raw spans stay
# available through trim_punctuation=false.

# SentencePiece marks word starts with this glyph. Fast tokenizers normally exclude it from
# offsets, but trimming it defensively costs nothing and keeps the offset invariant honest.
SPIECE_PREFIX = "\u2581"


class EngineError(RuntimeError):
    """Raised for runtime failures the caller should report verbatim to a human."""


@dataclass
class Outcome:
    """Per-text outcome of a batch: either a result or a human-readable error."""

    result: ExtractionResult | None = None
    error: str | None = None


def aggregate_entities(
    *,
    text: str,
    offsets: Sequence[tuple[int, int]],
    word_ids: Sequence[int | None],
    label_ids: Sequence[int],
    scores: Sequence[float],
    id2label: Mapping[int, str],
    merge_adjacent: bool = True,
) -> list[Entity]:
    """Group sub-token predictions into word-level, then entity-level, spans.

    Pure function: no torch, no tokenizer. That is what makes the aggregation contract testable
    without downloading a model.

    `label_ids` and `scores` are the argmax and its probability for each sub-token; `offsets`
    are character spans in `text`.
    """
    words = _group_words(offsets, word_ids, label_ids, scores, id2label)
    entities: list[Entity] = []
    for label, start, end, score in words:
        start, end = _trim(text, start, end)
        if start >= end or label not in LABELS:
            continue
        if (
            merge_adjacent
            and entities
            and entities[-1].label == label
            and entities[-1].end <= start
            and not text[entities[-1].end : start].strip()
        ):
            # Merged entity keeps the best word score: averaging unweighted word scores would
            # understate a confident short word glued to a weaker long one.
            previous = entities.pop()
            start, end, score = previous.start, end, max(previous.score, score)
        entities.append(
            Entity(text=text[start:end], label=label, start=start, end=end, score=round(score, 6))
        )
    return entities


def _group_words(
    offsets: Sequence[tuple[int, int]],
    word_ids: Sequence[int | None],
    label_ids: Sequence[int],
    scores: Sequence[float],
    id2label: Mapping[int, str],
) -> list[tuple[str, int, int, float]]:
    """Collapse sub-tokens into words: first sub-token decides the label, scores are averaged."""
    words: list[tuple[str, int, int, float]] = []
    current_id: int | None = None
    current_label = "O"
    current_score = 0.0
    current_count = 0
    current_start = 0
    current_end = 0
    for (start, end), word_id, label_id, score in zip(
        offsets, word_ids, label_ids, scores, strict=True
    ):
        if word_id is None:  # special token or padding
            continue
        if word_id != current_id:
            if current_id is not None:
                words.append(
                    (
                        current_label,
                        current_start,
                        current_end,
                        current_score / max(current_count, 1),
                    )
                )
            current_id = word_id
            current_label = id2label.get(label_id, "O")
            current_score = 0.0
            current_count = 0
            current_start, current_end = start, end
        current_score += score
        current_count += 1
        current_start = min(current_start, start)
        current_end = max(current_end, end)
    if current_id is not None:
        words.append(
            (current_label, current_start, current_end, current_score / max(current_count, 1))
        )
    return words


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    """Drop leading/trailing whitespace and any SentencePiece marker, moving offsets, not text."""
    length = len(text)
    start = max(0, min(start, length))
    end = max(start, min(end, length))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    while start < end and text.startswith(SPIECE_PREFIX, start):
        start += len(SPIECE_PREFIX)
    return start, end


class NerEngine:
    """Lazy-loading NER engine shared by the CLI and the MCP server."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self._lock = threading.Lock()
        self._backend = OnnxBackend(self.settings)
        if self.settings.preload:
            self.load()

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._backend.loaded

    @property
    def device(self) -> str:
        return self._backend.device if self.loaded else self.settings.device

    @property
    def precision(self) -> str:
        return self._backend.precision

    def load(self) -> None:
        """Download and open the graph. Idempotent and thread-safe."""
        with self._lock:
            if self._backend.loaded:
                return
            started = time.perf_counter()
            LOGGER.info(
                "loading %s (%s), first run downloads the weights",
                self.settings.artifact_file,
                self.settings.artifact_repo,
            )
            try:
                self._backend.load()
            except (BackendError, ArtifactError) as exc:
                raise EngineError(str(exc)) from exc

            unknown = set(self._backend.id2label.values()) - {"O", *LABELS}
            if unknown:
                LOGGER.warning(
                    "%s has unexpected labels %s; only %s are reported",
                    self.settings.model_id,
                    sorted(unknown),
                    ", ".join(LABELS),
                )
            LOGGER.info(
                "model ready in %.1fs (%d-token context, %s, %s)",
                time.perf_counter() - started,
                self._backend.max_tokens,
                self._backend.precision,
                self.device,
            )

    # -- public API --------------------------------------------------------------------

    def extract(
        self,
        text: str,
        *,
        labels: Sequence[str] | None = None,
        min_score: float | None = None,
        merge_adjacent: bool = True,
        trim_punctuation: bool = True,
    ) -> ExtractionResult:
        """Run NER on one text and return the wire schema."""
        outcome = self._extract_all(
            [text],
            labels=labels,
            min_score=min_score,
            merge_adjacent=merge_adjacent,
            trim_punctuation=trim_punctuation,
        )[0]
        if outcome.error is not None:
            raise EngineError(outcome.error)
        assert outcome.result is not None
        return outcome.result

    def extract_many(
        self,
        texts: Sequence[str],
        *,
        labels: Sequence[str] | None = None,
        min_score: float | None = None,
        merge_adjacent: bool = True,
        trim_punctuation: bool = True,
    ) -> list[Outcome]:
        """Run NER on several texts in one batched pass; a bad text does not fail the batch."""
        return self._extract_all(
            texts,
            labels=labels,
            min_score=min_score,
            merge_adjacent=merge_adjacent,
            trim_punctuation=trim_punctuation,
        )

    def extract_batch(
        self,
        texts: Sequence[str],
        *,
        labels: Sequence[str] | None = None,
        min_score: float | None = None,
        merge_adjacent: bool = True,
        trim_punctuation: bool = True,
    ) -> BatchResult:
        """Batched extraction shaped as the `ner_batch` tool output."""
        started = time.perf_counter()
        outcomes = self.extract_many(
            texts,
            labels=labels,
            min_score=min_score,
            merge_adjacent=merge_adjacent,
            trim_punctuation=trim_punctuation,
        )
        items = [
            BatchItem(
                index=index,
                text_length=len(texts[index]),
                entities=outcome.result.entities if outcome.result else [],
                chunks=outcome.result.chunks if outcome.result else 0,
                error=outcome.error,
            )
            for index, outcome in enumerate(outcomes)
        ]
        return BatchResult(
            model=self.settings.model_id,
            precision=self.precision,
            device=self.device,
            results=items,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def info(self) -> ModelInfo:
        """Describe the engine without loading the model."""
        return ModelInfo(
            model=self.settings.model_id,
            precision=self.precision,
            device=self.device,
            device_requested=self.settings.device,
            labels=list(LABELS),
            max_tokens=self._backend.max_tokens if self.loaded else None,
            max_input_chars=self.settings.max_input_chars,
            loaded=self.loaded,
            package_version=_package_version(),
            attribution=ATTRIBUTION,
            cache_dir=self.settings.cache_dir,
            project_url=PROJECT_URL,
        )

    # -- internals ---------------------------------------------------------------------

    def _extract_all(
        self,
        texts: Sequence[str],
        *,
        labels: Sequence[str] | None,
        min_score: float | None,
        merge_adjacent: bool,
        trim_punctuation: bool = True,
    ) -> list[Outcome]:
        started = time.perf_counter()
        wanted = _normalize_labels(labels)
        threshold = self.settings.min_score if min_score is None else min_score
        if not 0.0 <= threshold <= 1.0:
            raise EngineError(f"min_score must be between 0 and 1, got {threshold}")

        outcomes = [Outcome() for _ in texts]
        plans: list[list[Window]] = []
        planned_indices: list[int] = []
        backend: OnnxBackend | None = None

        for index, text in enumerate(texts):
            problem = self._validate_text(text)
            if problem is not None:
                outcomes[index] = Outcome(error=problem)
                plans.append([])
                continue
            backend = backend or self._ensure_loaded()
            plans.append(split_windows(text, backend.count_tokens, backend.max_tokens))
            planned_indices.append(index)

        if backend is None or not planned_indices:
            return outcomes

        # One batched pass over every window of every text.
        flat = [window.text for plan in plans for window in plan]
        predictions = self._infer(flat, backend, merge_adjacent=merge_adjacent)

        cursor = 0
        for index in planned_indices:
            text = texts[index]
            plan = plans[index]
            per_window = predictions[cursor : cursor + len(plan)]
            cursor += len(plan)
            outcomes[index] = Outcome(
                result=self._assemble(
                    text=text,
                    plan=plan,
                    per_window=per_window,
                    wanted=wanted,
                    threshold=threshold,
                    trim_punctuation=trim_punctuation,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            )
        return outcomes

    def _ensure_loaded(self) -> OnnxBackend:
        if not self._backend.loaded:
            self.load()
        return self._backend

    def _validate_text(self, text: str) -> str | None:
        if not isinstance(text, str):
            return f"text must be a string, got {type(text).__name__}"
        if not text.strip():
            return "text is empty"
        if len(text) > self.settings.max_input_chars:
            return (
                f"text is too long: {len(text)} code points, limit is "
                f"{self.settings.max_input_chars} (NER_MAX_INPUT_CHARS)"
            )
        return None

    def _infer(
        self, texts: Sequence[str], backend: OnnxBackend, *, merge_adjacent: bool = True
    ) -> list[list[Entity]]:
        """Run the graph and aggregate, per window, with window-local offsets."""
        try:
            predictions = backend.predict(list(texts))
        except BackendError as exc:
            raise EngineError(str(exc)) from exc

        entities: list[list[Entity]] = []
        for text, prediction in zip(texts, predictions, strict=True):
            entities.append(
                aggregate_entities(
                    text=text,
                    offsets=prediction.offsets,
                    word_ids=prediction.word_ids,
                    label_ids=prediction.label_ids,
                    scores=prediction.scores,
                    id2label=backend.id2label,
                    merge_adjacent=merge_adjacent,
                )
            )
        return entities

    def _assemble(
        self,
        *,
        text: str,
        plan: Sequence[Window],
        per_window: Sequence[Sequence[tuple[str, int, int, float]]],
        wanted: set[str],
        threshold: float,
        trim_punctuation: bool,
        elapsed_ms: int,
    ) -> ExtractionResult:
        """Rebase window-local spans onto the original text, deduplicate, filter."""
        merged: dict[tuple[str, int, int], Entity] = {}
        for window, entities in zip(plan, per_window, strict=True):
            for entity in entities:
                # Windows are exact substrings, so rebasing is a plain addition.
                start, end = window.start + entity.start, window.start + entity.end
                if start >= end:
                    continue
                candidate = Entity(
                    text=text[start:end],
                    label=entity.label,
                    start=start,
                    end=end,
                    score=entity.score,
                )
                key = (entity.label, start, end)
                previous = merged.get(key)
                if previous is None or candidate.score > previous.score:
                    merged[key] = candidate

        kept: list[Entity] = []
        dropped = 0
        candidates = sorted(merged.values(), key=lambda e: (e.start, e.end, e.label))
        if trim_punctuation:
            candidates = [
                trimmed
                for entity in candidates
                if (trimmed := _strip_punctuation(text, entity)) is not None
            ]
        for entity in candidates:
            if entity.label not in wanted:
                continue
            if entity.score < threshold:
                dropped += 1
                continue
            kept.append(entity)

        return ExtractionResult(
            model=self.settings.model_id,
            precision=self.precision,
            device=self.device,
            entities=kept,
            text_length=len(text),
            chunks=max(len(plan), 1),
            # Oversized inputs are rejected in _validate_text, so nothing is ever truncated.
            truncated=False,
            dropped_by_min_score=dropped,
            elapsed_ms=elapsed_ms,
        )


# `|` and `~` are symbols, not punctuation, but they are Markdown table pipes and subscripts.
SYMBOL_NOISE = "|~"


def _is_edge_noise(character: str) -> bool:
    """Whitespace, Unicode punctuation and Markdown/Symbol decoration.

    Category-based rather than a hardcoded list: it covers `.` `,` `**` `#` `_` `«»` `()` without
    a new bug every time a document uses another mark.
    """
    return (
        character.isspace()
        or character in SYMBOL_NOISE
        or unicodedata.category(character).startswith("P")
    )


def _strip_punctuation(text: str, entity: Entity) -> Entity | None:
    """Drop punctuation and Markdown marks the model glued onto an entity span."""
    start, end = entity.start, entity.end
    while start < end and _is_edge_noise(text[start]):
        start += 1
    while end > start and _is_edge_noise(text[end - 1]):
        end -= 1
    if (start, end) == (entity.start, entity.end):
        return entity
    if start >= end:
        return None
    return entity.model_copy(update={"start": start, "end": end, "text": text[start:end]})


def _normalize_labels(labels: Sequence[str] | None) -> set[str]:
    if labels is None:
        return set(LABELS)
    normalized = {str(label).strip().upper() for label in labels if str(label).strip()}
    if not normalized:
        return set(LABELS)
    unknown = normalized - set(LABELS)
    if unknown:
        raise EngineError(
            f"unknown labels {sorted(unknown)}; expected a subset of {', '.join(LABELS)}"
        )
    return normalized


def _package_version() -> str:
    from . import __version__

    return __version__
