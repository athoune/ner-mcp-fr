"""Transformer-backed French NER engine.

This module knows nothing about MCP or argparse: it takes text in, gives schemas out. Both the
MCP server and the CLI call it, so what you verify on the command line is exactly what an agent
receives.

Design notes:

* Token aggregation is re-implemented instead of using
  ``pipeline(aggregation_strategy="simple")`` because we must guarantee two things the pipeline
  does not: character offsets that validate against the original text, and windowing for texts
  longer than the model context.
* Model inference is lazily loaded: ``tools/list`` and ``ner-mcp-fr info`` must work without
  downloading 400 MB.
"""

from __future__ import annotations

import logging
import threading
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .chunking import Window, split_windows
from .config import LABELS, PROJECT_URL, Settings
from .schemas import BatchItem, BatchResult, Entity, ExtractionResult, ModelInfo

LOGGER = logging.getLogger("ner_mcp.engine")

# Extra room kept for special tokens ([CLS], [SEP], and padding on the last window).
SPECIAL_TOKENS_RESERVE = 2

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


@dataclass
class _Loaded:
    """Handles and derived limits, only available once the model is in memory."""

    tokenizer: Any
    model: Any
    device: str
    max_tokens: int
    id2label: Mapping[int, str]
    count_tokens: Any = field(repr=False)


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
        self._loaded: _Loaded | None = None
        self._torch: Any = None
        if self.settings.preload:
            self.load()

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    @property
    def device(self) -> str:
        return self._loaded.device if self._loaded else self.settings.device

    def load(self) -> None:
        """Load tokenizer and weights. Idempotent and thread-safe."""
        with self._lock:
            if self._loaded is not None:
                return
            self._loaded = self._load()

    def _load(self) -> _Loaded:
        started = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency error
            raise EngineError(
                "torch and transformers are required; install the project with `uv sync`"
            ) from exc

        self._torch = torch
        if self.settings.threads:
            torch.set_num_threads(self.settings.threads)

        device = _resolve_device(torch, self.settings.device)
        LOGGER.info(
            "loading %s on %s, first run downloads the weights", self.settings.model_id, device
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.model_id,
                cache_dir=self.settings.cache_dir,
                local_files_only=self.settings.offline,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.settings.model_id,
                cache_dir=self.settings.cache_dir,
                local_files_only=self.settings.offline,
            )
        except Exception as exc:  # re-raised below with an actionable message
            raise EngineError(
                f"cannot load model {self.settings.model_id!r}: {exc}. "
                "Check the network, or set NER_CACHE_DIR to an existing Hugging Face cache."
            ) from exc

        # Keep the checkpoint dtype: it is a property of the weights, not a user preference.
        # Downcasting here would change predictions without anyone asking for it.
        model.to(device=device)
        model.eval()

        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        unknown = set(id2label.values()) - {"O", *LABELS}
        if unknown:
            LOGGER.warning(
                "%s has unexpected labels %s; only %s are reported",
                self.settings.model_id,
                sorted(unknown),
                ", ".join(LABELS),
            )

        max_tokens = _max_tokens(model, tokenizer)

        def count_tokens(chunk: str) -> int:
            return len(tokenizer(chunk, add_special_tokens=False)["input_ids"])

        LOGGER.info(
            "model ready in %.1fs (%d-token context)",
            time.perf_counter() - started,
            max_tokens,
        )
        return _Loaded(
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_tokens=max_tokens,
            id2label=id2label,
            count_tokens=count_tokens,
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
            profile=self.settings.profile,
            device=self.device,
            results=items,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    def _dtype_label(self) -> str | None:
        model = self._loaded.model if self._loaded else None
        if model is None:
            return None
        return str(model.dtype).removeprefix("torch.")

    def info(self) -> ModelInfo:
        """Describe the engine without loading the model."""
        return ModelInfo(
            model=self.settings.model_id,
            profile=self.settings.profile,
            device=self.device,
            device_requested=self.settings.device,
            dtype=self._dtype_label(),
            labels=list(LABELS),
            max_tokens=self._loaded.max_tokens if self._loaded else None,
            max_input_chars=self.settings.max_input_chars,
            loaded=self.loaded,
            package_version=_package_version(),
            attribution=(
                "CATIE-AQ/NERmembert & NERmemberta, fine-tuned on frenchNER_4entities (MIT)"
            ),
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
        loaded: _Loaded | None = None

        for index, text in enumerate(texts):
            problem = self._validate_text(text)
            if problem is not None:
                outcomes[index] = Outcome(error=problem)
                plans.append([])
                continue
            loaded = loaded or self._ensure_loaded()
            plans.append(split_windows(text, loaded.count_tokens, loaded.max_tokens))
            planned_indices.append(index)

        if loaded is None or not planned_indices:
            return outcomes

        # One batched pass over every window of every text.
        flat = [window.text for plan in plans for window in plan]
        predictions = self._infer(flat, loaded, merge_adjacent=merge_adjacent)

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

    def _ensure_loaded(self) -> _Loaded:
        if self._loaded is None:
            self.load()
        assert self._loaded is not None
        return self._loaded

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
        self, texts: Sequence[str], loaded: _Loaded, *, merge_adjacent: bool = True
    ) -> list[list[Entity]]:
        """Run the model and return per-window entities, with window-local offsets."""
        torch = self._torch
        predictions: list[list[Entity]] = []
        batch_size = max(self.settings.batch_size, 1)
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = loaded.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=loaded.max_tokens,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping").tolist()
            word_ids = [encoded.word_ids(i) for i in range(len(batch))]
            encoded = {key: value.to(loaded.device) for key, value in encoded.items()}

            with torch.inference_mode():
                logits = loaded.model(**encoded).logits
            probabilities = logits.softmax(dim=-1)

            for i, text in enumerate(batch):
                length = int(encoded["attention_mask"][i].sum())
                token_probs = probabilities[i, :length]
                best = token_probs.argmax(dim=-1)
                best_scores = token_probs.gather(1, best.unsqueeze(1)).squeeze(1)
                predictions.append(
                    aggregate_entities(
                        text=text,
                        offsets=offsets[i][:length],
                        word_ids=word_ids[i][:length],
                        label_ids=best.tolist(),
                        scores=best_scores.tolist(),
                        id2label=loaded.id2label,
                        merge_adjacent=merge_adjacent,
                    )
                )
        return predictions

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
            profile=self.settings.profile,
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


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise EngineError("NER_DEVICE=cuda but CUDA is not available")
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise EngineError("NER_DEVICE=mps but MPS is not available on this machine")
    return requested


def _max_tokens(model: Any, tokenizer: Any) -> int:
    """Usable context, minus room for special tokens."""
    limit = getattr(model.config, "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit and tokenizer_limit < 100_000:
        limit = min(limit or tokenizer_limit, tokenizer_limit)
    return max(int(limit or 512) - SPECIAL_TOKENS_RESERVE, 8)


def _package_version() -> str:
    from . import __version__

    return __version__
