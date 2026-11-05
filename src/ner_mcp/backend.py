"""Inference backend: an ONNX graph run by onnxruntime, plus its tokenizer.

No torch, no transformers. The tokenizer comes from the `tokenizers` library alone, which was
verified to produce byte-identical ids, offsets and word ids to `AutoTokenizer`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from . import artifacts
from .config import Settings


class BackendError(RuntimeError):
    """Raised when the graph or the tokenizer cannot be used."""


@dataclass
class TokenPrediction:
    """Per-sub-token predictions for one text, ready for aggregation."""

    offsets: list[tuple[int, int]]
    word_ids: list[int | None]
    label_ids: list[int]
    scores: list[float]


class OnnxBackend:
    """Lazy ONNX session with its tokenizer and label map."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._session = None
        self._tokenizer: Tokenizer | None = None
        self._id2label: dict[int, str] = {}
        self._pad_id = 1
        self._max_tokens = 512
        self._device = "cpu"
        self._artifacts: artifacts.Artifacts | None = None

    # -- lifecycle ---------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def id2label(self) -> dict[int, str]:
        return self._id2label

    @property
    def precision(self) -> str:
        return self._artifacts.precision if self._artifacts else "unknown"

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - dependency error
            raise BackendError(
                "onnxruntime is required; install the project with `uv sync`"
            ) from exc

        resolved = artifacts.resolve(self.settings)
        self._artifacts = resolved

        config = json.loads(resolved.config.read_text(encoding="utf-8"))
        self._id2label = {int(k): str(v) for k, v in config["id2label"].items()}
        self._pad_id = int(config.get("pad_token_id", 1))
        limit = int(config.get("max_position_embeddings", 514))
        # One seat for [CLS], one for [SEP].
        self._max_tokens = max(limit - 2, 8)

        try:
            self._tokenizer = Tokenizer.from_file(str(resolved.tokenizer))
        except Exception as exc:
            raise BackendError(f"cannot load tokenizer: {exc}") from exc

        providers = self._providers(ort)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.settings.threads:
            options.intra_op_num_threads = self.settings.threads
        try:
            self._session = ort.InferenceSession(
                str(resolved.model), options, providers=providers
            )
        except Exception as exc:
            raise BackendError(f"cannot load {resolved.model.name}: {exc}") from exc
        self._device = self._session.get_providers()[0]

    def _providers(self, ort) -> list[str]:
        available = ort.get_available_providers()
        requested = self.settings.device
        if requested == "auto":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"] if (
                "CUDAExecutionProvider" in available
            ) else ["CPUExecutionProvider"]
        wanted = "CUDAExecutionProvider" if requested == "cuda" else "CPUExecutionProvider"
        if wanted not in available:
            raise BackendError(
                f"NER_DEVICE={requested} but {wanted} is not available "
                f"(onnxruntime offers {', '.join(available)}). "
                "Install onnxruntime-gpu for CUDA, or use NER_DEVICE=cpu."
            )
        return [wanted, "CPUExecutionProvider"]

    # -- inference ---------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self._require_tokenizer().encode(text, add_special_tokens=False).ids)

    def predict(self, texts: list[str]) -> list[TokenPrediction]:
        """Run the graph on a batch of texts, padding to the longest one."""
        session = self._require_session()
        tokenizer = self._require_tokenizer()
        batch_size = max(self.settings.batch_size, 1)
        results: list[TokenPrediction] = []

        for start in range(0, len(texts), batch_size):
            encodings = [tokenizer.encode(text) for text in texts[start : start + batch_size]]
            length = max(len(encoding.ids) for encoding in encodings)

            input_ids = np.full((len(encodings), length), self._pad_id, dtype=np.int64)
            attention = np.zeros((len(encodings), length), dtype=np.int64)
            for row, encoding in enumerate(encodings):
                size = len(encoding.ids)
                input_ids[row, :size] = encoding.ids
                attention[row, :size] = 1

            try:
                logits = session.run(
                    None, {"input_ids": input_ids, "attention_mask": attention}
                )[0]
            except Exception as exc:
                raise BackendError(f"inference failed: {exc}") from exc

            for row, encoding in enumerate(encodings):
                size = len(encoding.ids)
                token_logits = logits[row, :size].astype(np.float64)
                probabilities = _softmax(token_logits)
                best = probabilities.argmax(axis=-1)
                results.append(
                    TokenPrediction(
                        offsets=[tuple(pair) for pair in encoding.offsets],
                        word_ids=list(encoding.word_ids),
                        label_ids=best.tolist(),
                        scores=probabilities[np.arange(size), best].tolist(),
                    )
                )
        return results

    def _require_session(self):
        if self._session is None:
            raise BackendError("backend not loaded")
        return self._session

    def _require_tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise BackendError("backend not loaded")
        return self._tokenizer


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


def artifact_filename(path: Path | None = None) -> str:
    return path.name if path else artifacts.DEFAULT_ARTIFACT_FILE
