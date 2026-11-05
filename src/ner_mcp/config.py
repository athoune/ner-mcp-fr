"""Runtime configuration, read once from environment variables.

Invalid configuration fails fast with an actionable :class:`ConfigError`: silently falling back
to a default would hide a typo for the whole lifetime of a long-running server.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

# Entity labels produced by the frenchNER_4entities models.
LABELS: tuple[str, ...] = ("PER", "LOC", "ORG", "MISC")

# Tokenizer and label map come from the upstream, MIT licensed model. The weights are an ONNX
# file of ours, because onnxruntime cannot read PyTorch checkpoints.
SOURCE_MODEL = "CATIE-AQ/NERmembert-base-4entities"
DEFAULT_ARTIFACT_REPO = "athoune/ner-mcp-fr-onnx"
DEFAULT_ARTIFACT_FILE = "model-fp16.onnx"
# Printed by `scripts/export_onnx.py`. Pinning it means a swapped or corrupted artifact fails
# loudly instead of silently changing predictions.
DEFAULT_ARTIFACT_SHA256 = "afd6f6dad996fc2585ad33615e8434ba2734a8270e9e7ebc0deb217e062cd2b5"

VALID_DEVICES = ("auto", "cpu", "cuda")
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

PROJECT_URL = "https://github.com/athoune/ner-mcp-fr"

UPSTREAM_LICENSE = "MIT"
ATTRIBUTION = (
    "CATIE-AQ/NERmembert, fine-tuned on frenchNER_4entities (MIT); "
    "exported to ONNX by this project"
)


class ConfigError(ValueError):
    """Raised when the environment provides an unusable configuration."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings."""

    model_id: str = SOURCE_MODEL
    device: str = "auto"
    max_input_chars: int = 200_000
    batch_size: int = 8
    min_score: float = 0.0
    cache_dir: str | None = None
    preload: bool = False
    offline: bool = False
    threads: int | None = None
    log_level: str = "INFO"
    # Empty means "no restriction": the server runs with the user's own permissions.
    allowed_dirs: tuple[str, ...] = ()
    artifact_repo: str = DEFAULT_ARTIFACT_REPO
    artifact_file: str = DEFAULT_ARTIFACT_FILE
    artifact_sha256: str | None = None
    # Local ONNX file, bypassing the download entirely (development, or a private export).
    onnx_path: str | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the process environment (or an explicit mapping)."""
        env = os.environ if env is None else env

        device = (env.get("NER_DEVICE") or "auto").strip().lower()
        if device == "mps":
            raise ConfigError(
                "NER_DEVICE=mps no longer applies: the ONNX backend has no Metal provider. "
                "Use NER_DEVICE=auto or cpu (CoreML was measured slower than CPU)."
            )
        if device not in VALID_DEVICES:
            raise ConfigError(
                f"NER_DEVICE={device!r} is invalid; expected one of {', '.join(VALID_DEVICES)}"
            )

        log_level = (env.get("NER_LOG_LEVEL") or "INFO").strip().upper()
        if log_level not in VALID_LOG_LEVELS:
            raise ConfigError(
                f"NER_LOG_LEVEL={log_level!r} is invalid; expected one of "
                f"{', '.join(VALID_LOG_LEVELS)}"
            )

        return cls(
            model_id=(env.get("NER_MODEL_ID") or "").strip() or SOURCE_MODEL,
            device=device,
            max_input_chars=_int(env, "NER_MAX_INPUT_CHARS", 200_000, minimum=1),
            batch_size=_int(env, "NER_BATCH_SIZE", 8, minimum=1),
            min_score=_float(env, "NER_MIN_SCORE", 0.0, low=0.0, high=1.0),
            cache_dir=(env.get("NER_CACHE_DIR") or "").strip() or None,
            preload=_bool(env, "NER_PRELOAD", False),
            offline=_bool(env, "NER_OFFLINE", False),
            threads=_optional_int(env, "NER_THREADS", minimum=1),
            log_level=log_level,
            allowed_dirs=_paths(env, "NER_ALLOWED_DIRS"),
            artifact_repo=(env.get("NER_ARTIFACT_REPO") or "").strip() or DEFAULT_ARTIFACT_REPO,
            artifact_file=(env.get("NER_ARTIFACT_FILE") or "").strip() or DEFAULT_ARTIFACT_FILE,
            artifact_sha256=(env.get("NER_ARTIFACT_SHA256") or "").strip()
            or (DEFAULT_ARTIFACT_SHA256 or None),
            onnx_path=(env.get("NER_ONNX_PATH") or "").strip() or None,
        )


def _int(env: Mapping[str, str], name: str, default: int, *, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name}={value} must be >= {minimum}")
    return value


def _optional_int(env: Mapping[str, str], name: str, *, minimum: int) -> int | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return _int(env, name, minimum, minimum=minimum)


def _float(
    env: Mapping[str, str], name: str, default: float, *, low: float, high: float
) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} must be a number") from exc
    if not low <= value <= high:
        raise ConfigError(f"{name}={value} must be between {low} and {high}")
    return value


def _paths(env: Mapping[str, str], name: str) -> tuple[str, ...]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return ()
    return tuple(part for part in (chunk.strip() for chunk in raw.split(os.pathsep)) if part)


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name}={raw!r} must be a boolean (1/0, true/false)")
