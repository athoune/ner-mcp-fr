"""Locate the ONNX weights and the tokenizer, download them once, verify what was received.

`onnxruntime` cannot read PyTorch checkpoints, so the weights are re-published as a single
`.onnx` file in a small Hugging Face repository of our own (see docs/PUBLISHING.md). Everything
else (`tokenizer.json`, `config.json`) still comes from the upstream CATIE-AQ repository: no
reason to mirror files we already trust.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from .config import Settings

CHUNK = 1 << 20


class ArtifactError(RuntimeError):
    """Raised when the weights are missing, unreadable or do not match their checksum."""


@dataclass(frozen=True)
class Artifacts:
    """Resolved local paths."""

    model: Path
    tokenizer: Path
    config: Path

    @property
    def precision(self) -> str:
        name = self.model.name.lower()
        for tag in ("fp16", "int8", "fp32"):
            if tag in name:
                return tag
        return "unknown"


def resolve(settings: Settings) -> Artifacts:
    """Download (once) and verify every file the engine needs."""
    if settings.onnx_path:
        model = Path(settings.onnx_path).expanduser()
        if not model.is_file():
            raise ArtifactError(
                f"NER_ONNX_PATH points to {model}, which does not exist. "
                "Unset it to download the published artifact instead."
            )
    else:
        model = _download(
            settings.artifact_repo,
            settings.artifact_file,
            cache_dir=settings.cache_dir,
            offline=settings.offline,
        )

    if settings.artifact_sha256:
        verify(model, settings.artifact_sha256)

    tokenizer = _download(
        settings.model_id, "tokenizer.json", cache_dir=settings.cache_dir, offline=settings.offline
    )
    config = _download(
        settings.model_id, "config.json", cache_dir=settings.cache_dir, offline=settings.offline
    )
    return Artifacts(model=model, tokenizer=tokenizer, config=config)


def _download(repo_id: str, filename: str, *, cache_dir: str | None, offline: bool) -> Path:
    try:
        return Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_files_only=offline,
            )
        )
    except Exception as exc:  # re-raised with the remediation the user needs
        raise ArtifactError(
            f"cannot fetch {filename} from {repo_id}: {exc}. "
            "Check the network, or set NER_CACHE_DIR to an existing cache, "
            "or point NER_ONNX_PATH at a local model file."
        ) from exc


def verify(path: Path, expected: str) -> None:
    """Fail loudly when the published weights are not the ones this code was tested against."""
    digest = sha256(path)
    if digest != expected:
        raise ArtifactError(
            f"{path.name} checksum mismatch: got {digest}, expected {expected}. "
            "The artifact was replaced, or the download is corrupt; remove the cached file."
        )


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()
