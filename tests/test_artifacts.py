"""Artifact resolution: the checksum is the only thing standing between a good and a bad model."""

from __future__ import annotations

import pytest

from ner_mcp import artifacts
from ner_mcp.config import DEFAULT_ARTIFACT_SHA256, Settings


def test_pinned_checksum_looks_like_a_sha256() -> None:
    # A truncated or mistyped pin would silently disable verification for everybody.
    assert len(DEFAULT_ARTIFACT_SHA256) == 64
    assert all(character in "0123456789abcdef" for character in DEFAULT_ARTIFACT_SHA256)


def test_verify_accepts_the_right_digest(tmp_path) -> None:
    payload = tmp_path / "model.onnx"
    payload.write_bytes(b"weights")
    artifacts.verify(payload, artifacts.sha256(payload))


def test_verify_rejects_a_swapped_artifact(tmp_path) -> None:
    payload = tmp_path / "model.onnx"
    payload.write_bytes(b"other weights")
    with pytest.raises(artifacts.ArtifactError, match="checksum mismatch"):
        artifacts.verify(payload, "0" * 64)


def test_precision_is_read_from_the_filename() -> None:
    assert artifacts.Artifacts(
        model=__import__("pathlib").Path("model-fp16.onnx"),
        tokenizer=__import__("pathlib").Path("tokenizer.json"),
        config=__import__("pathlib").Path("config.json"),
    ).precision == "fp16"


def test_missing_local_artifact_is_reported_before_anything_else(tmp_path) -> None:
    settings = Settings.from_env({"NER_ONNX_PATH": str(tmp_path / "nope.onnx")})
    with pytest.raises(artifacts.ArtifactError, match="does not exist"):
        artifacts.resolve(settings)
