"""Configuration is parsed once; a typo must fail loudly, not silently fall back."""

from __future__ import annotations

import os

import pytest

from ner_mcp.config import (
    DEFAULT_ARTIFACT_FILE,
    DEFAULT_ARTIFACT_REPO,
    SOURCE_MODEL,
    ConfigError,
    Settings,
)


def test_defaults_point_at_the_published_artifact() -> None:
    settings = Settings.from_env({})
    assert settings.model_id == SOURCE_MODEL
    assert settings.artifact_repo == DEFAULT_ARTIFACT_REPO
    assert settings.artifact_file == DEFAULT_ARTIFACT_FILE
    assert settings.device == "auto"
    assert settings.batch_size == 8
    assert settings.offline is False
    assert settings.onnx_path is None


def test_artifact_can_be_overridden() -> None:
    settings = Settings.from_env(
        {"NER_ARTIFACT_REPO": "me/my-model", "NER_ARTIFACT_FILE": "model.onnx"}
    )
    assert settings.artifact_repo == "me/my-model"
    assert settings.artifact_file == "model.onnx"


def test_local_onnx_path_bypasses_the_download(tmp_path) -> None:
    settings = Settings.from_env({"NER_ONNX_PATH": str(tmp_path / "model.onnx")})
    assert settings.onnx_path == str(tmp_path / "model.onnx")


def test_mps_is_rejected_with_an_explanation() -> None:
    with pytest.raises(ConfigError, match="no Metal provider"):
        Settings.from_env({"NER_DEVICE": "mps"})


def test_source_model_is_not_a_broken_camemberta_tokenizer() -> None:
    # The Camembertva models ship a byte-level tokenizer without merge rules; see SPECS.md.
    assert "camemberta" not in SOURCE_MODEL.lower()


def test_allowed_dirs_are_parsed_from_os_pathsep() -> None:
    settings = Settings.from_env({"NER_ALLOWED_DIRS": f"/tmp{os.pathsep}/srv/docs"})
    assert settings.allowed_dirs == ("/tmp", "/srv/docs")


def test_allowed_dirs_default_to_no_restriction() -> None:
    assert Settings.from_env({}).allowed_dirs == ()


@pytest.mark.parametrize(
    "env",
    [
        {"NER_DEVICE": "tpu"},
        {"NER_LOG_LEVEL": "LOUD"},
        {"NER_MAX_INPUT_CHARS": "0"},
        {"NER_MAX_INPUT_CHARS": "many"},
        {"NER_BATCH_SIZE": "0"},
        {"NER_MIN_SCORE": "1.5"},
        {"NER_PRELOAD": "maybe"},
        {"NER_THREADS": "0"},
    ],
)
def test_invalid_values_are_rejected(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_booleans_accept_common_spellings() -> None:
    assert Settings.from_env({"NER_OFFLINE": "yes"}).offline is True
    assert Settings.from_env({"NER_OFFLINE": "0"}).offline is False
    assert Settings.from_env({"NER_OFFLINE": "TRUE"}).offline is True
