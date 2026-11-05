"""Configuration is parsed once; a typo must fail loudly, not silently fall back."""

from __future__ import annotations

import os

import pytest

from ner_mcp.config import PROFILES, ConfigError, Settings


def test_defaults_use_the_fast_profile() -> None:
    settings = Settings.from_env({})
    assert settings.profile == "fast"
    assert settings.model_id == PROFILES["fast"]
    assert settings.device == "auto"
    assert settings.batch_size == 8
    assert settings.offline is False


def test_explicit_model_id_wins_over_profile() -> None:
    settings = Settings.from_env({"NER_PROFILE": "large", "NER_MODEL_ID": "someone/else"})
    assert settings.profile == "large"
    assert settings.model_id == "someone/else"


def test_profile_selects_the_large_model() -> None:
    settings = Settings.from_env({"NER_PROFILE": "large"})
    assert settings.model_id == PROFILES["large"]


def test_fast_profile_avoids_the_broken_camemberta_tokenizer() -> None:
    # The Camembertva models ship a byte-level tokenizer without merge rules; see SPECS.md.
    assert "camemberta" not in PROFILES["fast"].lower()


def test_allowed_dirs_are_parsed_from_os_pathsep() -> None:
    settings = Settings.from_env({"NER_ALLOWED_DIRS": f"/tmp{os.pathsep}/srv/docs"})
    assert settings.allowed_dirs == ("/tmp", "/srv/docs")


def test_allowed_dirs_default_to_no_restriction() -> None:
    assert Settings.from_env({}).allowed_dirs == ()


@pytest.mark.parametrize(
    "env",
    [
        {"NER_PROFILE": "xl"},
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
