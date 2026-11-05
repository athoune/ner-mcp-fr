"""CLI behaviour: input handling, formats, exit codes. The engine is faked, no model loads."""

from __future__ import annotations

import io
import json
import sys
import types
from typing import ClassVar

import pytest

from ner_mcp import cli
from ner_mcp.config import Settings
from ner_mcp.schemas import Entity, ExtractionResult, ModelInfo

SENTENCE = "Sylvain Boyer est à Paris."


def _result(entities: list[Entity] | None = None) -> ExtractionResult:
    if entities is None:
        entities = [
            Entity(text="Sylvain Boyer", label="PER", start=0, end=13, score=0.99),
            Entity(text="Paris", label="LOC", start=20, end=25, score=0.97),
        ]
    return ExtractionResult(
        model="CATIE-AQ/NERmemberta-4entities",
        precision="fp16",
        device="cpu",
        entities=entities,
        text_length=len(SENTENCE),
        chunks=1,
        truncated=False,
        dropped_by_min_score=0,
        elapsed_ms=11,
    )


class FakeEngine:
    """Captures the arguments the CLI forwards to the real engine."""

    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.empty = False

    def extract(
        self,
        text: str,
        *,
        labels=None,
        min_score=None,
        merge_adjacent=True,
        trim_punctuation=True,
    ):
        FakeEngine.calls.append(
            {
                "text": text,
                "labels": labels,
                "merge_adjacent": merge_adjacent,
                "trim_punctuation": trim_punctuation,
            }
        )
        return _result([] if self.empty else None)

    def info(self) -> ModelInfo:
        return ModelInfo(
            model="CATIE-AQ/NERmemberta-4entities",
            precision="fp16",
            device="cpu",
            device_requested="auto",
            labels=["PER", "LOC", "ORG", "MISC"],
            max_tokens=None,
            max_input_chars=200_000,
            loaded=False,
            package_version="0.1.0",
            attribution="CATIE-AQ (MIT)",
        )


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> type[FakeEngine]:
    FakeEngine.calls = []
    monkeypatch.setattr(cli, "NerEngine", FakeEngine)
    return FakeEngine


def _read_file(tmp_path, content: str = SENTENCE) -> str:
    path = tmp_path / "article.txt"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_piped_output_defaults_to_json(
    tmp_path, capsys: pytest.CaptureFixture[str], fake_engine: type[FakeEngine]
) -> None:
    assert cli.main(["extract", _read_file(tmp_path)]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["entities"]) == 2
    assert payload["entities"][1]["text"] == "Paris"


def test_file_and_stdin_extract_the_same_entities(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _read_file(tmp_path)
    assert cli.main(["extract", path]) == cli.EXIT_OK
    from_file = json.loads(capsys.readouterr().out)

    monkeypatch.setattr(
        sys, "stdin", types.SimpleNamespace(buffer=io.BytesIO(SENTENCE.encode("utf-8")))
    )
    assert cli.main(["extract", "-"]) == cli.EXIT_OK
    from_stdin = json.loads(capsys.readouterr().out)

    # Only the provenance differs; the analysis must not depend on how the text arrived.
    assert from_file["entities"] == from_stdin["entities"]
    assert from_file["source"] == path
    assert from_stdin["source"] is None


def test_utf8_bom_is_stripped_for_decoding(
    tmp_path, capsys: pytest.CaptureFixture[str], fake_engine: type[FakeEngine]
) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes("\ufeff".encode() + SENTENCE.encode("utf-8"))
    assert cli.main(["extract", str(path)]) == cli.EXIT_OK
    assert capsys.readouterr().out.startswith("{")
    # The BOM is consumed by decoding, so offsets match the file content the user sees.
    assert fake_engine.calls[0]["text"] == SENTENCE


def test_text_format_is_used_when_asked(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["extract", "--format", "text", _read_file(tmp_path)]) == cli.EXIT_OK
    assert "LABEL" in capsys.readouterr().out


def test_highlight_without_colour_is_readable(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["extract", "--format", "highlight", "--no-color", "--no-legend", _read_file(tmp_path)]
    assert cli.main(args) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "Sylvain Boyer (PER) est à Paris (LOC)."


def test_labels_are_parsed_and_forwarded(
    tmp_path, capsys: pytest.CaptureFixture[str], fake_engine: type[FakeEngine]
) -> None:
    args = ["extract", "--labels", "per, loc", _read_file(tmp_path)]
    assert cli.main(args) == cli.EXIT_OK
    assert fake_engine.calls[0]["labels"] == ["PER", "LOC"]


def test_unknown_label_is_a_clean_error(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["extract", "--labels", "PER,XYZ", _read_file(tmp_path)]) == cli.EXIT_ERROR
    assert "unknown labels" in capsys.readouterr().err


def test_no_merge_is_forwarded(
    tmp_path, capsys: pytest.CaptureFixture[str], fake_engine: type[FakeEngine]
) -> None:
    assert cli.main(["extract", "--no-merge", _read_file(tmp_path)]) == cli.EXIT_OK
    assert fake_engine.calls[0]["merge_adjacent"] is False


def test_stats_go_to_stderr_and_leave_stdout_clean(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["extract", "--stats", _read_file(tmp_path)]) == cli.EXIT_OK
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout is still valid JSON
    assert "chunks=1" in captured.err


def test_missing_file_exits_with_one(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["extract", "/nonexistent/article.txt"]) == cli.EXIT_ERROR
    assert "file not found" in capsys.readouterr().err


def test_empty_input_exits_with_one(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    assert cli.main(["extract", str(path)]) == cli.EXIT_ERROR
    assert "empty input" in capsys.readouterr().err


def test_fail_on_empty_exits_with_three(
    tmp_path, capsys: pytest.CaptureFixture[str], fake_engine: type[FakeEngine], monkeypatch
) -> None:
    monkeypatch.setattr(
        cli,
        "NerEngine",
        lambda settings: _EmptyEngine(settings),
    )
    assert cli.main(["extract", "--fail-on-empty", _read_file(tmp_path)]) == cli.EXIT_EMPTY
    capsys.readouterr()


def test_empty_result_without_the_flag_is_a_success(
    tmp_path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.setattr(cli, "NerEngine", lambda settings: _EmptyEngine(settings))
    assert cli.main(["extract", _read_file(tmp_path)]) == cli.EXIT_OK
    capsys.readouterr()


def test_unknown_argument_exits_with_two() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["extract", "--nope"])
    assert excinfo.value.code == 2


def test_info_prints_engine_details_without_loading(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["info"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "CATIE-AQ/NERmemberta-4entities" in out
    assert "unresolved" in out


def test_info_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["info", "--format", "json"]) == cli.EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["loaded"] is False
    assert payload["max_tokens"] is None


class _EmptyEngine(FakeEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def extract(self, text: str, **_kwargs) -> ExtractionResult:
        return _result([])
