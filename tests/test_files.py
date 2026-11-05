"""File input: the server reads bytes, the caller gets exact text or a clear error."""

from __future__ import annotations

import pytest

from ner_mcp.files import InputError, decode_bytes, read_file

TEXT = "Le Grand Rex accueille Sylvain Boyer à Paris."


def test_read_file_returns_text_and_resolved_path(tmp_path) -> None:
    path = tmp_path / "article.txt"
    path.write_text(TEXT, encoding="utf-8")
    text, resolved = read_file(str(path))
    assert text == TEXT
    assert resolved == str(path.resolve())


def test_utf8_bom_is_consumed(tmp_path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes("\ufeff".encode() + TEXT.encode("utf-8"))
    text, _ = read_file(str(path))
    assert text == TEXT


def test_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(InputError, match="file not found"):
        read_file(str(tmp_path / "nope.txt"))


def test_directory_is_rejected(tmp_path) -> None:
    with pytest.raises(InputError, match="not a file"):
        read_file(str(tmp_path))


def test_empty_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    with pytest.raises(InputError, match="empty input"):
        read_file(str(path))


def test_oversized_file_is_rejected_before_reading(tmp_path) -> None:
    path = tmp_path / "big.txt"
    path.write_text("a" * 500, encoding="utf-8")
    with pytest.raises(InputError, match="too large"):
        read_file(str(path), max_chars=100)


def test_undecodable_file_fails_loudly(tmp_path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("déjà vu".encode("latin-1"))
    with pytest.raises(InputError, match="cannot decode"):
        read_file(str(path))


def test_latin1_can_be_requested_explicitly(tmp_path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("déjà vu".encode("latin-1"))
    text, _ = read_file(str(path), encoding="latin-1")
    assert text == "déjà vu"


def test_allowed_dirs_restricts_reads(tmp_path) -> None:
    inside = tmp_path / "docs"
    inside.mkdir()
    path = inside / "article.txt"
    path.write_text(TEXT, encoding="utf-8")

    text, _ = read_file(str(path), allowed_dirs=(str(tmp_path),))
    assert text == TEXT

    with pytest.raises(InputError, match="outside NER_ALLOWED_DIRS"):
        read_file(str(path), allowed_dirs=(str(tmp_path / "elsewhere"),))


def test_decode_bytes_never_replaces_characters() -> None:
    with pytest.raises(InputError):
        decode_bytes(b"\xff\xfe invalid", origin="test")
