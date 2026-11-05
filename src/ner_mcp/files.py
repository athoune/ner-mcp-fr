"""Input reading shared by the CLI and the MCP file tool.

Reading a file server-side is the whole point of `ner_extract_file`: the text never travels
through the model context, so a 200 kB article costs nothing in tokens.
"""

from __future__ import annotations

import codecs
import sys
from pathlib import Path

# UTF-8 never uses more than 4 bytes per character. Used to reject huge files before reading.
MAX_BYTES_PER_CHAR = 4


class InputError(ValueError):
    """User-facing input problem: the message is meant to be shown as-is."""


def decode_bytes(raw: bytes, *, origin: str, encoding: str | None = None) -> str:
    """Decode bytes without ever replacing characters silently."""
    if not raw.strip():
        raise InputError(f"empty input from {origin}")
    resolved = encoding or ("utf-8-sig" if raw.startswith(codecs.BOM_UTF8) else "utf-8")
    try:
        return raw.decode(resolved)
    except (UnicodeDecodeError, LookupError) as exc:
        raise InputError(
            f"cannot decode {origin} as {resolved}: {exc}. Try --encoding latin-1."
        ) from exc


def read_stdin() -> str:
    return decode_bytes(sys.stdin.buffer.read(), origin="stdin")


def read_file(
    path: str,
    *,
    encoding: str | None = None,
    max_chars: int | None = None,
    allowed_dirs: tuple[str, ...] = (),
) -> tuple[str, str]:
    """Read and decode a file. Returns `(text, resolved_path)`.

    `allowed_dirs` is opt-in (see NER_ALLOWED_DIRS): the server runs with the user's
    permissions, so the default is the same trust domain as the agent's shell.
    """
    file = Path(path).expanduser().resolve()
    if not file.exists():
        raise InputError(f"file not found: {file}")
    if file.is_dir():
        raise InputError(f"not a file: {file}")
    if allowed_dirs and not _is_allowed(file, allowed_dirs):
        raise InputError(f"path outside NER_ALLOWED_DIRS: {file}")

    try:
        size = file.stat().st_size
    except OSError as exc:
        raise InputError(f"cannot stat {file}: {exc}") from exc
    if max_chars is not None and size > max_chars * MAX_BYTES_PER_CHAR:
        raise InputError(
            f"file too large: {size} bytes, limit is {max_chars} characters "
            "(NER_MAX_INPUT_CHARS)"
        )

    try:
        raw = file.read_bytes()
    except OSError as exc:
        raise InputError(f"cannot read {file}: {exc}") from exc

    text = decode_bytes(raw, origin=str(file), encoding=encoding)
    if max_chars is not None and len(text) > max_chars:
        raise InputError(
            f"file too long: {len(text)} characters, limit is {max_chars} "
            "(NER_MAX_INPUT_CHARS)"
        )
    return text, str(file)


def _is_allowed(file: Path, allowed_dirs: tuple[str, ...]) -> bool:
    for directory in allowed_dirs:
        root = Path(directory).expanduser().resolve()
        if file == root or root in file.parents:
            return True
    return False
