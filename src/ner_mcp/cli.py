"""Command line interface.

    ner-mcp-fr                     run the MCP server on stdio (default)
    ner-mcp-fr mcp                 same, explicit
    ner-mcp-fr extract FILE        analyse a file
    cat FILE | ner-mcp-fr extract  analyse stdin
    ner-mcp-fr info                print engine details, do not load the model

The CLI is the verification surface: it shares `engine.py` with the MCP server, so what you see
here is exactly what an agent receives. stdout carries data only; logs and stats go to stderr.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import replace

from . import __version__
from .config import LABELS, PROFILES, VALID_DEVICES, VALID_LOG_LEVELS, ConfigError, Settings
from .engine import EngineError, NerEngine
from .files import InputError, read_file, read_stdin
from .render import (
    render_highlight,
    render_json,
    render_jsonl,
    render_markdown,
    render_stats,
    render_text,
)
from .schemas import ExtractionResult

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_EMPTY = 3  # --fail-on-empty and nothing found

FORMATS = ("json", "jsonl", "text", "highlight", "markdown")


class CliError(RuntimeError):
    """User-facing failure: printed as-is, exit code 1."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-mcp-fr",
        description="French NER: MCP server (default) and verification CLI.",
    )
    parser.add_argument("--version", action="version", version=f"ner-mcp-fr {__version__}")
    parser.add_argument(
        "--log-level",
        choices=VALID_LOG_LEVELS,
        default=None,
        help="override NER_LOG_LEVEL",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{mcp,extract,info}")

    subparsers.add_parser("mcp", help="run the MCP server on stdio (default when no command)")

    extract = subparsers.add_parser("extract", help="analyse a file or stdin and print entities")
    extract.add_argument(
        "path",
        nargs="?",
        default="-",
        help="file to analyse, '-' for stdin (default: stdin)",
    )
    extract.add_argument(
        "-f",
        "--format",
        choices=FORMATS,
        default=None,
        help="output format (default: 'text' on a terminal, 'json' when piped)",
    )
    extract.add_argument(
        "--labels",
        default=None,
        metavar="PER,LOC",
        help=f"comma-separated subset of {','.join(LABELS)} (default: all)",
    )
    extract.add_argument(
        "--min-score", type=float, default=None, help="drop entities below this score"
    )
    extract.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="do not merge adjacent same-label entities",
    )
    extract.add_argument(
        "--keep-punctuation",
        dest="trim_punctuation",
        action="store_false",
        help="keep the model's raw spans, trailing punctuation included",
    )
    extract.add_argument("--profile", choices=tuple(PROFILES), default=None)
    extract.add_argument("--model", default=None, help="explicit Hugging Face model id")
    extract.add_argument("--device", choices=VALID_DEVICES, default=None)
    extract.add_argument(
        "--encoding",
        default=None,
        help="input encoding (default: utf-8, BOM detected)",
    )
    extract.add_argument(
        "--color",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force or disable ANSI colours in highlight format",
    )
    extract.add_argument("--no-legend", action="store_true", help="hide the colour legend")
    extract.add_argument(
        "--stats",
        action="store_true",
        help="write chunks, timing and drop counts to stderr",
    )
    extract.add_argument(
        "--fail-on-empty",
        action="store_true",
        help=f"exit with {EXIT_EMPTY} when no entity is found",
    )

    info = subparsers.add_parser("info", help="print engine and model details without loading")
    info.add_argument("--format", choices=("text", "json"), default="text")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _configure_logging(args.log_level)
        if args.command == "extract":
            return _run_extract(args)
        if args.command == "info":
            return _run_info(args)
        return _run_mcp(args)
    except (CliError, ConfigError, EngineError, InputError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # e.g. `... | head`; not an error
        return EXIT_OK


def _configure_logging(level: str | None) -> None:
    resolved = level or os.environ.get("NER_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        stream=sys.stderr,  # stdout is the wire (MCP) or the data (extract)
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _run_mcp(args: argparse.Namespace) -> int:
    from .server import build_server  # imported late: keeps `info` fast

    logging.getLogger("ner_mcp.cli").info(
        "serving MCP over stdio (stdout is the protocol, logs go here); Ctrl-C to stop"
    )
    build_server(Settings.from_env()).run()  # stdio by default
    return EXIT_OK


def _run_extract(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    # Read the input before loading a 400 MB model: a missing file must fail immediately.
    text, source = _read_text(args.path, args.encoding, settings)

    engine = NerEngine(settings)
    result = engine.extract(
        text,
        labels=_parse_labels(args.labels),
        merge_adjacent=args.merge,
        trim_punctuation=args.trim_punctuation,
    )
    if source is not None:
        result = result.model_copy(update={"source": source})

    fmt = args.format or ("text" if sys.stdout.isatty() else "json")
    color = args.color
    if color is None:
        color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    output = _render(fmt, result, text, color=color, legend=not args.no_legend)
    if output:
        print(output)

    if args.stats:
        print(render_stats(result), file=sys.stderr)

    if args.fail_on_empty and not result.entities:
        return EXIT_EMPTY
    return EXIT_OK


def _run_info(args: argparse.Namespace) -> int:
    info = NerEngine(Settings.from_env()).info()
    if args.format == "json":
        print(info.model_dump_json(indent=2))
        return EXIT_OK

    rows = [
        ("model", info.model),
        ("profile", info.profile),
        ("device", _device_label(info)),
        ("dtype", info.dtype or "from checkpoint (model not loaded)"),
        (
            "context",
            f"{info.max_tokens} tokens" if info.max_tokens else "unknown (model not loaded)",
        ),
        ("labels", ", ".join(info.labels)),
        ("max input", f"{info.max_input_chars} code points"),
        ("loaded", "yes" if info.loaded else "no (loads on first extraction)"),
        ("cache", info.cache_dir or "Hugging Face default"),
        ("version", info.package_version),
        ("license", f"MIT - {info.attribution}"),
    ]
    width = max(len(key) for key, _ in rows)
    for key, value in rows:
        print(f"{key.ljust(width)}  {value}")
    return EXIT_OK


def _device_label(info) -> str:
    if info.loaded:
        return f"{info.device_requested} -> {info.device}"
    return f"{info.device_requested} (unresolved)"


def _settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    updates: dict[str, object] = {}
    if args.profile:
        updates["profile"] = args.profile
        if not args.model:
            updates["model_id"] = PROFILES[args.profile]
    if args.model:
        updates["model_id"] = args.model
    if args.device:
        updates["device"] = args.device
    if args.min_score is not None:
        if not 0.0 <= args.min_score <= 1.0:
            raise CliError(f"--min-score must be between 0 and 1, got {args.min_score}")
        updates["min_score"] = args.min_score
    return replace(settings, **updates) if updates else settings


def _parse_labels(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    labels = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not labels:
        raise CliError("--labels is empty; expected e.g. --labels PER,LOC")
    unknown = [label for label in labels if label not in LABELS]
    if unknown:
        raise CliError(f"unknown labels {unknown}; expected a subset of {', '.join(LABELS)}")
    return labels


def _read_text(
    path: str | None, encoding: str | None, settings: Settings
) -> tuple[str, str | None]:
    """Read stdin or a file. Returns `(text, source)`; `source` is None for stdin."""
    if path in (None, "-"):
        return read_stdin(), None
    return read_file(
        path,
        encoding=encoding,
        max_chars=settings.max_input_chars,
        allowed_dirs=settings.allowed_dirs,
    )


def _render(
    fmt: str,
    result: ExtractionResult,
    source: str,
    *,
    color: bool,
    legend: bool,
) -> str:
    if fmt == "json":
        return render_json(result)
    if fmt == "jsonl":
        return render_jsonl(result)
    if fmt == "text":
        return render_text(result)
    if fmt == "markdown":
        return render_markdown(result, source)
    if fmt == "highlight":
        return render_highlight(result, source, color=color, legend=legend)
    raise CliError(f"unknown format: {fmt}")


if __name__ == "__main__":
    sys.exit(main())
