"""MCP surface: three tools over the shared engine.

Nothing model-specific lives here. The tool signatures *are* the contract: the SDK derives the
input schema from type hints and the output schema from the returned Pydantic models.
"""

from __future__ import annotations

from mcp.server import MCPServer

from . import __version__
from .config import Settings
from .engine import EngineError, NerEngine
from .files import InputError, read_file
from .schemas import BatchResult, ExtractionResult, ModelInfo

MAX_BATCH_TEXTS = 64

INSTRUCTIONS = (
    "French named-entity recognition (PER, LOC, ORG, MISC). "
    "Use ner_extract for short inline text, ner_extract_file for a file on disk (the content "
    "never enters your context: pass the path, not the text), ner_batch for several short "
    "texts (max 64), ner_info to diagnose the engine. Offsets are Unicode code points into the "
    "exact text, and text[start:end] always equals the entity text."
)


def build_server(settings: Settings | None = None) -> MCPServer:
    """Wire the engine into an MCP server. Does not load the model until first use."""
    settings = settings or Settings.from_env()
    engine = NerEngine(settings)
    mcp = MCPServer("ner-fr", version=__version__, instructions=INSTRUCTIONS)

    @mcp.tool()
    def ner_extract(
        text: str,
        labels: list[str] | None = None,
        min_score: float = 0.0,
        merge_adjacent: bool = True,
        trim_punctuation: bool = True,
    ) -> ExtractionResult:
        """Extract named entities from French text.

        Args:
            text: French text to analyse. Texts longer than the model context are split
                automatically; `chunks` reports how many windows were used.
            labels: Subset of ["PER", "LOC", "ORG", "MISC"] to keep. Defaults to all four.
            min_score: Drop entities below this confidence (0..1).
            merge_adjacent: Merge neighbouring entities that share a label and are separated
                only by whitespace (e.g. "Paris 2024").
            trim_punctuation: Drop sentence punctuation glued to a span ("Grand Rex." ->
                "Grand Rex"). Set false to get the model's raw spans.
        """
        return engine.extract(
            text,
            labels=labels,
            min_score=min_score,
            merge_adjacent=merge_adjacent,
            trim_punctuation=trim_punctuation,
        )

    @mcp.tool()
    def ner_extract_file(
        path: str,
        labels: list[str] | None = None,
        min_score: float = 0.0,
        merge_adjacent: bool = True,
        trim_punctuation: bool = True,
        encoding: str | None = None,
    ) -> ExtractionResult:
        """Extract named entities from a French text file, read server-side.

        Prefer this over ner_extract for anything long: the file content is read locally and
        never travels through the conversation, so it costs no tokens. `source` in the result
        is the resolved absolute path.

        Args:
            path: File to analyse (absolute, or relative to the server's working directory).
            labels: Subset of ["PER", "LOC", "ORG", "MISC"] to keep. Defaults to all four.
            min_score: Drop entities below this confidence (0..1).
            merge_adjacent: Merge neighbouring same-label entities separated by whitespace.
            trim_punctuation: Drop sentence punctuation glued to a span.
            encoding: File encoding. Defaults to utf-8 with BOM detection.
        """
        try:
            text, resolved = read_file(
                path,
                encoding=encoding,
                max_chars=settings.max_input_chars,
                allowed_dirs=settings.allowed_dirs,
            )
        except InputError as exc:
            raise EngineError(str(exc)) from exc
        result = engine.extract(
            text,
            labels=labels,
            min_score=min_score,
            merge_adjacent=merge_adjacent,
            trim_punctuation=trim_punctuation,
        )
        return result.model_copy(update={"source": resolved})

    @mcp.tool()
    def ner_batch(
        texts: list[str],
        labels: list[str] | None = None,
        min_score: float = 0.0,
        merge_adjacent: bool = True,
        trim_punctuation: bool = True,
    ) -> BatchResult:
        """Extract entities from several French texts in one batched pass (max 64).

        A text that fails validation reports its own `error`; the rest of the batch is still
        processed. Results keep the input order via `index`.
        """
        if not texts:
            raise EngineError("texts must contain at least one entry")
        if len(texts) > MAX_BATCH_TEXTS:
            raise EngineError(f"too many texts: {len(texts)}, limit is {MAX_BATCH_TEXTS}")
        return engine.extract_batch(
            texts,
            labels=labels,
            min_score=min_score,
            merge_adjacent=merge_adjacent,
            trim_punctuation=trim_punctuation,
        )

    @mcp.tool()
    def ner_info() -> ModelInfo:
        """Report the model, device, context limit and loading state without loading it."""
        return engine.info()

    return mcp


def main() -> None:
    build_server().run()  # stdio is the default transport


if __name__ == "__main__":
    main()
