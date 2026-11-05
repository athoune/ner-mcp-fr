"""Wire schemas shared by the MCP tools and the CLI.

These are the contract: the MCP SDK derives each tool's `output_schema` from them, and the CLI
serializes the very same models. Anything the agent sees is what the CLI prints.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["PER", "LOC", "ORG", "MISC"]


class Entity(BaseModel):
    """One named entity, with offsets into the *original* input text."""

    text: str = Field(description="Exact substring of the input text.")
    label: Label = Field(description="PER (person), LOC (place), ORG (organisation) or MISC.")
    start: int = Field(ge=0, description="Start offset, in Unicode code points.")
    end: int = Field(ge=0, description="End offset, exclusive, in Unicode code points.")
    score: float = Field(ge=0.0, le=1.0, description="Mean softmax probability of the entity.")


class ExtractionResult(BaseModel):
    """Result of a single extraction."""

    model: str
    profile: str
    device: str
    entities: list[Entity]
    source: str | None = Field(
        default=None,
        description="Origin of the text: absolute file path, or null for inline text.",
    )
    text_length: int = Field(ge=0, description="Length of the input in code points.")
    chunks: int = Field(ge=1, description="Inference windows actually run.")
    truncated: bool = Field(description="True only if a hard input limit was hit.")
    dropped_by_min_score: int = Field(ge=0)
    elapsed_ms: int = Field(ge=0)


class BatchItem(BaseModel):
    """One entry of a batch. A bad entry reports `error` instead of failing the batch."""

    index: int = Field(ge=0)
    text_length: int = Field(ge=0)
    entities: list[Entity] = Field(default_factory=list)
    chunks: int = 0
    error: str | None = None


class BatchResult(BaseModel):
    model: str
    profile: str
    device: str
    results: list[BatchItem]
    elapsed_ms: int = Field(ge=0)


class ModelInfo(BaseModel):
    """Everything needed to diagnose a surprising extraction without a shell."""

    model: str
    profile: str
    device: str = Field(description="Device actually used, or the request while unloaded.")
    device_requested: str = Field(description="Configured device: auto, cpu, mps or cuda.")
    dtype: str | None = Field(
        default=None,
        description="Weight dtype actually in use. Comes from the checkpoint, not from config.",
    )
    labels: list[str]
    max_tokens: int | None = Field(
        default=None, description="Model context limit in tokens; None until loaded."
    )
    max_input_chars: int
    loaded: bool
    package_version: str
    license: str = "MIT"
    attribution: str
    cache_dir: str | None = None
    project_url: str | None = None
