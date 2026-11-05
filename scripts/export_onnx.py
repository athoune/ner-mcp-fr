#!/usr/bin/env python
"""Export the NER checkpoint to ONNX (fp32), then to fp16.

This is the only place that needs torch. It is run by a maintainer, not by users:

    uv run --group export python scripts/export_onnx.py
    # writes dist/onnx/model-fp16.onnx, prints its size and SHA-256

Upload the resulting file to the artifact repository (see docs/PUBLISHING.md) and pin the
checksum in `src/ner_mcp/config.py` so a swapped artifact fails loudly instead of silently
changing predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path

OPSET = 17


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="CATIE-AQ/NERmembert-base-4entities")
    parser.add_argument("--out", type=Path, default=Path("dist/onnx/model-fp16.onnx"))
    parser.add_argument(
        "--precision", choices=("fp16", "fp32"), default="fp16", help="fp16 halves the download"
    )
    args = parser.parse_args()

    import torch
    from onnxconverter_common import float16
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model).eval()

    sample = tokenizer("Bonjour le monde", return_tensors="pt")
    args_tuple = (sample["input_ids"], sample["attention_mask"])
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch", 1: "sequence"},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model.onnx"
        torch.onnx.export(
            model,
            args_tuple,
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        if args.precision == "fp16":
            import onnx

            graph = onnx.load(str(fp32))
            # keep_io_types: onnxruntime CPU is happier with int64 inputs and fp32 logits,
            # and the conversion is then lossless from the caller's point of view.
            onnx.save(
                float16.convert_float_to_float16(graph, keep_io_types=True), str(args.out)
            )
        else:
            args.out.write_bytes(fp32.read_bytes())

    size = args.out.stat().st_size / 1048576
    print(f"{args.out}  {size:.0f} MB")
    print(f"sha256 {sha256(args.out)}")
    print("\nNow: upload it to the artifact repository, then set DEFAULT_ARTIFACT_SHA256 in")
    print("src/ner_mcp/config.py (and DEFAULT_ARTIFACT_FILE if the name differs).")


if __name__ == "__main__":
    main()
