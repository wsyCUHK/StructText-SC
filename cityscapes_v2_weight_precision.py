#!/usr/bin/env python3
"""Inspect safetensors headers before comparing BF16 and quantized image backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MAX_HEADER_BYTES = 64 * 1024 * 1024
ALLOWED_BF16_PROFILE_DTYPES = {"BF16", "F32", "F64", "I32", "I64", "BOOL"}


def inspect_safetensors(path: Path) -> dict:
    if path.suffix.lower() != ".safetensors":
        raise ValueError(f"BF16 profile requires safetensors, got: {path}")
    with path.open("rb") as source:
        header_bytes = source.read(8)
        if len(header_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = int.from_bytes(header_bytes, "little")
        if not 2 <= header_size <= MAX_HEADER_BYTES or header_size > path.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(source.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors tensor table: {path}")
    counts: dict[str, int] = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or not isinstance(tensor.get("dtype"), str):
            raise ValueError(f"Invalid safetensors tensor entry in {path}: {name}")
        shape = tensor.get("shape")
        if not isinstance(shape, list) or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError(f"Invalid safetensors tensor shape in {path}: {name}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        counts[tensor["dtype"]] = counts.get(tensor["dtype"], 0) + elements
    if not counts:
        raise ValueError(f"No tensors in safetensors file: {path}")
    return {"path": str(path.resolve()), "elements_by_dtype": counts}


def verify_bf16_weights(diffusion_model: Path, llm: Path, vae: Path) -> dict:
    summaries = {}
    for name, path in (("diffusion_model", diffusion_model), ("llm", llm), ("vae", vae)):
        summary = inspect_safetensors(path)
        counts = summary["elements_by_dtype"]
        lower_precision = set(counts) - ALLOWED_BF16_PROFILE_DTYPES
        if counts.get("BF16", 0) == 0 or lower_precision:
            raise ValueError(
                f"{name} is not BF16-aligned: {path}; element dtypes={counts}. "
                "Use BF16 weights, or explicitly choose precision profile mixed for a separate experiment."
            )
        summaries[name] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffusion-model", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_bf16_weights(args.diffusion_model, args.llm, args.vae),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
