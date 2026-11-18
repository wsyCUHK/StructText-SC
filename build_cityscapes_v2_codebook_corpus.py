#!/usr/bin/env python3
"""Freeze a shared Huffman corpus from a Cityscapes *train* caption run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cityscapes_v2_transport import by_unique_id, read_jsonl


def build_corpus(train_run_dir: Path, output: Path) -> None:
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FileExistsError(f"Corpus output already exists: {output}")
    manifest = by_unique_id(read_jsonl(train_run_dir / "manifest.jsonl"), "manifest")
    captions = by_unique_id(read_jsonl(train_run_dir / "captions.jsonl"), "caption")
    if set(manifest) != set(captions):
        raise ValueError("Train manifest and caption IDs differ")
    if any(row.get("split") != "train" for row in manifest.values()):
        raise ValueError("Codebook corpus must come entirely from Cityscapes train")
    if any(not row.get("caption") for row in captions.values()):
        raise ValueError("Empty training caption")
    content = "\n".join(captions[sample_id]["caption"] for sample_id in sorted(captions)) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Store exact UTF-8 bytes on every OS; the hash is over these transmitted
    # corpus bytes, so Windows newline conversion would invalidate metadata.
    output.write_bytes(content.encode("utf-8"))
    metadata = {
        "source_run_dir": str(train_run_dir.resolve()),
        "sample_count": len(captions),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "split": "train",
        "sorted_ids": sorted(captions),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Frozen {len(captions)} train captions to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_corpus(args.train_run_dir, args.output)


if __name__ == "__main__":
    main()
