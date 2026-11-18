#!/usr/bin/env python3
"""Build sender-side Cityscapes condition maps from a frozen DRN-D-105."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from drn_sender_model import load_model, make_predictor
from swinjscc_common import PALETTE


def materialize_sender_maps(run_dir: Path, predictor) -> int:
    manifest_path = run_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate sender manifest")
    for row in rows:
        if row.get("mask_source") != "DRN-D-105 predicted":
            raise ValueError(f"Manifest is not in DRN sender mode: {row['id']}")
        source = Path(row["source"])
        label_path = Path(row["sender_train_ids"])
        color_path = Path(row["condition_map"])
        if (label_path.parent.resolve() != (run_dir / "sender_drn_train_ids").resolve() or
                color_path.parent.resolve() != (run_dir / "samples" / row["id"]).resolve() or
                label_path.exists() or color_path.exists()):
            raise ValueError(f"Unexpected or existing DRN sender output: {row['id']}")
        with Image.open(source) as image:
            size = image.size
        if size != tuple(row["model_size"]):
            raise ValueError(f"Sender source size mismatch: {row['id']}")
        train_ids = np.asarray(predictor(source))
        if (train_ids.dtype != np.uint8 or train_ids.shape != (size[1], size[0]) or
                np.any(train_ids > 18)):
            raise ValueError(f"Invalid DRN trainId prediction: {row['id']}")
        Image.fromarray(train_ids, mode="L").save(label_path)
        Image.fromarray(PALETTE[train_ids], mode="RGB").save(color_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--multi-scale", action="store_true")
    args = parser.parse_args()
    profile_path = args.run_dir / "sender_drn_profile.json"
    if profile_path.exists():
        raise FileExistsError(profile_path)
    model, device = load_model(args.repo_root, args.checkpoint, args.device)
    count = materialize_sender_maps(
        args.run_dir, make_predictor(model, device, args.multi_scale)
    )
    profile_path.write_text(json.dumps({
        "mask_source": "DRN-D-105 predicted",
        "checkpoint": str(args.checkpoint.resolve()),
        "source": "resized RGB in manifest.source",
        "train_ids": "Cityscapes 0..18",
        "palette": "Cityscapes 19-class palette from swinjscc_common.py",
        "device": str(device), "multi_scale": args.multi_scale,
        "sample_count": count,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {count} DRN sender maps in {args.run_dir}", flush=True)


if __name__ == "__main__":
    main()
