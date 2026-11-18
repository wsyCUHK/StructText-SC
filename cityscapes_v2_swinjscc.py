#!/usr/bin/env python3
"""Decode Cityscapes sender maps through the trained SwinJSCC latent AWGN link.

The CPU preflight checks the checkpoint and exact network source without CUDA.
Inference requires CUDA and writes the same map contract as the packet receiver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from cityscapes_v2_transport import by_unique_id, read_jsonl
from swinjscc_common import MODEL, PALETTE, model_config, nearest_train_ids


MAP_MODE = "swinjscc_awgn_latent"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_payload(payload: dict, repo_root: Path) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
        raise ValueError("Expected a SwinJSCC checkpoint with model_state")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint lacks training configuration")
    if (config.get("model") != MODEL or config.get("model_size") not in {"small", "base"}
            or config.get("channel") != "awgn" or config.get("distortion") != "RGB_MSE"):
        raise ValueError("Checkpoint architecture/channel differs from this adapter")
    for name in ("C", "snr_db", "width", "height", "crop_size"):
        if not isinstance(config.get(name), int):
            raise ValueError(f"Checkpoint field {name} must be an integer")
    if (config["C"] <= 0 or config["C"] % 2
            or config["width"] <= 0 or config["height"] <= 0
            or config["width"] % 128 or config["height"] % 128
            or config["crop_size"] <= 0 or config["crop_size"] % 128):
        raise ValueError("Checkpoint dimensions or C are invalid")
    network = repo_root / "net" / "network.py"
    distortion = repo_root / "loss" / "distortion.py"
    if not network.is_file() or not distortion.is_file():
        raise FileNotFoundError(f"SwinJSCC net/ and loss/ missing under {repo_root}")
    actual_hash = sha256_file(network)
    if actual_hash != config.get("network_sha256"):
        raise ValueError("net/network.py SHA-256 differs from the training checkpoint")
    if not isinstance(payload.get("step"), int) or payload["step"] <= 0:
        raise ValueError("Checkpoint step is missing or invalid")
    return config


def load_model(checkpoint: Path, repo_root: Path):
    import torch

    saved = torch.load(checkpoint, map_location="cpu")
    config = validate_checkpoint_payload(saved, repo_root)
    sys.path.insert(0, str(repo_root.resolve()))
    from net.network import SwinJSCC

    args = SimpleNamespace(model_size=config["model_size"], C=config["C"],
                           snr=config["snr_db"], crop_size=config["crop_size"])
    network_args, network_config = model_config(torch, args)
    model = SwinJSCC(network_args, network_config)
    model.load_state_dict(saved["model_state"], strict=True)
    metadata = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": saved["step"],
        "training_config": config,
        "network_path": str((repo_root / "net" / "network.py").resolve()),
        "torch_version": torch.__version__,
        "strict_load": True,
    }
    return model, metadata


def infer(run_dir: Path, output_dir: Path, model, metadata: dict,
          snr_db: float, seed: int, receiver_map_variant: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SwinJSCC inference needs an available CUDA GPU")
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Packet transport must be prepared first: {output_dir}")
    for name in ("maps", "maps_raw", "maps_palette", "maps_trainIds", "map_stats.jsonl"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Will not overwrite: {output_dir / name}")
    config = metadata["training_config"]
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    by_unique_id(manifest, "manifest")
    expected_size = (config["width"], config["height"])
    for row in manifest:
        sample_id = row["id"]
        if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
            raise ValueError(f"Invalid sample ID: {sample_id}")
        if not Path(row["condition_map"]).is_file():
            raise FileNotFoundError(row["condition_map"])
        if tuple(row["model_size"]) != expected_size:
            raise ValueError(f"Model size differs from checkpoint for {sample_id}")

    for name in ("maps", "maps_raw", "maps_palette", "maps_trainIds"):
        (output_dir / name).mkdir(parents=True, exist_ok=False)
    model = model.cuda().eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows = []
    with torch.no_grad():
        for index, row in enumerate(manifest):
            sample_id = row["id"]
            source_map = Path(row["condition_map"])
            with Image.open(source_map) as source:
                if source.size != expected_size:
                    raise ValueError(f"Input size differs from checkpoint for {sample_id}")
                source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
            image = np.ascontiguousarray(source_rgb.transpose(2, 0, 1),
                                          dtype=np.float32) / 255.0
            tensor = torch.from_numpy(image).unsqueeze(0).cuda()
            torch.manual_seed(seed + index)
            torch.cuda.manual_seed_all(seed + index)
            reconstructed, cbr, _, _, _ = model(tensor, snr_db, config["C"])
            raw = (reconstructed[0].clamp(0., 1.).permute(1, 2, 0)
                   .mul(255).round().to(torch.uint8).cpu().numpy())
            train_ids = nearest_train_ids(raw)
            palette = PALETTE[train_ids]
            raw_path = output_dir / "maps_raw" / f"{sample_id}.png"
            palette_path = output_dir / "maps_palette" / f"{sample_id}.png"
            ids_path = output_dir / "maps_trainIds" / f"{sample_id}.png"
            receiver_path = output_dir / "maps" / f"{sample_id}.png"
            Image.fromarray(raw, mode="RGB").save(raw_path)
            Image.fromarray(palette, mode="RGB").save(palette_path)
            Image.fromarray(train_ids, mode="L").save(ids_path)
            receiver = raw if receiver_map_variant == "raw" else palette
            Image.fromarray(receiver, mode="RGB").save(receiver_path)
            cbr_value = float(cbr)
            channel_uses = cbr_value * tensor.numel()
            if not math.isfinite(channel_uses) or abs(channel_uses - round(channel_uses)) > 1e-4:
                raise ValueError(f"Invalid complex channel use count for {sample_id}")
            rows.append({
                "id": sample_id,
                "sender_map": str(source_map.resolve()),
                "receiver_map": str(receiver_path.resolve()),
                "map_raw": str(raw_path.resolve()),
                "map_palette": str(palette_path.resolve()),
                "map_train_ids": str(ids_path.resolve()),
                "map_channel_mode": MAP_MODE,
                "map_postprocess": receiver_map_variant,
                "snr_db": snr_db,
                "map_seed": seed + index,
                "cbr": cbr_value,
                "map_channel_uses": int(round(channel_uses)),
                "map_channel_use_unit": "complex_symbol",
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            })
            print(json.dumps({"id": sample_id, "map_channel_uses": rows[-1]["map_channel_uses"]}), flush=True)
    with (output_dir / "map_stats.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "sample_count": len(rows),
        "map_mode": MAP_MODE,
        "receiver_map_variant": receiver_map_variant,
        "snr_db": snr_db,
        "map_seed": seed,
        "checkpoint_sha256": metadata["checkpoint_sha256"],

    }
    (output_dir / "map_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "swinjscc_config.json").write_text(
        json.dumps({**metadata, "inference": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--preflight", action="store_true",
                        help="Strict-load checkpoint on CPU without CUDA or outputs")
    parser.add_argument("--run-dir", type=Path,
                        help="Prepared Cityscapes data run containing manifest.jsonl")
    parser.add_argument("--output-dir", type=Path,
                        help="Existing packet transport directory; maps subdirectories must be new")
    parser.add_argument("--snr-db", type=float,
                        help="Latent AWGN SNR; defaults to checkpoint training SNR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--receiver-map-variant", choices=("raw", "palette"), default="raw")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint missing: {args.checkpoint}")
    model, metadata = load_model(args.checkpoint, args.repo_root)
    if args.preflight:
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
        return
    if args.run_dir is None or args.output_dir is None:
        parser.error("--run-dir and --output-dir are required for inference")
    snr_db = metadata["training_config"]["snr_db"] if args.snr_db is None else args.snr_db
    if not math.isfinite(snr_db) or not -100 <= snr_db <= 100:
        parser.error("--snr-db must be finite and between -100 and 100")
    random.seed(args.seed)
    np.random.seed(args.seed)
    infer(args.run_dir, args.output_dir, model, metadata, snr_db,
          args.seed, args.receiver_map_variant)


if __name__ == "__main__":
    main()
