#!/usr/bin/env python3
"""Pair packet-loss captions with pixel-AWGN or precomputed SwinJSCC maps.

Pixel AWGN is an uncoded surrogate; SwinJSCC maps come from a separate adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cityscapes_v2_transport import by_unique_id, load_shared_corpus, read_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(run_dir: Path, output_dir: Path, corpus_path: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Will not overwrite: {output_dir}")
    _, digest = load_shared_corpus(corpus_path, require_train_metadata=True)
    if corpus_path.resolve() == (run_dir / "captions.jsonl").resolve():
        raise ValueError("Evaluation captions cannot train their own codebook")
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    by_unique_id(manifest, "manifest")
    captions = by_unique_id(read_jsonl(run_dir / "captions.jsonl"), "caption")
    if set(captions) != {row["id"] for row in manifest}:
        raise ValueError("Manifest and caption IDs differ")
    lines = []
    ids = []
    for row in manifest:
        sample_id = row["id"]
        if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
            raise ValueError(f"Invalid sample ID: {sample_id}")
        if not Path(row["condition_map"]).is_file():
            raise FileNotFoundError(row["condition_map"])
        caption = " ".join(captions[sample_id]["caption"].split())
        if not caption:
            raise ValueError(f"Empty sender caption: {sample_id}")
        lines.append(caption)
        ids.append({"id": sample_id, "sender_caption": caption})
    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "sender_ids.jsonl", ids)
    (output_dir / "sender_lines.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "codebook.json").write_text(json.dumps({
        "corpus": str(corpus_path.resolve()), "sha256": digest,
        "train_metadata_verified": True, "sample_count": len(ids),
    }, indent=2), encoding="utf-8")
    print(f"Prepared {len(ids)} captions for packet channel in {output_dir}")


def awgn_rgb_map(source: Path, destination: Path, snr_db: float, seed: int) -> dict:
    if not np.isfinite(snr_db) or not -100 <= snr_db <= 100:
        raise ValueError("SNR must be finite and between -100 and 100 dB")
    with Image.open(source) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    signal = pixels / 127.5 - 1.0
    signal_power = float(np.mean(np.square(signal)))
    if signal_power <= 0:
        raise ValueError(f"Cannot define signal-relative SNR for zero-power map: {source}")
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.default_rng(seed).normal(
        0.0, np.sqrt(target_noise_power), size=signal.shape
    )
    noisy = signal + noise
    realized_noise_power = float(np.mean(np.square(noise)))
    clipped_fraction = float(np.mean((noisy < -1.0) | (noisy > 1.0)))
    quantized = np.rint((np.clip(noisy, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(quantized, mode="RGB").save(destination)
    return {
        "snr_db_target": snr_db,
        "snr_db_realized_before_clipping": float(10 * np.log10(signal_power / realized_noise_power)),
        "signal_power": signal_power,
        "target_noise_power": target_noise_power,
        "realized_noise_power": realized_noise_power,
        "clipped_scalar_fraction": clipped_fraction,
        "pixel_scalar_count": int(signal.size),
        "map_channel_uses": None,
        "map_channel_mode": "uncoded_rgb_pixel_awgn_surrogate",
    }


def finalize(run_dir: Path, output_dir: Path, snr_db: float, seed: int,
             map_stats_path: Path | None = None) -> None:
    if (output_dir / "receiver.jsonl").exists():
        raise FileExistsError(output_dir / "receiver.jsonl")
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    ids = read_jsonl(output_dir / "sender_ids.jsonl")
    if [row["id"] for row in manifest] != [row["id"] for row in ids]:
        raise ValueError("Manifest order changed after packet input preparation")
    recovered = (output_dir / "recovered_lines.txt").read_text(encoding="utf-8").splitlines()
    summary = json.loads((output_dir / "packet_summary.json").read_text(encoding="utf-8"))
    codebook = json.loads((output_dir / "codebook.json").read_text(encoding="utf-8"))
    corpus_path = Path(codebook["corpus"])
    _, digest = load_shared_corpus(corpus_path, require_train_metadata=True)
    if digest != codebook["sha256"] or Path(summary["codebook_corpus"]).resolve() != corpus_path:
        raise ValueError("Packet channel did not use the frozen codebook corpus")
    per_caption = summary["per_caption"]
    if (len(recovered) != len(ids) or len(per_caption) != len(ids)
            or summary["caption_count"] != len(ids)):
        raise ValueError("Packet output count differs from input manifest")
    if any(row["caption_index"] != index for index, row in enumerate(per_caption)):
        raise ValueError("Packet summary order differs from input manifest")
    for row in manifest:
        if not Path(row["condition_map"]).is_file():
            raise FileNotFoundError(row["condition_map"])
    precomputed_maps = None
    if map_stats_path is not None:
        precomputed_maps = read_jsonl(map_stats_path)
        if [row["id"] for row in precomputed_maps] != [row["id"] for row in manifest]:
            raise ValueError("SwinJSCC map IDs/order differ from input manifest")
        for index, (source_row, map_row) in enumerate(zip(manifest, precomputed_maps)):
            map_path = Path(map_row["receiver_map"]).resolve()
            if (map_path.parent != (output_dir / "maps").resolve()
                    or not map_path.is_file()
                    or Path(map_row["sender_map"]).resolve()
                    != Path(source_row["condition_map"]).resolve()
                    or map_row["map_channel_mode"] != "swinjscc_awgn_latent"
                    or map_row["snr_db"] != snr_db
                    or map_row["map_seed"] != seed + index):
                raise ValueError(f"Invalid SwinJSCC map record: {map_row['id']}")
    else:
        (output_dir / "maps").mkdir()
    (output_dir / "receiver_text").mkdir()
    receiver_rows = []
    stats_rows = []
    for index, row in enumerate(manifest):
        sample_id = row["id"]
        if precomputed_maps is None:
            map_path = output_dir / "maps" / f"{sample_id}.png"
            map_stats = awgn_rgb_map(Path(row["condition_map"]), map_path,
                                     snr_db, seed + index)
        else:
            map_stats = precomputed_maps[index]
            map_path = Path(map_stats["receiver_map"]).resolve()
        text_path = output_dir / "receiver_text" / f"{sample_id}.txt"
        text_path.write_text(recovered[index] + "\n", encoding="utf-8")
        receiver_rows.append({
            "id": sample_id, "caption": recovered[index],
            "caption_path": str(text_path.resolve()),
            "condition_map": str(map_path.resolve()),
            "text_link": "huffman_rs_packet_loss",
            "map_link": map_stats["map_channel_mode"],
        })
        stats_rows.append({
            "id": sample_id, "sender_utf8_bits": len(ids[index]["sender_caption"].encode("utf-8")) * 8,
            **per_caption[index], **map_stats,
        })
    write_jsonl(output_dir / "receiver.jsonl", receiver_rows)
    write_jsonl(output_dir / "stats.jsonl", stats_rows)
    (output_dir / "config.json").write_text(json.dumps({
        "text_mode": "huffman_rs_packet_loss",
        "packet_summary": str((output_dir / "packet_summary.json").resolve()),
        "map_mode": "swinjscc_awgn_latent" if precomputed_maps is not None
                    else "uncoded_rgb_pixel_awgn_surrogate",
        "map_snr_db": snr_db, "map_seed": seed,
        "map_stats_path": str(map_stats_path.resolve()) if map_stats_path else None,
        "note": "SwinJSCC maps are decoded after latent AWGN; pixel-AWGN mode is an "
                "uncoded RGB surrogate. Both are simulations, not a digital modem.",
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(receiver_rows)} packet receiver records to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-dir", required=True, type=Path)
    prep.add_argument("--output-dir", required=True, type=Path)
    prep.add_argument("--codebook-corpus", required=True, type=Path)
    finish = sub.add_parser("finalize")
    finish.add_argument("--run-dir", required=True, type=Path)
    finish.add_argument("--output-dir", required=True, type=Path)
    finish.add_argument("--map-snr-db", required=True, type=float)
    finish.add_argument("--map-seed", required=True, type=int)
    finish.add_argument("--map-stats-jsonl", type=Path,
                        help="Precomputed SwinJSCC maps/stats; omit for pixel AWGN")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_dir, args.output_dir, args.codebook_corpus)
    else:
        finalize(args.run_dir, args.output_dir, args.map_snr_db, args.map_seed,
                 args.map_stats_jsonl)


if __name__ == "__main__":
    main()
