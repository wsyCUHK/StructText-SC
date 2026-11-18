#!/usr/bin/env python3
"""Generate Cityscapes captions with a local BLIP captioning checkpoint.

BLIP sees only each manifest row's original RGB image. The GT segmentation
map remains a separate sender-side input for the image channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from PIL import Image


def check_model(model_path: Path) -> dict:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"BLIP config.json missing: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "blip":
        raise ValueError(f"Expected a BLIP checkpoint, got {config.get('model_type')!r}")
    if not any((model_path / name).is_file()
               for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"BLIP weights missing in {model_path}")
    return {
        "model_path": str(model_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }


def read_manifest(run_dir: Path) -> list[dict]:
    path = run_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        sample_id = row["id"]
        if (not isinstance(sample_id, str) or not sample_id or
                Path(sample_id).name != sample_id or sample_id in {".", ".."} or
                sample_id in seen):
            raise ValueError(f"Invalid or duplicate sample ID: {sample_id!r}")
        seen.add(sample_id)
        source = Path(row["source_original"])
        if not source.is_file():
            raise FileNotFoundError(f"Original Cityscapes image missing: {source}")
    return rows


def generate_captions(args: argparse.Namespace, *, processor=None, model=None) -> None:
    if args.max_new_tokens < 1 or args.num_beams < 1 or args.cpu_threads < 1:
        raise ValueError("Generation limits and CPU threads must be positive")
    model_info = check_model(args.model)
    manifest = read_manifest(args.run_dir)
    worker_index = getattr(args, "worker_index", None)
    worker_count = getattr(args, "worker_count", 1)
    if worker_index is not None:
        if worker_count < 1 or not 0 <= worker_index < worker_count:
            raise ValueError("Invalid BLIP worker index/count")
        manifest = manifest[worker_index::worker_count]
        if not manifest:
            raise ValueError("BLIP worker has no samples")
    suffix = "" if worker_index is None else f"_worker_{worker_index}"
    caption_path = args.run_dir / f"captions{suffix}.jsonl"
    sender_dir = args.run_dir / "sender_text"
    profile_path = args.run_dir / f"blip_caption_profile{suffix}.json"
    if caption_path.exists() or profile_path.exists() or (worker_index is None and sender_dir.exists()):
        raise FileExistsError("BLIP caption outputs already exist; refusing to overwrite")
    if (processor is None) != (model is None):
        raise ValueError("processor and model must both be supplied for an injected run")

    load_started = time.monotonic()
    if processor is None:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("BLIP_DEVICE=cuda requested but CUDA is unavailable")
        if args.device == "cpu":
            torch.set_num_threads(args.cpu_threads)
        processor = BlipProcessor.from_pretrained(str(args.model), local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(
            str(args.model), local_files_only=True
        ).to(args.device).eval()
        inference_context = torch.inference_mode
    else:
        inference_context = nullcontext
    load_seconds = round(time.monotonic() - load_started, 3)

    records = []
    sample_times = []
    for index, row in enumerate(manifest, start=1):
        started = time.monotonic()
        with Image.open(row["source_original"]) as image:
            inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(args.device)
        with inference_context():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams, do_sample=False,
            )
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        if not caption:
            raise ValueError(f"BLIP returned an empty caption for {row['id']}")
        elapsed = round(time.monotonic() - started, 3)
        sample_times.append({"id": row["id"], "seconds": elapsed})
        records.append({
            "id": row["id"],
            "caption": caption,
            "source_original": row["source_original"],
            "model": model_info["model_path"],
            "backend": "transformers_blip_local",
            "input": "original_image_only",
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "do_sample": False,
        })
        print(f"BLIP caption progress: {index}/{len(manifest)}; {elapsed:.1f}s", flush=True)

    if worker_index is None:
        sender_dir.mkdir()
        for row in records:
            (sender_dir / f"{row['id']}.txt").write_text(
                row["caption"] + "\n", encoding="utf-8"
            )
    with caption_path.open("x", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    profile_path.write_text(json.dumps({
        **model_info,
        "sample_count": len(records),
        "device": args.device,
        "cpu_threads": args.cpu_threads if args.device == "cpu" else None,
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": False,
        "input": "source_original_only",
        "model_load_seconds": load_seconds,
        "sample_times": sample_times,
        "worker_index": worker_index,
        "worker_count": worker_count,
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} BLIP captions to {caption_path}", flush=True)


def worker_devices(gpu_devices: str, workers: int, samples: int) -> list[str]:
    devices = [part.strip() for part in gpu_devices.split(",")]
    if (not devices or any(not part.isdecimal() for part in devices) or
            len(devices) != len(set(devices))):
        raise ValueError("GPU devices must be distinct numeric IDs")
    if workers < 1 or workers > min(len(devices), samples):
        raise ValueError("BLIP workers must be between 1 and available GPUs/samples")
    return devices[:workers]


def merge_worker_captions(run_dir: Path, manifest: list[dict],
                          devices: list[str], model_info: dict,
                          args: argparse.Namespace) -> None:
    rows = {}
    profiles = []
    assignment = {row["id"]: index % len(devices)
                  for index, row in enumerate(manifest)}
    for index, device in enumerate(devices):
        path = run_dir / f"captions_worker_{index}.jsonl"
        profile_path = run_dir / f"blip_caption_profile_worker_{index}.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if (profile["worker_index"] != index or profile["worker_count"] != len(devices)
                or profile["model_path"] != model_info["model_path"]
                or profile["config_sha256"] != model_info["config_sha256"]
                or profile["sample_count"] != len(manifest[index::len(devices)])):
            raise ValueError(f"Invalid BLIP worker profile: {profile_path}")
        profiles.append(profile_path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["id"]
            if (sample_id in rows or assignment.get(sample_id) != index or
                    row["model"] != model_info["model_path"] or
                    not row["caption"].strip()):
                raise ValueError(f"Duplicate or mismatched BLIP caption: {sample_id}")
            rows[sample_id] = row
    expected = [row["id"] for row in manifest]
    if set(rows) != set(expected):
        raise ValueError("BLIP workers did not cover the manifest exactly")
    sender_dir = run_dir / "sender_text"
    sender_dir.mkdir()
    for sample_id in expected:
        (sender_dir / f"{sample_id}.txt").write_text(
            rows[sample_id]["caption"] + "\n", encoding="utf-8")
    with (run_dir / "captions.jsonl").open("x", encoding="utf-8") as handle:
        for sample_id in expected:
            handle.write(json.dumps(rows[sample_id], ensure_ascii=False) + "\n")
    (run_dir / "blip_caption_profile.json").write_text(json.dumps({
        **model_info, "sample_count": len(expected), "device": "cuda",
        "gpu_devices": devices, "worker_count": len(devices),
        "worker_profiles": [str(path.resolve()) for path in profiles],
        "max_new_tokens": args.max_new_tokens, "num_beams": args.num_beams,
        "do_sample": False, "input": "source_original_only",
    }, indent=2), encoding="utf-8")
    print(f"Saved {len(expected)} BLIP captions to {run_dir / 'captions.jsonl'}", flush=True)


def generate_parallel(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        raise ValueError("Parallel BLIP captioning requires --device cuda")
    model_info = check_model(args.model)
    manifest = read_manifest(args.run_dir)
    devices = worker_devices(args.gpu_devices, args.workers, len(manifest))
    if any((args.run_dir / name).exists() for name in
           ("captions.jsonl", "sender_text", "blip_caption_profile.json", "blip_worker_logs")):
        raise FileExistsError("BLIP caption outputs already exist; refusing to overwrite")
    log_dir = args.run_dir / "blip_worker_logs"
    log_dir.mkdir()
    processes = []
    handles = []
    log_paths = []
    try:
        for index, device in enumerate(devices):
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--run-dir", str(args.run_dir), "--model", str(args.model),
                       "--device", "cuda", "--max-new-tokens", str(args.max_new_tokens),
                       "--num-beams", str(args.num_beams),
                       "--cpu-threads", str(args.cpu_threads),
                       "--worker-index", str(index), "--worker-count", str(len(devices))]
            log_path = log_dir / f"worker_{index}_gpu_{device}.log"
            log_paths.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            processes.append(subprocess.Popen(command, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT))
            print(f"Started BLIP worker {index}: GPU={device}; log={log_path}",
                  flush=True)
        remaining = set(range(len(processes)))
        while remaining:
            for index in tuple(remaining):
                status = processes[index].poll()
                if status is None:
                    continue
                remaining.remove(index)
                print(f"BLIP worker {index} exited with code {status}", flush=True)
                if status != 0:
                    tail = "\n".join(log_paths[index].read_text(
                        encoding="utf-8", errors="replace").splitlines()[-30:])
                    raise RuntimeError(f"BLIP worker {index} failed ({status}); "
                                       f"log={log_paths[index]}\n{tail}")
            if remaining:
                time.sleep(1)
        merge_worker_captions(args.run_dir, manifest, devices, model_info, args)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpu-devices", default="0")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers > 1 and args.worker_index is None:
        generate_parallel(args)
    else:
        generate_captions(args)


if __name__ == "__main__":
    main()
