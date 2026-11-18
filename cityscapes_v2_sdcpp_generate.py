#!/usr/bin/env python3
"""Generate Cityscapes fusion outputs with sd-server.

Each worker starts one server on a separate physical GPU and keeps it alive
across its assigned images. The manifest/results.jsonl contracts are preserved.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from cityscapes_gt_smoke import (
    GENERATION_PROMPT_VERSION,
    fusion_mask_guidance_applies,
    generation_prompt,
    make_comparison,
    print_generation_progress,
    read_new_generation_count,
    read_jsonl,
    save_jsonl_exclusive,
)
from cityscapes_v2_weight_precision import verify_bf16_weights


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict]]:
    for label in ("server_bin", "diffusion_model", "vae", "llm"):
        path = getattr(args, label)
        if not path.is_file():
            raise FileNotFoundError(f"{label} must be an existing file: {path}")
    if not os.access(args.server_bin, os.X_OK):
        raise PermissionError(f"sd-server is not executable: {args.server_bin}")
    if args.llm.suffix.lower() == ".gguf" and args.llm_vision is None:
        raise ValueError("Qwen-Image-2.1 editing with a GGUF LLM needs --llm-vision (mmproj)")
    if args.llm_vision is not None and not args.llm_vision.is_file():
        raise FileNotFoundError(f"llm_vision must be an existing file: {args.llm_vision}")
    args.weight_dtype_profile = (verify_bf16_weights(args.diffusion_model, args.llm, args.vae)
                                 if args.precision_profile == "bf16" else None)
    if not str(args.gpu_device).isdecimal():
        raise ValueError("--gpu-device must be a physical numeric CUDA device ID")
    if not args.backend_spec or not args.backend_spec.strip():
        raise ValueError("--backend-spec must not be empty")
    if args.offload_to_cpu and args.auto_fit == "on":
        raise ValueError("--offload-to-cpu requires --auto-fit off")
    worker_index = getattr(args, "worker_index", None)
    if args.output_dir.exists() and worker_index is None:
        raise FileExistsError(f"Will not overwrite generated outputs: {args.output_dir}")
    if worker_index is not None and not args.output_dir.is_dir():
        raise FileNotFoundError(f"Worker output directory missing: {args.output_dir}")
    if not 1 <= args.port <= 65535:
        raise ValueError("Invalid port")
    if (args.startup_timeout <= 0 or args.job_timeout <= 0 or
            getattr(args, "capabilities_timeout", 30) <= 0):
        raise ValueError("Timeouts must be positive")
    fusion_mask_guidance_applies(None, args.fusion_mask_prompt_policy)

    args.seed = 42 if args.seed is None else args.seed
    args.steps = 40 if args.steps is None else args.steps
    args.cfg_scale = 1.0 if args.cfg_scale is None else args.cfg_scale
    if args.receiver_jsonl is None:
        raise ValueError("--receiver-jsonl is required for fusion generation")
    manifest = read_jsonl(args.run_dir / "manifest.jsonl")
    receiver_rows = read_jsonl(args.receiver_jsonl)
    if args.steps < 1 or args.cfg_scale is None or args.cfg_scale <= 0:
        raise ValueError("Invalid steps or CFG scale")
    if not manifest or len({row["id"] for row in manifest}) != len(manifest):
        raise ValueError("Manifest must contain distinct sample IDs")
    if len({row["id"] for row in receiver_rows}) != len(receiver_rows):
        raise ValueError("Receiver records must have distinct sample IDs")
    receiver = {row["id"]: row for row in receiver_rows}
    if set(receiver) != {row["id"] for row in manifest}:
        raise ValueError("Receiver IDs do not match prepared sample IDs")
    for row in manifest:
        width, height = row["model_size"]
        if width <= 0 or height <= 0 or width % 32 or height % 32:
            raise ValueError(f"sd.cpp requires dimensions divisible by 32: {row['id']}")
        if not receiver[row["id"]].get("condition_map"):
            raise ValueError(f"Receiver condition map missing for {row['id']}")
        condition_path = Path(receiver[row["id"]]["condition_map"])
        with Image.open(condition_path) as condition:
            if condition.size != (width, height):
                raise ValueError(f"Receiver map size mismatch for {row['id']}")
        if "caption" not in receiver[row["id"]]:
            raise ValueError(f"Receiver caption missing for {row['id']}")
    return manifest, receiver


def check_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))


def http_json(url: str, payload: dict | None = None, timeout: float = 10) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.load(response)
            if not isinstance(data, dict):
                raise RuntimeError(f"Unexpected JSON response from {url}")
            if body is not None and response.status != 202:
                raise RuntimeError(f"Expected HTTP 202 from {url}; got {response.status}: {data}")
            return data
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"sd-server HTTP {exc.code} from {url}: {detail}") from exc


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float,
                    capabilities_timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "no capabilities response yet"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sd-server exited during startup (code {process.returncode})")
        try:
            request_timeout = min(capabilities_timeout,
                                  max(0.1, deadline - time.monotonic()))
            capabilities = http_json(base_url + "/sdcpp/v1/capabilities",
                                     timeout=request_timeout)
        except RuntimeError as exc:
            if "HTTP 404" in str(exc):
                raise RuntimeError("This sd-server binary lacks /sdcpp/v1/capabilities; use a current build") from exc
            last_error = str(exc)
            time.sleep(2)
            continue
        except (OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2)
            continue
        features = capabilities.get("features_by_mode", {}).get("img_gen", {})
        if not capabilities.get("model") or "img_gen" not in capabilities.get("supported_modes", []):
            last_error = "capabilities lacks model or img_gen mode"
            time.sleep(2)
            continue
        if not features.get("ref_images"):
            raise RuntimeError("Loaded sd-server model does not advertise img_gen.ref_images")
        return capabilities
    raise TimeoutError(f"sd-server did not load within {timeout}s; last check: {last_error}")


def submit_and_wait(base_url: str, payload: dict, timeout: float) -> tuple[Image.Image, dict]:
    job = http_json(base_url + "/sdcpp/v1/img_gen", payload, timeout=60)
    job_id = job.get("id")
    poll_path = job.get("poll_url")
    if not isinstance(job_id, str) or not isinstance(poll_path, str) or \
            not poll_path.startswith("/sdcpp/v1/jobs/") or "?" in poll_path:
        raise RuntimeError(f"Malformed sd-server job response: {job}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = http_json(base_url + poll_path, timeout=30)
        if state.get("id") != job_id:
            raise RuntimeError("sd-server returned the wrong job ID")
        status = state.get("status")
        if status == "completed":
            images = (state.get("result") or {}).get("images", [])
            if len(images) != 1 or "b64_json" not in images[0]:
                raise RuntimeError(f"sd-server returned {len(images)} output images; expected 1")
            raw = base64.b64decode(images[0]["b64_json"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                return image.convert("RGB"), state
        if status in {"failed", "cancelled"}:
            raise RuntimeError(f"sd-server job {job_id} {status}: {state.get('error')}")
        if status not in {"queued", "generating"}:
            raise RuntimeError(f"Unknown sd-server job status: {status}")
        time.sleep(1)
    try:
        http_json(base_url + poll_path + "/cancel", {}, timeout=10)
    except (OSError, RuntimeError):
        pass
    raise TimeoutError(f"sd-server job {job_id} exceeded {timeout}s")


def request_payload(prompt: str, image_path: Path | None, width: int, height: int,
                    seed: int, steps: int, cfg_scale: float) -> dict:
    return {
        "prompt": prompt,
        "ref_images": [base64.b64encode(image_path.read_bytes()).decode("ascii")]
        if image_path is not None else [],
        "width": width,
        "height": height,
        "seed": seed,
        "batch_count": 1,
        "sample_params": {
            "sample_method": "euler",
            "sample_steps": steps,
            "guidance": {"txt_cfg": cfg_scale, "img_cfg": 1.0},
        },
        "output_format": "png",
        "preview": "none",
    }


def file_info(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def server_job_duration_seconds(state: dict) -> float | None:
    """sd-server job timestamps in the tested build are Unix seconds."""
    started, completed = state.get("started"), state.get("completed")
    if (not isinstance(started, (int, float)) or isinstance(started, bool) or
            not isinstance(completed, (int, float)) or isinstance(completed, bool) or
            completed < started):
        return None
    return round(completed - started, 3)


def worker_devices(gpu_devices: str, workers: int, samples: int) -> list[str]:
    devices = [part.strip() for part in gpu_devices.split(",")]
    if (not devices or any(not part.isdecimal() for part in devices) or
            len(devices) != len(set(devices))):
        raise ValueError("GPU devices must be distinct numeric IDs")
    if workers < 1 or workers > len(devices):
        raise ValueError("Workers must be between 1 and the number of GPU devices")
    return devices[:min(workers, samples)]


def worker_command(args: argparse.Namespace, index: int, count: int,
                   device: str, port: int) -> list[str]:
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--run-dir", str(args.run_dir), "--output-dir", str(args.output_dir),
               "--server-bin", str(args.server_bin),
               "--diffusion-model", str(args.diffusion_model),
               "--vae", str(args.vae), "--llm", str(args.llm),
               "--precision-profile", args.precision_profile,
               "--gpu-device", device, "--port", str(port),
               "--backend-spec", args.backend_spec, "--auto-fit", args.auto_fit,
               "--seed", str(args.seed), "--steps", str(args.steps),
               "--cfg-scale", str(args.cfg_scale),
               "--capabilities-timeout", str(getattr(args, "capabilities_timeout", 30)),
               "--startup-timeout", str(args.startup_timeout),
               "--job-timeout", str(args.job_timeout),
               "--fusion-mask-prompt-policy", args.fusion_mask_prompt_policy,
               "--worker-index", str(index), "--worker-count", str(count)]
    if args.receiver_jsonl:
        command += ["--receiver-jsonl", str(args.receiver_jsonl)]
    if args.llm_vision:
        command += ["--llm-vision", str(args.llm_vision)]
    if args.max_vram:
        command += ["--max-vram", args.max_vram]
    if args.diffusion_fa:
        command.append("--diffusion-fa")
    if args.verbose:
        command.append("--verbose")
    if args.offload_to_cpu:
        command.append("--offload-to-cpu")
    return command


def merge_worker_results(output_dir: Path, manifest: list[dict],
                         worker_count: int) -> None:
    branches = ("fusion",)
    expected = [(record["id"], branch) for record in manifest for branch in branches]
    expected_set = set(expected)
    sizes = {record["id"]: tuple(record["model_size"]) for record in manifest}
    assignment = {record["id"]: index % worker_count
                  for index, record in enumerate(manifest)}
    rows: dict[tuple[str, str], dict] = {}
    for worker_index in range(worker_count):
        path = output_dir / f"worker_{worker_index}_results.jsonl"
        for row in read_jsonl(path):
            key = (row.get("id"), row.get("branch"))
            if (key in rows or key not in expected_set or
                    assignment.get(row.get("id")) != worker_index or
                    row.get("worker_index") != worker_index):
                raise ValueError(f"Invalid worker output: {path}: {key}")
            image_path = output_dir / row["id"] / f"{row['branch']}.png"
            if Path(row.get("path", "")).resolve() != image_path.resolve():
                raise ValueError(f"Worker image path mismatch: {key}")
            with Image.open(image_path) as image:
                if image.mode != "RGB" or image.size != sizes[row["id"]]:
                    raise ValueError(f"Worker image mode or size mismatch: {key}")
            rows[key] = row
    if len(expected) != len(rows):
        raise ValueError("Worker results do not cover every expected sample and branch")
    for record in manifest:
        if not (output_dir / record["id"] / "comparison.png").is_file():
            raise FileNotFoundError(f"Missing comparison for {record['id']}")
    save_jsonl_exclusive(output_dir / "results.jsonl", [rows[key] for key in expected])


def generate_parallel(args: argparse.Namespace) -> None:
    manifest, _ = validate_inputs(args)
    devices = worker_devices(args.gpu_devices, args.workers, len(manifest))
    if len(devices) == 1:
        args.gpu_device = devices[0]
        args.workers = 1
        generate(args)
        return
    if args.port + len(devices) - 1 > 65535:
        raise ValueError("Worker port range exceeds 65535")
    for index in range(len(devices)):
        check_port_available(args.port + index)
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "worker_logs"
    logs.mkdir()
    processes = []
    handles = []
    log_paths = []
    offsets = [0] * len(devices)
    pending = [b""] * len(devices)
    total = len(manifest)
    completed = 0
    last_reported = -1
    last_report_time = 0.0
    progress_started = time.monotonic()
    try:
        for index, device in enumerate(devices):
            command = worker_command(args, index, len(devices), device, args.port + index)
            log_path = logs / f"worker_{index}_gpu_{device}.log"
            log_paths.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(command, stdout=handle,
                                              stderr=subprocess.STDOUT))
            print(f"Started sd-server worker {index}: GPU={device}, port={args.port + index}, "
                  f"log={log_path}", flush=True)
        remaining = set(range(len(devices)))
        failure = None
        while remaining:
            for index, log_path in enumerate(log_paths):
                count, offsets[index], pending[index] = read_new_generation_count(
                    log_path, offsets[index], pending[index])
                completed += count
            if completed > total:
                raise RuntimeError("Worker progress exceeds expected image count")
            now = time.monotonic()
            if completed != last_reported or now - last_report_time >= 30:
                print_generation_progress(completed, total, progress_started)
                last_reported, last_report_time = completed, now
            for index in tuple(remaining):
                status = processes[index].poll()
                if status is not None:
                    remaining.remove(index)
                    print(f"sd-server worker {index} exited with code {status}", flush=True)
                    if status != 0:
                        failure = (index, status)
                        break
            if failure:
                break
            if remaining:
                time.sleep(1)
        if failure:
            index, status = failure
            tail = "\n".join(log_paths[index].read_text(
                encoding="utf-8", errors="replace").splitlines()[-30:])
            raise RuntimeError(f"sd-server worker {index} failed ({status}); "
                               f"log={log_paths[index]}\n{tail}")
        merge_worker_results(args.output_dir, manifest, len(devices))
        worker_profiles = [args.output_dir / f"worker_{index}_sdcpp_profile.json"
                           for index in range(len(devices))]
        profile = json.loads(worker_profiles[0].read_text(encoding="utf-8"))
        profile.update(worker_count=len(devices), gpu_devices=devices,
                       worker_profiles=[str(path.resolve()) for path in worker_profiles],
                       worker_commands=[json.loads(path.read_text(encoding="utf-8"))["command"]
                                        for path in worker_profiles],
                       command=None, server_startup_seconds=None,
                       first_request_wall_seconds=None, sample_count=len(manifest))
        (args.output_dir / "sdcpp_profile.json").write_text(
            json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        print_generation_progress(total, total, progress_started)
        print(f"Saved {total} outputs in {args.output_dir}", flush=True)
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


def generate(args: argparse.Namespace) -> None:
    if getattr(args, "workers", 1) > 1 and getattr(args, "worker_index", None) is None:
        generate_parallel(args)
        return
    manifest, receiver = validate_inputs(args)
    worker_index = getattr(args, "worker_index", None)
    if worker_index is not None:
        worker_count = args.worker_count
        if worker_count < 2 or not 0 <= worker_index < worker_count:
            raise ValueError("Invalid worker index/count")
        manifest = [record for index, record in enumerate(manifest)
                    if index % worker_count == worker_index]
        if not manifest:
            raise ValueError("Worker has no assigned samples")
    check_port_available(args.port)
    if worker_index is None:
        args.output_dir.mkdir(parents=True)
    base_url = f"http://127.0.0.1:{args.port}"
    command = [str(args.server_bin), "--diffusion-model", str(args.diffusion_model),
               "--vae", str(args.vae), "--llm", str(args.llm)]
    if args.llm_vision:
        command += ["--llm_vision", str(args.llm_vision)]
    # CUDA_VISIBLE_DEVICES exposes exactly one physical GPU as cuda0 to sd.cpp.
    command += ["--backend", args.backend_spec, "--auto-fit", args.auto_fit,
                "--listen-ip", "127.0.0.1", "--listen-port", str(args.port)]
    if args.max_vram:
        command += ["--max-vram", args.max_vram]
    if args.diffusion_fa:
        command.append("--diffusion-fa")
    if args.verbose:
        command.append("--verbose")
    if args.offload_to_cpu:
        command.append("--offload-to-cpu")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    log_path = args.output_dir / ("sdcpp_server.log" if worker_index is None else
                                  f"worker_{worker_index}_sdcpp_server.log")
    profile_path = args.output_dir / ("sdcpp_profile.json" if worker_index is None else
                                      f"worker_{worker_index}_sdcpp_profile.json")
    profile = {
        "backend": "sdcpp_server", "command": command,
        "precision_profile": args.precision_profile,
        "weight_dtype_profile": args.weight_dtype_profile,
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "backend_spec": args.backend_spec,
        "auto_fit": args.auto_fit,
        "max_vram": args.max_vram,
        "diffusion_fa": args.diffusion_fa,
        "verbose": args.verbose,
        "offload_to_cpu": args.offload_to_cpu,
        "server_binary": file_info(args.server_bin),
        "diffusion_model": file_info(args.diffusion_model),
        "vae": file_info(args.vae), "llm": file_info(args.llm),
        "llm_vision": file_info(args.llm_vision) if args.llm_vision else None,
        "sample_count": len(manifest), "worker_index": worker_index,
        "seed": args.seed,
        "steps": args.steps, "cfg_scale": args.cfg_scale, "img_cfg_scale": 1.0,
        "sampler": "euler", "api": "sdcpp/v1/img_gen",
    }
    with log_path.open("w", encoding="utf-8") as server_log:
        started = time.monotonic()
        process = subprocess.Popen(command, stdout=server_log,
                                   stderr=subprocess.STDOUT, env=env)
        try:
            capabilities = wait_for_server(base_url, process, args.startup_timeout,
                                           getattr(args, "capabilities_timeout", 30))
            startup_seconds = round(time.monotonic() - started, 3)
            profile["server_startup_seconds"] = startup_seconds
            # sd-server reports ready before loading the model tensors. Their load
            # time is included in the first request and is not isolated here.
            profile["model_load_seconds"] = None
            profile["server_model"] = capabilities.get("model")
            profile["server_capabilities"] = capabilities
            profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
            print(f"sd-server ready in {startup_seconds}s; GPU={args.gpu_device}; "
                  "model tensors may load on first request", flush=True)
            rows = []
            progress_started = time.monotonic()
            output_count = len(manifest)
            print_generation_progress(0, output_count, progress_started)
            for record in manifest:
                sample_id = record["id"]
                sample_dir = args.output_dir / sample_id
                sample_dir.mkdir()
                width, height = record["model_size"]
                received = receiver[sample_id]
                condition_path = Path(received["condition_map"])
                branch_paths = {}
                for branch, caption_text in (("fusion", received["caption"]),):
                    prompt = generation_prompt(caption_text, args.fusion_mask_prompt_policy)
                    prompt_version = GENERATION_PROMPT_VERSION
                    payload = request_payload(prompt, condition_path, width, height,
                                              args.seed, args.steps, args.cfg_scale)
                    generation_started = time.monotonic()
                    image, state = submit_and_wait(base_url, payload, args.job_timeout)
                    generation_seconds = round(time.monotonic() - generation_started, 3)
                    if "first_request_wall_seconds" not in profile:
                        profile["first_request_wall_seconds"] = generation_seconds
                        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
                    if image.size != (width, height):
                        raise RuntimeError(f"Unexpected output size for {sample_id}/{branch}: {image.size}")
                    path = sample_dir / f"{branch}.png"
                    image.save(path)
                    branch_paths[branch] = path
                    rows.append({
                        "id": sample_id, "branch": branch, "path": str(path.resolve()),
                        "prompt": prompt,
                        "generation_prompt_version": prompt_version,
                        "fusion_mask_prompt_policy": args.fusion_mask_prompt_policy,
                        "caption_mask_count": caption_text.count("[MASK]") if caption_text else 0,
                        "mask_guidance_applied": fusion_mask_guidance_applies(
                            caption_text, args.fusion_mask_prompt_policy),
                        "seed": args.seed, "steps": args.steps,
                        "generation_seconds": generation_seconds,
                        "server_generation_seconds": server_job_duration_seconds(state),
                        "model": str(args.diffusion_model), "backend": "sdcpp_server",
                        "cfg_scale": args.cfg_scale, "img_cfg_scale": 1.0,
                        "sampler": "euler",
                        "placement": ("cpu_offload" if args.offload_to_cpu else
                                      "auto_fit" if args.auto_fit == "on" else "resident"),
                        "server_startup_seconds": startup_seconds,
                        "model_load_seconds": None,
                        "worker_index": worker_index,
                        "cuda_visible_devices": str(args.gpu_device),
                        "condition_map": str(condition_path) if condition_path else None,
                        "caption_utf8_bits": len(received["caption"].encode("utf-8")) * 8,
                        "receiver_jsonl": str(args.receiver_jsonl) if args.receiver_jsonl else None,
                        "server_job_id": state["id"],
                    })
                    print(f"Generated {sample_id}/{branch}: {path}", flush=True)
                    print_generation_progress(len(rows), output_count, progress_started)
                make_comparison({**record, "condition_map": str(condition_path)},
                                branch_paths, sample_dir / "comparison.png")
            result_path = args.output_dir / ("results.jsonl" if worker_index is None else
                                             f"worker_{worker_index}_results.jsonl")
            save_jsonl_exclusive(result_path, rows)
            print(f"Saved {len(rows)} outputs in {args.output_dir}", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--receiver-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-bin", type=Path, required=True)
    parser.add_argument("--diffusion-model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--llm-vision", type=Path)
    parser.add_argument("--precision-profile", choices=["bf16", "mixed"], default="bf16")
    parser.add_argument("--gpu-device", required=True)
    parser.add_argument("--gpu-devices", default="",
                        help="Physical GPU IDs for parallel servers, e.g. 0,1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--backend-spec", default="cuda0")
    parser.add_argument("--auto-fit", choices=["on", "off"], default="off")
    parser.add_argument("--max-vram", default="")
    parser.add_argument("--diffusion-fa", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--capabilities-timeout", type=float, default=30,
                        help="Timeout in seconds for each capabilities health request")
    parser.add_argument("--job-timeout", type=float, default=1200)
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--fusion-mask-prompt-policy", choices=["conditional", "off"],
                        default="conditional")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.worker_index is not None:
        # The parent can stop a failed parallel run without orphaning sd-server.
        signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    generate(args)


if __name__ == "__main__":
    main()
