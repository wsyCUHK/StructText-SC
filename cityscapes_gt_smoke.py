#!/usr/bin/env python3
"""Cityscapes input preparation, captioning and image generation stages.

The caption command is an OpenAI-compatible client for the Docker/SGLang model
service. Channel transmission is handled separately by the workflow.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_LLM_API = "http://127.0.0.1:30000/v1/chat/completions"

# The 19 evaluated classes in the official Cityscapes labels.py. Other labels
# remain in the original GT maps but are ignored in the standard class mIoU.
CLASS_LEGEND = (
    "road=purple; sidewalk=pink; building=gray; wall=blue-gray; "
    "fence=pale-red; pole=light-gray; traffic light=orange; "
    "traffic sign=yellow; vegetation=green; terrain=light-green; "
    "sky=blue; person=red; rider=bright-red; car=dark-blue; "
    "truck=navy; bus=teal; train=dark-teal; motorcycle=blue; bicycle=maroon"
)

CAPTION_SYSTEM = (
    "You write concise, factual captions for reconstructing driving scenes. "
    "The supplied image is the original camera photograph. Describe the "
    "scene as a single English paragraph of at most 120 words. Prioritize "
    "visible objects, counts, spatial relationships, distinctive appearance, "
    "lighting, and weather. State only details supported by the photograph. "
    "Do not invent objects or infer hidden content."
)
CAPTION_USER_INSTRUCTION = "Describe the supplied photograph for faithful image reconstruction."
CAPTION_MAP_SYSTEM = (
    "You write concise, factual captions for reconstructing driving scenes. "
    "Image 1 is the original camera image. Image 2 is its Cityscapes semantic "
    "segmentation visualization, available only to the sender. Describe the "
    "scene as a single English paragraph of at most 120 words. Prioritize "
    "appearance, lighting, weather, distinctive object attributes, counts, "
    "and spatial relationships that the layout alone does not express. Do not "
    "invent objects. Do not mention the segmentation image or its colors."
)
CAPTION_MAP_USER_INSTRUCTION = "Describe Image 1 using Image 2 only as layout support."
CAPTION_INPUT_MODES = ("source_only", "source_and_map")

GENERATION_PROMPT_VERSION = "segmentation_to_photo_v3_conditional_mask"
GENERATION_PREFIX = (
    "Transform the supplied Cityscapes semantic segmentation map into one "
    "photorealistic color photograph from a front-facing urban driving camera. "
    "The supplied image is a class-label diagram, not a photograph: its flat "
    "colors encode object categories and must not become the output's colors or "
    "visual style. Use the boundaries and shapes as the scene layout. Render "
    "only the object classes actually present, with plausible real-world "
    "materials, textures, lighting, shadows, and "
    "perspective. Preserve the visible object counts, positions, relative sizes, "
    "and road geometry. Do not output a colored segmentation map, flat class "
    "regions, labels, annotations, or a side-by-side comparison. The result "
    "must be a single natural dashboard-camera photograph in the same 2:1 framing."
)
FUSION_MISSING_TEXT_GUIDANCE = (
    "The following scene description was recovered after packet loss and contains "
    "the literal [MASK] marker for missing words. [MASK] is not an object in "
    "the scene. Do not guess missing words or treat uncertain fragments as "
    "facts. Use the supplied segmentation map for visible object layout, "
    "counts, and boundaries. Use intact text for appearance, lighting, and "
    "details not encoded by the map. Do not add objects solely to fill gaps."
)


def cityscapes_pairs(dataset_root: Path, split: str, limit: int,
                     mask_source: str = "gt") -> list[dict]:
    if mask_source not in {"gt", "drn"}:
        raise ValueError("mask_source must be gt or drn")
    if split not in ({"train", "val"} if mask_source == "gt" else
                     {"train", "val", "test"}):
        raise ValueError("Use train/val for GT; Cityscapes test labels are withheld.")
    if limit < 1:
        raise ValueError("limit must be positive")
    image_root = dataset_root / "leftImg8bit" / split
    gt_root = dataset_root / "gtFine" / split
    if not image_root.is_dir() or (split != "test" and not gt_root.is_dir()):
        raise FileNotFoundError(f"Missing Cityscapes directories: {image_root}, {gt_root}")

    by_city: dict[str, list[dict]] = {}
    for image_path in sorted(image_root.glob("*/*_leftImg8bit.png")):
        city = image_path.parent.name
        sample_id = image_path.name.removesuffix("_leftImg8bit.png")
        gt_prefix = gt_root / city / f"{sample_id}_gtFine"
        label_path = Path(f"{gt_prefix}_labelIds.png")
        color_path = Path(f"{gt_prefix}_color.png")
        if split != "test" and (not label_path.is_file() or not color_path.is_file()):
            raise FileNotFoundError(f"Missing labelIds/color GT for {image_path}")
        pair = {"id": sample_id, "city": city,
                "source_original": str(image_path.resolve())}
        if split != "test":
            pair.update(label_original=str(label_path.resolve()),
                        color_original=str(color_path.resolve()))
        by_city.setdefault(city, []).append(pair)
    if not by_city:
        raise ValueError(f"No Cityscapes images in {image_root}")

    # Round-robin across cities makes the tiny check less dependent on one city.
    selected: list[dict] = []
    cities = sorted(by_city)
    depth = 0
    while len(selected) < limit:
        added = False
        for city in cities:
            if depth < len(by_city[city]):
                selected.append(by_city[city][depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return selected


def save_jsonl_exclusive(path: Path, records: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def prepare(args: argparse.Namespace) -> None:
    mask_source = getattr(args, "mask_source", "gt")
    pairs = cityscapes_pairs(args.dataset_root, args.split, args.limit, mask_source)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    if mask_source == "drn":
        (args.run_dir / "sender_drn_train_ids").mkdir()
    records = []
    for pair in pairs:
        sample_dir = args.run_dir / "samples" / pair["id"]
        sample_dir.mkdir(parents=True)
        with Image.open(pair["source_original"]) as source:
            original_size = list(source.size)
            target_size = (args.width, args.height)
            if getattr(args, "require_native_size", False) and source.size != target_size:
                raise ValueError(
                    f"Original image size for {pair['id']} is {source.size}, "
                    f"but workflow requires {target_size}; refusing to resize")
            source_rgb = source.convert("RGB")
            if source_rgb.size != target_size:
                source_rgb = source_rgb.resize(target_size, Image.Resampling.LANCZOS)
            source_path = sample_dir / "source.png"
            map_path = sample_dir / "condition_map.png"
            source_rgb.save(source_path)
        record = {**pair, "source": str(source_path.resolve()),
                  "condition_map": str(map_path.resolve()),
                  "original_size": original_size,
                  "model_size": [args.width, args.height],
                  "split": args.split,
                  "mask_source": ("Cityscapes gtFine ground truth" if mask_source == "gt"
                                  else "DRN-D-105 predicted")}
        if args.split != "test":
            with (Image.open(pair["label_original"]) as labels,
                  Image.open(pair["color_original"]) as color):
                if tuple(original_size) != labels.size or labels.size != color.size:
                    raise ValueError(f"Source/GT dimensions differ for {pair['id']}")
                if labels.mode != "L":
                    raise ValueError(f"Expected single-channel label IDs for {pair['id']}, got {labels.mode}")
                labels_path = sample_dir / "label_ids.png"
                label_image = labels.convert("L")
                if label_image.size != target_size:
                    label_image = label_image.resize(target_size, Image.Resampling.NEAREST)
                label_image.save(labels_path)
                record["label_ids"] = str(labels_path.resolve())
                if mask_source == "gt":
                    color_image = color.convert("RGB")
                    if color_image.size != target_size:
                        color_image = color_image.resize(target_size, Image.Resampling.NEAREST)
                    color_image.save(map_path)
        if mask_source == "drn":
            record["sender_train_ids"] = str(
                (args.run_dir / "sender_drn_train_ids" / f"{pair['id']}.png").resolve())
        records.append(record)
    save_jsonl_exclusive(args.run_dir / "manifest.jsonl", records)
    print(f"Prepared {len(records)} samples in {args.run_dir}")


def image_path_to_api_url(image_path: str, image_format: str) -> str:
    raw = Path(image_path).read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    if image_format == "legacy_base64":
        # Keep the request layout used by the original local Qwen/SGLang path.
        return encoded
    if image_format == "data_url":
        mime_type = mimetypes.guess_type(image_path)[0] or "application/octet-stream"
        return f"data:{mime_type};base64,{encoded}"
    raise ValueError(f"Unsupported Qwen image format: {image_format}")


def flatten_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        for key in ("text", "content", "value"):
            if key in content:
                return flatten_content_to_text(content[key])
        return ""
    if isinstance(content, list):
        parts = [flatten_content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part).strip()
    return "" if content is None else str(content).strip()


def request_openai_compatible(
    llm_api: str,
    api_model_name: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
    extra_body: dict | None = None,
) -> str:
    # Import lazily so `prepare` remains usable without the API client package.
    import requests

    payload: dict[str, object] = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if api_model_name:
        payload["model"] = api_model_name
    if temperature > 0:
        payload["temperature"] = temperature
    if extra_body:
        protected = {"messages", "model", "max_tokens", "temperature", "stream"}
        collisions = protected.intersection(extra_body)
        if collisions:
            raise ValueError(
                "--api-extra-body-json cannot override request fields: "
                + ", ".join(sorted(collisions))
            )
        payload.update(extra_body)

    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(llm_api, json=payload, timeout=timeout_seconds)
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"HTTP {response.status_code} from {llm_api}; "
                    "response body omitted from caption log"
                ) from exc
            response_json = response.json()
            text = flatten_content_to_text(
                response_json["choices"][0]["message"]["content"]
            )
            if not text:
                raise RuntimeError("SGLang returned an empty caption")
            return text
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            time.sleep(min(5 * (attempt + 1), 15))
    raise RuntimeError(
        f"OpenAI-compatible request failed after {max_retries + 1} attempts: "
        f"{last_error}"
    )


def validate_caption_prompt_input(prompt: dict, input_mode: str) -> None:
    """Reject declared input mismatches and legacy two-image wording in one-image runs."""
    if input_mode not in CAPTION_INPUT_MODES:
        raise ValueError(f"Unsupported caption input mode: {input_mode}")
    declared_mode = prompt.get("input_mode", input_mode)
    if declared_mode != input_mode:
        raise ValueError(
            f"Prompt input_mode={declared_mode!r} conflicts with "
            f"caption input mode={input_mode!r}"
        )
    prompt_text = prompt["system_prompt"] + "\n" + prompt["user_instruction"]
    if input_mode == "source_only" and re.search(
        r"\bimage\s*(?:2|two)\b|\bsecond\s+image\b", prompt_text, re.IGNORECASE
    ):
        raise ValueError("source_only prompt must not refer to a second image")


def load_caption_prompt(path: Path | None, input_mode: str = "source_only") -> dict:
    """Load a prompt and validate its declared caption input contract.

    Older three-field custom prompts are accepted with the selected input mode.
    New prompts should include input_mode to make that contract explicit.
    """
    if path is None:
        if input_mode not in CAPTION_INPUT_MODES:
            raise ValueError(f"Unsupported caption input mode: {input_mode}")
        use_map = input_mode == "source_and_map"
        return {"name": f"default_{input_mode}", "input_mode": input_mode,
                "system_prompt": CAPTION_MAP_SYSTEM if use_map else CAPTION_SYSTEM,
                "user_instruction": (CAPTION_MAP_USER_INSTRUCTION if use_map
                                     else CAPTION_USER_INSTRUCTION), "sha256": None}
    raw = path.read_bytes()
    prompt = json.loads(raw.decode("utf-8"))
    required = {"name", "system_prompt", "user_instruction"}
    if (not isinstance(prompt, dict) or not required.issubset(prompt)
            or set(prompt) - required - {"input_mode"}):
        raise ValueError("Prompt JSON requires name, system_prompt, user_instruction; "
                         "input_mode is the only optional field")
    for key in ("name", "system_prompt", "user_instruction"):
        if not isinstance(prompt[key], str) or not prompt[key].strip():
            raise ValueError(f"Prompt {key} must be a non-empty string")
    if len(prompt["system_prompt"]) > 4000 or len(prompt["user_instruction"]) > 2000:
        raise ValueError("Prompt is unexpectedly long")
    validate_caption_prompt_input(prompt, input_mode)
    return {**prompt, "input_mode": input_mode,
            "sha256": hashlib.sha256(raw).hexdigest()}


def caption_messages(record: dict, image_format: str = "legacy_base64",
                     prompt: dict | None = None,
                     model_family: str = "qwen3_vl",
                     input_mode: str = "source_only") -> list[dict]:
    prompt = prompt or load_caption_prompt(None, input_mode)
    validate_caption_prompt_input(prompt, input_mode)
    if model_family not in {"gemma4", "qwen3_vl"}:
        raise ValueError(f"Unsupported caption model family: {model_family}")
    source_image = {"type": "image_url", "image_url": {
        "url": image_path_to_api_url(record["source"], image_format)}}
    if input_mode == "source_only":
        # No condition_map lookup or encoding occurs in this path.
        content = [source_image, {"type": "text", "text": (
            "Original Cityscapes street photograph. " + prompt["user_instruction"])}]
        return [{"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": content}]
    condition_image = {"type": "image_url", "image_url": {
        "url": image_path_to_api_url(record["condition_map"], image_format)}}
    map_instruction = (
        "Image 2: aligned semantic segmentation map. "
        f"Legend: {CLASS_LEGEND}. {prompt['user_instruction']}"
    )
    if model_family == "gemma4":
        # Gemma 4's documented multi-image template places both images before text.
        content = [source_image, condition_image, {"type": "text", "text": (
            "Image 1: original Cityscapes street photograph. " + map_instruction)}]
    elif model_family == "qwen3_vl":
        content = [source_image,
                   {"type": "text", "text": "Image 1: original Cityscapes street photograph."},
                   condition_image, {"type": "text", "text": map_instruction}]
    return [
        {"role": "system", "content": prompt["system_prompt"]},
        {"role": "user", "content": content},
    ]


def format_progress_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}m{seconds}s"


def print_generation_progress(completed: int, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    filled = round(24 * completed / total)
    bar = "#" * filled + "-" * (24 - filled)
    remaining = (elapsed / completed) * (total - completed) if completed else None
    eta = format_progress_time(remaining) if remaining is not None else "--"
    print(f"生图进度 [{bar}] {completed}/{total} "
          f"({100 * completed / total:.1f}%) | 已用 {format_progress_time(elapsed)} "
          f"| 预计剩余 {eta}", flush=True)


def read_new_generation_count(log_path: Path, offset: int,
                              pending: bytes) -> tuple[int, int, bytes]:
    """Count completed PNGs from a worker log without rereading earlier output."""
    with log_path.open("rb") as handle:
        handle.seek(offset)
        chunk = handle.read()
    lines = (pending + chunk).split(b"\n")
    pending = lines.pop()
    completed = sum(line.rsplit(b"\r", 1)[-1].strip().startswith(b"Generated ")
                    for line in lines)
    return completed, offset + len(chunk), pending


def caption(args: argparse.Namespace) -> None:
    records = read_jsonl(args.run_dir / "manifest.jsonl")
    input_mode = getattr(args, "caption_input_mode", "source_only")
    prompt = load_caption_prompt(getattr(args, "prompt_file", None), input_mode)
    model_family = getattr(args, "i2t_model_family", "qwen3_vl")
    if model_family == "gemma4" and args.qwen_image_format != "data_url":
        raise ValueError("Gemma 4 requires data_url image format")
    output_path = args.run_dir / "captions.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    try:
        extra_body = json.loads(args.api_extra_body_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--api-extra-body-json is not valid JSON: {exc}") from exc
    if not isinstance(extra_body, dict):
        raise ValueError("--api-extra-body-json must decode to a JSON object")

    outputs = []
    batch_started = time.monotonic()
    total = len(records)
    for record in records:
        started = time.monotonic()
        messages = caption_messages(record, args.qwen_image_format, prompt,
                                    model_family, input_mode)
        text = request_openai_compatible(
            llm_api=args.llm_api,
            api_model_name=args.api_model_name,
            messages=messages,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_seconds=args.request_timeout,
            max_retries=args.max_retries,
            extra_body=extra_body,
        )
        outputs.append(
            {
                "id": record["id"],
                "caption": text,
                "utf8_bits": len(text.encode("utf-8")) * 8,
                "caption_seconds": round(time.monotonic() - started, 3),
                "model": args.api_model_name,
                "model_family": model_family,
                "caption_input_mode": input_mode,
                "backend": "sglang_openai_compatible_api",
                "llm_api": args.llm_api,
                "qwen_image_format": args.qwen_image_format,
                "system_prompt": prompt["system_prompt"],
                "user_prompt": messages[1]["content"][-1]["text"],
                "prompt_name": prompt["name"],
                "prompt_sha256": prompt["sha256"],
            }
        )
        completed = len(outputs)
        elapsed = time.monotonic() - batch_started
        remaining = (elapsed / completed) * (total - completed)
        print(
            f"\rProcessing progress: {completed}/{total} "
            f"({100.0 * completed / total:.1f}%) | "
            f"Elapsed: {format_progress_time(elapsed)} | "
            f"Estimated remaining: {format_progress_time(remaining)}",
            end="",
            flush=True,
        )
    print()
    save_jsonl_exclusive(output_path, outputs)
    text_dir = args.run_dir / "sender_text"
    text_dir.mkdir(exist_ok=False)
    for row in outputs:
        (text_dir / f"{row['id']}.txt").write_text(row["caption"] + "\n", encoding="utf-8")
    print(f"Saved captions: {output_path}")


def fusion_mask_guidance_applies(caption_text: str | None, policy: str) -> bool:
    if policy not in {"conditional", "off"}:
        raise ValueError(f"Unknown fusion mask prompt policy: {policy}")
    return policy == "conditional" and caption_text is not None and "[MASK]" in caption_text


def generation_prompt(caption_text: str | None,
                      fusion_mask_prompt_policy: str = "conditional") -> str:
    apply_guidance = fusion_mask_guidance_applies(caption_text, fusion_mask_prompt_policy)
    if not caption_text or not caption_text.strip():
        return GENERATION_PREFIX
    guidance = f"\n{FUSION_MISSING_TEXT_GUIDANCE}" if apply_guidance else ""
    return (f"{GENERATION_PREFIX}{guidance}\n"
            f"Additional factual scene description: {caption_text}")


def make_comparison(record: dict, branch_paths: dict[str, Path], output_path: Path) -> None:
    width, height = record["model_size"]
    names = ["source", "condition_map", "fusion"]
    paths = {
        "source": Path(record["source"]),
        "condition_map": Path(record["condition_map"]),
        **branch_paths,
    }
    canvas = Image.new("RGB", (3 * width, height + 25), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        x = index * width
        y = 0
        with Image.open(paths[name]) as image:
            canvas.paste(image.convert("RGB"), (x, y + 25))
        draw.text((x + 5, y + 5), name, fill="black")
    canvas.save(output_path)


def merge_generation_results(output_root: Path, manifest: list[dict], worker_count: int) -> None:
    """Publish the original results contract only after every worker succeeds."""
    rows_by_key = {}
    for index in range(worker_count):
        for row in read_jsonl(output_root / f"worker_{index}_results.jsonl"):
            key = (row["id"], row["branch"])
            if key in rows_by_key:
                raise ValueError(f"Duplicate generated result: {key}")
            if not Path(row["path"]).is_file():
                raise FileNotFoundError(f"Missing generated image: {row['path']}")
            rows_by_key[key] = row
    expected = [(record["id"], branch) for record in manifest
                for branch in ("fusion",)]
    if len(expected) != len(set(expected)) or set(rows_by_key) != set(expected):
        raise ValueError("Worker results do not match manifest IDs and branches")
    for record in manifest:
        comparison = output_root / record["id"] / "comparison.png"
        if not comparison.is_file():
            raise FileNotFoundError(f"Missing comparison image: {comparison}")
    save_jsonl_exclusive(output_root / "results.jsonl", [rows_by_key[key] for key in expected])


def generate_parallel(args: argparse.Namespace) -> None:
    """Run one independent image pipeline per selected GPU."""
    placement = getattr(args, "placement", None) or (
        "cpu_offload" if args.cpu_offload else "resident")
    manifest = read_jsonl(args.run_dir / "manifest.jsonl")
    receiver_jsonl = getattr(args, "receiver_jsonl", None)
    if receiver_jsonl is None:
        raise ValueError("--receiver-jsonl is required for fusion generation")
    captions_path = receiver_jsonl
    caption_ids = [row["id"] for row in read_jsonl(captions_path)]
    manifest_ids = [row["id"] for row in manifest]
    if len(set(manifest_ids)) != len(manifest_ids) or len(set(caption_ids)) != len(caption_ids):
        raise ValueError("Duplicate sample or caption IDs")
    if set(caption_ids) != set(manifest_ids):
        raise ValueError("Caption IDs do not match prepared sample IDs")

    devices = [device.strip() for device in args.gpu_devices.split(",") if device.strip()]
    if len(devices) != len(set(devices)) or any(not device.isdecimal() for device in devices):
        raise ValueError("GPU devices must be a comma-separated list of distinct numeric IDs")
    if args.workers < 2 or args.workers > len(devices):
        raise ValueError("Parallel workers must be between 2 and the number of GPU devices")
    worker_count = min(args.workers, len(manifest))
    if worker_count < 2:
        print("Only one sample; using one image worker", flush=True)
        args.workers = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
        generate(args)
        return
    output_root = getattr(args, "output_dir", None) or args.run_dir / "generated"
    if output_root.exists():
        raise FileExistsError(f"Will not overwrite generated outputs: {output_root}")
    output_root.mkdir()
    worker_logs = output_root / "worker_logs"
    worker_logs.mkdir()

    processes = []
    handles = []
    log_paths = []
    progress_offsets = [0] * worker_count
    progress_pending = [b""] * worker_count
    progress_total = len(manifest)
    progress_completed = 0
    last_reported = -1
    last_report_time = 0.0
    progress_started = time.monotonic()
    try:
        for index in range(worker_count):
            device = devices[index]
            command = [
                sys.executable, "-u", str(Path(__file__).resolve()), "generate",
                "--run-dir", str(args.run_dir), "--model", str(args.model),
                "--output-dir", str(output_root), "--seed", str(args.seed),
                "--steps", str(args.steps), "--worker-index", str(index),
                "--worker-count", str(worker_count),
                "--fusion-mask-prompt-policy",
                getattr(args, "fusion_mask_prompt_policy", "conditional"),
            ]
            if receiver_jsonl:
                command.extend(("--receiver-jsonl", str(receiver_jsonl)))
            command.extend(("--placement", placement))
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = device
            log_path = worker_logs / f"worker_{index}_gpu_{device}.log"
            log_paths.append(log_path)
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(
                command, env=environment, stdout=handle, stderr=subprocess.STDOUT,
            ))
            print(f"Started image worker {index} on physical GPU {device}; log: {log_path}", flush=True)
        failure = None
        remaining = set(range(worker_count))
        while remaining:
            for index, log_path in enumerate(log_paths):
                count, progress_offsets[index], progress_pending[index] = (
                    read_new_generation_count(
                        log_path, progress_offsets[index], progress_pending[index]))
                progress_completed += count
            if progress_completed > progress_total:
                raise RuntimeError("Worker progress exceeds the expected image count")
            now = time.monotonic()
            if (progress_completed != last_reported or
                    now - last_report_time >= 30):
                print_generation_progress(progress_completed, progress_total,
                                          progress_started)
                last_reported = progress_completed
                last_report_time = now
            for index in tuple(remaining):
                status = processes[index].poll()
                if status is None:
                    continue
                remaining.remove(index)
                print(f"Image worker {index} exited with code {status}", flush=True)
                if status != 0:
                    failure = (index, status)
                    break
            if failure:
                for index in remaining:
                    processes[index].terminate()
                for index in remaining:
                    try:
                        processes[index].wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        processes[index].kill()
                        processes[index].wait()
                break
            if remaining:
                time.sleep(1)
        if failure:
            index, status = failure
            log_path = worker_logs / f"worker_{index}_gpu_{devices[index]}.log"
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
            raise RuntimeError(f"Image worker {index} failed (exit {status}); log: {log_path}\n{tail}")
        merge_generation_results(output_root, manifest, worker_count)
        if last_reported != progress_total:
            print_generation_progress(progress_total, progress_total, progress_started)
        print(f"Saved {len(manifest)} outputs in {output_root}", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


def generate(args: argparse.Namespace) -> None:
    if getattr(args, "receiver_jsonl", None) is None:
        raise ValueError("--receiver-jsonl is required for fusion generation")
    if getattr(args, "workers", 1) < 1:
        raise ValueError("Image workers must be positive")
    fusion_mask_prompt_policy = getattr(args, "fusion_mask_prompt_policy", "conditional")
    fusion_mask_guidance_applies(None, fusion_mask_prompt_policy)
    placement = getattr(args, "placement", None) or (
        "cpu_offload" if args.cpu_offload else "resident")
    if placement not in {"cpu_offload", "resident"}:
        raise ValueError(f"Unknown image placement: {placement}")
    if getattr(args, "workers", 1) > 1 and getattr(args, "worker_index", None) is None:
        generate_parallel(args)
        return
    if getattr(args, "worker_index", None) is None and getattr(args, "gpu_devices", ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices.split(",")[0].strip()

    import torch
    import diffusers

    QwenImage21Pipeline = getattr(diffusers, "QwenImage21Pipeline", None)
    if QwenImage21Pipeline is None:
        raise RuntimeError(
            "Qwen-Image-2.1 requires diffusers>=0.41.0 with "
            "QwenImage21Pipeline; installed diffusers is "
            f"{getattr(diffusers, '__version__', '<unknown>')}. The older "
            "QwenImagePipeline used by the previous workflow is not a valid "
            "substitute because it would not provide the Qwen-Image-2.1 "
            "unified image-conditioned path. Install a compatible Diffusers environment."
        )

    manifest = read_jsonl(args.run_dir / "manifest.jsonl")
    worker_index = getattr(args, "worker_index", None)
    if worker_index is not None:
        worker_count = args.worker_count
        if worker_count < 2 or not 0 <= worker_index < worker_count:
            raise ValueError("Invalid internal image worker partition")
        manifest = [record for index, record in enumerate(manifest)
                    if index % worker_count == worker_index]
        if not manifest:
            raise ValueError("Image worker has no samples")
    receiver_jsonl = getattr(args, "receiver_jsonl", None)
    if receiver_jsonl is None:
        raise ValueError("--receiver-jsonl is required for fusion generation")
    captions = {row["id"]: row for row in read_jsonl(receiver_jsonl)}
    all_ids = {row["id"] for row in read_jsonl(args.run_dir / "manifest.jsonl")}
    if set(captions) != all_ids:
        raise ValueError("Caption IDs do not match prepared sample IDs")
    for sample_id, received in captions.items():
        if not received.get("condition_map") or "caption" not in received:
            raise ValueError(f"Receiver caption or condition map missing for {sample_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image-2.1 smoke generation requires CUDA")
    output_root = getattr(args, "output_dir", None) or args.run_dir / "generated"
    if output_root.exists() and worker_index is None:
        raise FileExistsError(f"Will not overwrite generated outputs: {output_root}")
    if worker_index is not None and not output_root.is_dir():
        raise FileNotFoundError(f"Worker output directory missing: {output_root}")
    load_started = time.monotonic()
    pipe = QwenImage21Pipeline.from_pretrained(
        str(args.model), torch_dtype=torch.bfloat16
    )
    component_dtypes = {
        name: str(getattr(pipe, name).dtype)
        for name in ("transformer", "text_encoder", "vae")
    }
    component_parameter_dtypes = {}
    for name in ("transformer", "text_encoder", "vae"):
        counts = {}
        for parameter in getattr(pipe, name).parameters():
            dtype = str(parameter.dtype)
            counts[dtype] = counts.get(dtype, 0) + parameter.numel()
        component_parameter_dtypes[name] = counts
    if placement == "cpu_offload":
        pipe.enable_model_cpu_offload()
    else:
        try:
            pipe.to("cuda")
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "Qwen-Image-2.1 cannot stay resident on this GPU. Retry in a "
                "new run with IMAGE_PLACEMENT=cpu_offload."
            ) from exc
    load_seconds = round(time.monotonic() - load_started, 3)

    if worker_index is None:
        output_root.mkdir()
    gpu_allocated_gib = (round(torch.cuda.memory_allocated() / 1024**3, 3)
                         if hasattr(torch.cuda, "memory_allocated") else None)
    profile_name = ("worker_single_profile.json" if worker_index is None
                    else f"worker_{worker_index}_profile.json")
    (output_root / profile_name).write_text(json.dumps({
        "worker_index": worker_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "placement": placement,
        "model_load_seconds": load_seconds,
        "gpu_allocated_after_load_gib": gpu_allocated_gib,
        "model": str(args.model),
        "requested_weight_dtype": str(torch.bfloat16),
        "component_dtypes": component_dtypes,
        "component_parameter_elements_by_dtype": component_parameter_dtypes,
        "sample_count": len(manifest),
    }, indent=2), encoding="utf-8")
    print(f"Loaded Qwen-Image-2.1 once on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}; "
          f"placement={placement}; load={load_seconds}s; "
          f"allocated={gpu_allocated_gib} GiB; component_dtypes={component_dtypes}", flush=True)
    results = []
    single_progress_started = time.monotonic()
    if worker_index is None:
        print_generation_progress(0, len(manifest), single_progress_started)
    for record in manifest:
        sample_dir = output_root / record["id"]
        sample_dir.mkdir()
        width, height = record["model_size"]
        branch_paths = {}
        condition_path = captions[record["id"]]["condition_map"]
        with Image.open(condition_path) as condition:
            if condition.size != (width, height):
                raise ValueError(
                    f"Receiver map size for {record['id']} is {condition.size}; "
                    f"expected {(width, height)}"
                )
            condition_rgb = condition.convert("RGB")
            for branch, caption_text in (
                ("fusion", captions[record["id"]]["caption"]),
            ):
                mask_guidance_applied = fusion_mask_guidance_applies(
                    caption_text, fusion_mask_prompt_policy)
                prompt = generation_prompt(caption_text, fusion_mask_prompt_policy)
                started = time.monotonic()
                generator = torch.Generator(device="cuda").manual_seed(args.seed)
                try:
                    image = pipe(
                        prompt=prompt,
                        image=condition_rgb,
                        width=width,
                        height=height,
                        num_inference_steps=args.steps,
                        generator=generator,
                    ).images[0]
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError(
                        f"Qwen-Image-2.1 ran out of GPU memory on "
                        f"{record['id']}/{branch}. Retry in a new run with "
                        "IMAGE_PLACEMENT=cpu_offload."
                    ) from exc
                if image.size != (width, height):
                    raise RuntimeError(
                        f"Unexpected output size for {record['id']}/{branch}: {image.size}"
                    )
                path = sample_dir / f"{branch}.png"
                image.convert("RGB").save(path)
                branch_paths[branch] = path
                results.append(
                    {
                        "id": record["id"],
                        "branch": branch,
                        "path": str(path.resolve()),
                        "prompt": prompt,
                        "generation_prompt_version": GENERATION_PROMPT_VERSION,
                        "fusion_mask_prompt_policy": fusion_mask_prompt_policy,
                        "caption_mask_count": caption_text.count("[MASK]") if caption_text else 0,
                        "mask_guidance_applied": mask_guidance_applied,
                        "seed": args.seed,
                        "steps": args.steps,
                        "generation_seconds": round(time.monotonic() - started, 3),
                        "model": str(args.model),
                        "placement": placement,
                        "model_load_seconds": load_seconds,
                        "torch_version": torch.__version__,
                        "diffusers_version": diffusers.__version__,
                        "worker_index": worker_index,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "condition_map": condition_path,
                        "caption_utf8_bits": len(captions[record["id"]]["caption"].encode("utf-8")) * 8 ,
                        "receiver_jsonl": str(receiver_jsonl) if receiver_jsonl else None,
                    }
                )
                print(f"Generated {record['id']}/{branch}: {path}")
                if worker_index is None:
                    print_generation_progress(len(results), len(manifest),
                                              single_progress_started)
        make_comparison({**record, "condition_map": condition_path}, branch_paths, sample_dir / "comparison.png")
    result_name = ("results.jsonl" if worker_index is None
                   else f"worker_{worker_index}_results.jsonl")
    save_jsonl_exclusive(output_root / result_name, results)
    print(f"Saved {len(results)} outputs in {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Pair Cityscapes GT and prepare model-sized images")
    prep.add_argument("--dataset-root", type=Path, required=True)
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--split", choices=["train", "val", "test"], default="val")
    prep.add_argument("--mask-source", choices=["gt", "drn"], default="gt")
    prep.add_argument("--limit", type=int, default=2)
    prep.add_argument("--width", type=int, default=1024)
    prep.add_argument("--height", type=int, default=512)
    prep.add_argument("--require-native-size", action="store_true",
                      help="Reject source images whose original size differs from width and height")
    prep.set_defaults(func=prepare)

    cap = sub.add_parser(
        "caption", help="Caption the source image; optional map input is an explicit extension"
    )
    cap.add_argument("--run-dir", type=Path, required=True)
    cap.add_argument("--llm-api", default=DEFAULT_LLM_API)
    cap.add_argument("--api-model-name", required=True)
    cap.add_argument("--i2t-model-family", choices=["qwen3_vl", "gemma4"],
                     default="qwen3_vl")
    cap.add_argument("--caption-input-mode", choices=CAPTION_INPUT_MODES,
                     default="source_only",
                     help="source_only uses one photograph; source_and_map adds the sender map")
    cap.add_argument(
        "--qwen-image-format",
        choices=["legacy_base64", "data_url"],
        default="legacy_base64",
        help="legacy_base64 matches the original local Qwen/SGLang workflow",
    )
    cap.add_argument("--max-new-tokens", type=int, default=256)
    cap.add_argument("--temperature", type=float, default=0.0)
    cap.add_argument("--request-timeout", type=int, default=300)
    cap.add_argument("--max-retries", type=int, default=3)
    cap.add_argument("--api-extra-body-json", default="{}")
    cap.add_argument("--prompt-file", type=Path,
                     help="JSON with name, system_prompt, user_instruction and optional input_mode")
    cap.set_defaults(func=caption)

    gen = sub.add_parser("generate", help="Generate fusion images from received text and maps")
    gen.add_argument("--run-dir", type=Path, required=True)
    gen.add_argument("--model", type=Path, required=True)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--steps", type=int, default=40)
    gen.add_argument("--cpu-offload", action="store_true")
    gen.add_argument("--placement", choices=("cpu_offload", "resident"))
    gen.add_argument("--fusion-mask-prompt-policy", choices=("conditional", "off"),
                     default="conditional",
                     help="Add missing-word guidance to Fusion only when the received caption contains [MASK]")
    gen.add_argument("--workers", type=int, default=1,
                     help="Parallel image workers; each worker uses one GPU")
    gen.add_argument("--gpu-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                     help="Physical GPU IDs for parallel image workers, e.g. 0,1")
    gen.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    gen.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    gen.add_argument("--receiver-jsonl", type=Path, required=True,
                     help="Recovered caption and map paths from a transport stage")
    gen.add_argument("--output-dir", type=Path,
                     help="Fresh directory for generated outputs; default RUN_DIR/generated")
    gen.set_defaults(func=generate)

    args = parser.parse_args()
    if args.command == "generate" and args.cpu_offload and args.placement == "resident":
        parser.error("--cpu-offload conflicts with --placement resident")
    if args.command == "prepare":
        if args.width < 64 or args.height < 64 or args.width % 16 or args.height % 16:
            parser.error("width and height must be multiples of 16 and at least 64")
    args.func(args)


if __name__ == "__main__":
    main()
