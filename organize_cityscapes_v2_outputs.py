#!/usr/bin/env python3
"""Build a readable, non-destructive view of one Cityscapes V2 run.

The pipeline's data/ paths remain authoritative. On the same
filesystem this view uses hard links, so it does not duplicate image bytes.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path


ARTIFACTS = "artifacts"


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def by_id(rows: list[dict], label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        sample_id = row["id"]
        if (not isinstance(sample_id, str) or not sample_id or
                Path(sample_id).name != sample_id or sample_id in {".", ".."}):
            raise ValueError(f"Unsafe {label} sample ID: {sample_id}")
        if sample_id in result:
            raise ValueError(f"Duplicate {label} sample ID: {sample_id}")
        result[sample_id] = row
    return result


def link_or_copy(source: Path, destination: Path) -> str:
    """Keep a portable fallback for an output view on another filesystem."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.ENOSYS,
                             getattr(errno, "EOPNOTSUPP", -1)}:
            raise
    shutil.copy2(source, destination)
    return "copy"


def resolve_single(data_dir: Path, pattern: str, label: str) -> Path:
    paths = [path for path in data_dir.glob(pattern) if path.is_dir()]
    if len(paths) != 1:
        raise ValueError(f"Expected one {label} directory in {data_dir}; found {len(paths)}")
    return paths[0]


def prepare_view(run_dir: Path, data_dir: Path | None = None,
                 transport_dir: Path | None = None,
                 generated_dir: Path | None = None) -> Path:
    data_dir = data_dir or run_dir / "data"
    transport_dir = transport_dir or resolve_single(data_dir, "transport_*", "transport")
    generated_dir = generated_dir or resolve_single(data_dir, "generated_*", "generation")
    view = run_dir / ARTIFACTS
    if view.exists() or (run_dir / "README.md").exists():
        raise FileExistsError(f"Refusing to overwrite output view in {run_dir}")

    manifest_rows = read_jsonl(data_dir / "manifest.jsonl")
    manifest = by_id(manifest_rows, "manifest")
    sender = by_id(read_jsonl(data_dir / "captions.jsonl"), "sender caption")
    receiver = by_id(read_jsonl(transport_dir / "receiver.jsonl"), "receiver")
    if set(manifest) != set(sender) or set(manifest) != set(receiver):
        raise ValueError("Manifest, sender captions, and receiver IDs differ")
    results = {}
    for row in read_jsonl(generated_dir / "results.jsonl"):
        key = (row["id"], row["branch"])
        if row["branch"] not in {"fusion"} or key in results:
            raise ValueError(f"Unexpected or duplicate generated result: {key}")
        results[key] = row
    expected = {(sample_id, branch) for sample_id in manifest
                for branch in ("fusion",)}
    if set(results) != expected:
        raise ValueError("Generated branches do not match prepared sample IDs")

    plan: list[tuple[Path, Path, str]] = []
    destinations = set()

    def add(source: Path | str, relative: str, role: str) -> None:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = Path(relative)
        if destination in destinations:
            raise ValueError(f"Duplicate output view path: {destination}")
        destinations.add(destination)
        plan.append((source, destination, role))

    add(data_dir / "manifest.jsonl", "01_source/manifest.jsonl", "source_manifest")
    add(data_dir / "captions.jsonl", "02_sender_text/captions.jsonl", "sender_captions")
    blip_profile = data_dir / "blip_caption_profile.json"
    if blip_profile.is_file():
        add(blip_profile, "02_sender_text/blip_caption_profile.json", "blip_caption_profile")
    drn_profile = data_dir / "sender_drn_profile.json"
    if drn_profile.is_file():
        add(drn_profile, "01_source/sender_drn_profile.json", "sender_drn_profile")
    add(transport_dir / "receiver.jsonl", "05_generation/receiver.jsonl", "receiver_inputs")
    add(generated_dir / "results.jsonl", "05_generation/results.jsonl", "generation_results")
    for row in manifest_rows:
        sample_id = row["id"]
        filename = f"{sample_id}.png"
        add(row["source"], f"01_source/rgb/{filename}", "source_rgb_resized")
        if row.get("mask_source") == "DRN-D-105 predicted":
            add(row["condition_map"], f"01_source/drn_color_mask/{filename}",
                "sender_drn_color_mask")
            add(row["sender_train_ids"], f"01_source/drn_train_ids/{filename}",
                "sender_drn_train_ids")
        else:
            add(row["condition_map"], f"01_source/gt_color_mask/{filename}",
                "sender_gt_color_mask")
        if row.get("label_ids"):
            label_folder = ("reference_gt_label_ids" if
                            row.get("mask_source") == "DRN-D-105 predicted" else
                            "gt_label_ids")
            add(row["label_ids"], f"01_source/{label_folder}/{filename}",
                label_folder)
        sender_text = data_dir / "sender_text" / f"{sample_id}.txt"
        if sender_text.is_file():
            add(sender_text, f"02_sender_text/per_sample/{sample_id}.txt", "sender_caption")
        received_text = receiver[sample_id].get("caption_path")
        if received_text:
            add(received_text, f"03_text_channel/per_sample/{sample_id}.txt",
                "recovered_caption")
        add(receiver[sample_id]["condition_map"],
            f"04_mask_channel/received/{filename}", "receiver_condition_map")
        for branch in ("fusion",):
            result = results[(sample_id, branch)]
            if result.get("condition_map") != receiver[sample_id]["condition_map"]:
                raise ValueError(f"Generation used a different map: {sample_id}/{branch}")
            add(result["path"], f"05_generation/{branch}/{filename}",
                f"generated_{branch}")
        comparison = generated_dir / sample_id / "comparison.png"
        add(comparison, f"05_generation/comparison/{filename}", "three_panel_comparison")

    text_files = {
        "sender_lines.txt": "sender_lines.txt",
        "sender_ids.jsonl": "sender_ids.jsonl",
        "recovered_lines.txt": "recovered_lines.txt",
        "packet_summary.json": "packet_summary.json",
        "codebook.json": "codebook.json",
        "stats.jsonl": "transport_stats.jsonl",
        "config.json": "transport_config.json",
    }
    for filename, target in text_files.items():
        source = transport_dir / filename
        if source.is_file():
            add(source, f"03_text_channel/{target}", filename)
    for source in sorted((transport_dir / "text_payloads").glob("*")):
        if source.is_file():
            add(source, f"03_text_channel/text_payloads/{source.name}", "encoded_text_payload")

    mask_files = {
        "map_stats.jsonl": "map_stats.jsonl",
        "map_summary.json": "map_summary.json",
        "swinjscc_config.json": "swinjscc_config.json",
    }
    for filename, target in mask_files.items():
        source = transport_dir / filename
        if source.is_file():
            add(source, f"04_mask_channel/{target}", filename)
    for source_name, target_name in (
        ("maps_raw", "raw_decoder"),
        ("maps_palette", "palette_projection"),
        ("maps_trainIds", "train_ids"),
    ):
        for source in sorted((transport_dir / source_name).glob("*")):
            if source.is_file():
                add(source, f"04_mask_channel/{target_name}/{source.name}",
                    f"map_{target_name}")

    view.mkdir()
    indexed = []
    for source, relative, role in plan:
        method = link_or_copy(source, view / relative)
        indexed.append({"role": role, "path": str(relative),
                        "source": str(source.resolve()), "method": method})
    with (view / "index.jsonl").open("x", encoding="utf-8") as handle:
        for row in indexed:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (view / "03_text_channel" / "recovered_captions.jsonl").open(
            "x", encoding="utf-8") as handle:
        for row in manifest_rows:
            sample_id = row["id"]
            handle.write(json.dumps({"id": sample_id,
                                     "caption": receiver[sample_id]["caption"]},
                                    ensure_ascii=False) + "\n")
    mask_sources = {row.get("mask_source", "Cityscapes gtFine ground truth")
                    for row in manifest_rows}
    if len(mask_sources) != 1:
        raise ValueError("Mixed sender mask sources in one run")
    mask_description = ("DRN 预测的彩色分割图和 trainId 标签；GT 标签若存在也会保存。" if
                        "DRN-D-105 predicted" in mask_sources else
                        "Cityscapes GT 彩色分割图与 labelIds。")
    (view / "README.md").write_text(
        "# GenSemCom V2 产物索引\n\n"
        "- `01_source/`：输入 RGB、发送端" + mask_description + "\n" +
        "RGB 用于图生文和可选的发送端 DRN 分割，不经过图像信道。\n"
        "- `02_sender_text/`：图生文模型生成的全部文本 `captions.jsonl` 和逐张文本。\n"
        "- `03_text_channel/`：文本信道输入、恢复文本、码本与丢包统计。\n"
        "- `04_mask_channel/`：信道后交给生图模型的分割图及诊断图、统计。\n"
        "- `05_generation/`：Fusion、三图对比、完整 prompt 和结果记录。\n"
        "本目录是方便查看的文件视图；`data/`、`logs/` 和 `config/` "
        "保留作为原始运行记录。`index.jsonl` 给出每个文件的原路径。"
        "同一文件系统优先创建硬链接，请将产物视为只读。\n",
        encoding="utf-8",
    )
    (run_dir / "README.md").write_text(
        "# GenSemCom V2 运行结果\n\n"
        "从 [`artifacts/`](artifacts/README.md) 按输入、信道和生成顺序查看。"
        "`data/` 是工作流原始文件，`logs/` 是过程日志，`config/` 保存本次配置和代码快照。"
        "\n",
        encoding="utf-8",
    )
    print(f"Organized {len(manifest)} samples into {view}")
    return view



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-run-dir", type=Path)
    parser.add_argument("--transport-dir", type=Path)
    parser.add_argument("--generated-dir", type=Path)
    args = parser.parse_args()
    prepare_view(args.run_dir, args.data_run_dir, args.transport_dir, args.generated_dir)

if __name__ == "__main__":
    main()
