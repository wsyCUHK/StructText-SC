#!/usr/bin/env python3
"""Validate the released StructText-SC JSONL without model dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPOSITORY_ROOT / "train_20260428.jsonl"


def validate_data(path: Path, image_root: Path | None = None,
                  expected_sha256: str | None = None) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError("Dataset SHA-256 differs from the expected release file")
    rows = []
    image_paths = []
    canonical_rows = set()
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or set(row) != {"messages", "images"}:
            raise ValueError(f"Line {line_number}: expected messages and images fields")
        messages = row["messages"]
        if (not isinstance(messages, list) or len(messages) != 3
                or any(not isinstance(message, dict) for message in messages)
                or [message.get("role") for message in messages]
                != ["system", "user", "assistant"]):
            raise ValueError(f"Line {line_number}: expected system/user/assistant messages")
        if any(not isinstance(message.get("content"), str)
               or not message["content"].strip() for message in messages):
            raise ValueError(f"Line {line_number}: empty or non-text message content")
        if "<image>" not in messages[1]["content"]:
            raise ValueError(f"Line {line_number}: user message lacks the image marker")
        images = row["images"]
        if (not isinstance(images, list) or len(images) != 1
                or not isinstance(images[0], str) or not images[0].strip()):
            raise ValueError(f"Line {line_number}: expected exactly one image reference")
        image = images[0]
        relative = PurePosixPath(image)
        if (relative.is_absolute() or PureWindowsPath(image).drive
                or ".." in relative.parts or "\\" in image
                or image.endswith("/") or not relative.suffix):
            raise ValueError(f"Line {line_number}: image path must be a relative file path")
        canonical = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if canonical in canonical_rows:
            raise ValueError(f"Line {line_number}: duplicate complete record")
        canonical_rows.add(canonical)
        image_paths.append(image)
        rows.append(row)
    if not rows:
        raise ValueError("Dataset contains no records")
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("Dataset contains duplicate image references")
    if image_root is not None:
        base = image_root.resolve()
        missing = []
        for image in image_paths:
            target = (base / image).resolve()
            if not target.is_relative_to(base):
                raise ValueError("An image reference resolves outside the image root")
            if not target.is_file():
                missing.append(image)
        if missing:
            raise ValueError(f"Missing {len(missing)} referenced images; first: {missing[0]}")
    return {
        "file": path.name,
        "bytes": len(raw),
        "sha256": digest,
        "records": len(rows),
        "unique_images": len(set(image_paths)),
        "images_per_record": 1,
        "duplicate_records": 0,
        "duplicate_image_references": 0,
        "message_roles": ["system", "user", "assistant"],
        "image_path_prefixes": dict(sorted(Counter(
            str(PurePosixPath(image).parent) for image in image_paths).items())),
        "image_files_checked": image_root is not None,
        "expected_sha256_checked": expected_sha256 is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output", type=Path,
                        help="Optional report; refuses to overwrite an existing file")
    args = parser.parse_args()
    try:
        expected = args.expected_sha256
        if expected is None and args.data.resolve() == DEFAULT_DATA:
            provenance = json.loads((REPOSITORY_ROOT / "data/provenance.json").read_text(
                encoding="utf-8"))
            expected = provenance["sha256"]
        report = validate_data(args.data, args.image_root, expected)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            if args.output.resolve() == args.data.resolve():
                raise ValueError("Report path must differ from the data file")
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized)
        print(serialized, end="")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")


if __name__ == "__main__":
    main()
