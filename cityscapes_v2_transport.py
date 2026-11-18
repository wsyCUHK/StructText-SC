#!/usr/bin/env python3
"""Lossless text-link and oracle-map receiver contract for Cityscapes V2.

This is a runnable interface stage, not a SwinJSCC or noisy-channel result.
The byte Huffman codebook is fixed by a separate, shared training corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import shutil
import struct
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def by_unique_id(rows: list[dict], name: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in rows:
        sample_id = row["id"]
        if sample_id in result:
            raise ValueError(f"Duplicate {name} ID: {sample_id}")
        result[sample_id] = row
    return result


class ByteHuffman:
    """Deterministic shared byte-level code; all 256 symbols remain encodable."""

    def __init__(self, training_bytes: bytes, pseudocount: int = 1) -> None:
        if pseudocount < 1:
            raise ValueError("pseudocount must be positive")
        frequencies = Counter({i: pseudocount for i in range(256)})
        frequencies.update(training_bytes)
        heap = []
        for byte in range(256):
            heapq.heappush(heap, (frequencies[byte], byte, byte))
        order = 256
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            heapq.heappush(heap, (left[0] + right[0], order, (left[2], right[2])))
            order += 1
        self.root = heap[0][2]
        self.codes: dict[int, tuple[int, ...]] = {}

        def visit(node: int | tuple, prefix: tuple[int, ...]) -> None:
            if isinstance(node, int):
                self.codes[node] = prefix or (0,)
            else:
                visit(node[0], prefix + (0,))
                visit(node[1], prefix + (1,))

        visit(self.root, ())

    def encode(self, raw: bytes) -> tuple[bytes, int]:
        bit_length = sum(len(self.codes[byte]) for byte in raw)
        payload = bytearray((bit_length + 7) // 8)
        offset = 0
        for byte in raw:
            for bit in self.codes[byte]:
                if bit:
                    payload[offset // 8] |= 1 << (7 - offset % 8)
                offset += 1
        return bytes(payload), bit_length

    def decode(self, payload: bytes, bit_length: int, expected_bytes: int) -> bytes:
        if bit_length < 0 or bit_length > len(payload) * 8:
            raise ValueError("Invalid Huffman bit length")
        output = bytearray()
        node = self.root
        for offset in range(bit_length):
            bit = (payload[offset // 8] >> (7 - offset % 8)) & 1
            node = node[bit]
            if isinstance(node, int):
                output.append(node)
                node = self.root
        if node is not self.root or len(output) != expected_bytes:
            raise ValueError("Huffman stream did not decode to the expected byte count")
        return bytes(output)


def load_shared_corpus(corpus_path: Path, require_train_metadata: bool) -> tuple[bytes, str]:
    corpus = corpus_path.read_bytes()
    if not corpus.strip():
        raise ValueError("Codebook training corpus is empty")
    corpus_sha256 = hashlib.sha256(corpus).hexdigest()
    if require_train_metadata:
        metadata_path = corpus_path.with_suffix(corpus_path.suffix + ".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("split") != "train" or metadata.get("sha256") != corpus_sha256:
            raise ValueError("Codebook metadata must verify a frozen train corpus")
    return corpus, corpus_sha256


def transport(run_dir: Path, corpus_path: Path, output_dir: Path, pseudocount: int,
              require_train_metadata: bool = False) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Will not overwrite: {output_dir}")
    corpus, corpus_sha256 = load_shared_corpus(corpus_path, require_train_metadata)
    captions_path = run_dir / "captions.jsonl"
    if corpus_path.resolve() == captions_path.resolve():
        raise ValueError("Evaluation captions cannot train their own Huffman codebook")
    codec = ByteHuffman(corpus, pseudocount)
    manifest = read_jsonl(run_dir / "manifest.jsonl")
    captions = by_unique_id(read_jsonl(captions_path), "caption")
    ids = by_unique_id(manifest, "manifest")
    if set(ids) != set(captions):
        raise ValueError("Manifest and caption IDs differ")

    # Validate all inputs before creating outputs.
    for row in manifest:
        if not Path(row["condition_map"]).is_file():
            raise FileNotFoundError(row["condition_map"])
        if not captions[row["id"]].get("caption"):
            raise ValueError(f"Empty caption: {row['id']}")

    (output_dir / "maps").mkdir(parents=True)
    (output_dir / "text_payloads").mkdir()
    (output_dir / "receiver_text").mkdir()
    receiver_rows = []
    stats_rows = []
    for row in manifest:
        sample_id = row["id"]
        text = captions[sample_id]["caption"]
        raw = text.encode("utf-8")
        payload, bit_length = codec.encode(raw)
        wire_path = output_dir / "text_payloads" / f"{sample_id}.bin"
        wire_path.write_bytes(struct.pack(">II", len(raw), bit_length) + payload)
        received_wire = wire_path.read_bytes()
        byte_count, received_bits = struct.unpack(">II", received_wire[:8])
        recovered = codec.decode(received_wire[8:], received_bits, byte_count).decode("utf-8")
        if recovered != text:
            raise RuntimeError(f"Lossless text transport failed: {sample_id}")
        recovered_text_path = output_dir / "receiver_text" / f"{sample_id}.txt"
        recovered_text_path.write_text(recovered + "\n", encoding="utf-8")
        source_map = Path(row["condition_map"])
        receiver_map = output_dir / "maps" / f"{sample_id}.png"
        shutil.copyfile(source_map, receiver_map)
        receiver_rows.append({
            "id": sample_id,
            "caption": recovered,
            "caption_path": str(recovered_text_path.resolve()),
            "condition_map": str(receiver_map.resolve()),
            "text_link": "ideal_huffman_lossless",
            "map_link": "oracle_clean_copy",
        })
        stats_rows.append({
            "id": sample_id,
            "utf8_bits": len(raw) * 8,
            "huffman_payload_bits": bit_length,
            "huffman_packed_bits": len(payload) * 8,
            "text_wire_bits": 64 + len(payload) * 8,
            "text_wire_path": str(wire_path.resolve()),
            "text_header_bits": 64,
            "map_png_file_bits_reference_only": source_map.stat().st_size * 8,
            "map_channel_uses": None,
            "map_channel_mode": "oracle_clean_copy",
            "codebook_sha256": corpus_sha256,
        })
    for name, rows in (("receiver.jsonl", receiver_rows), ("stats.jsonl", stats_rows)):
        with (output_dir / name).open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "config.json").write_text(json.dumps({
        "codebook_corpus": str(corpus_path.resolve()),
        "codebook_sha256": corpus_sha256,
        "train_metadata_verified": require_train_metadata,
        "pseudocount": pseudocount,
        "text_mode": "ideal_huffman_lossless",
        "map_mode": "oracle_clean_copy",
        "text_header_bits": 64,
        "note": "No FEC, noisy wireless link, or SwinJSCC. Map PNG bits are reference only.",
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(receiver_rows)} receiver records to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--codebook-corpus", required=True, type=Path,
                        help="Fixed text corpus from training split; shared by sender and receiver")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pseudocount", type=int, default=1)
    parser.add_argument("--require-train-metadata", action="store_true")
    parser.add_argument("--verify-codebook-only", action="store_true")
    args = parser.parse_args()
    if args.verify_codebook_only:
        _, digest = load_shared_corpus(args.codebook_corpus, args.require_train_metadata)
        print(f"Verified shared codebook corpus: sha256={digest}")
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --verify-codebook-only is set")
    transport(args.run_dir, args.codebook_corpus,
              args.output_dir or args.run_dir / "transport_oracle", args.pseudocount,
              args.require_train_metadata)


if __name__ == "__main__":
    main()
