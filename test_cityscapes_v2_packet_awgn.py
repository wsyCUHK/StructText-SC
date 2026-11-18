import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cityscapes_v2_packet_awgn import awgn_rgb_map, finalize, prepare
from cityscapes_v2_transport import ByteHuffman, read_jsonl


class PacketAwgnAdapterTests(unittest.TestCase):
    def test_existing_packet_codec_matches_oracle_byte_codebook(self):
        # Only the Huffman class is exercised; RS is absent in the local test env.
        module_name = "hf_rs_packet_loss_codebook_test"
        spec = importlib.util.spec_from_file_location(module_name, ROOT / "hf_rs_packet_loss.py")
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        reedsolo_stub = types.ModuleType("reedsolo")
        reedsolo_stub.RSCodec = object
        reedsolo_stub.ReedSolomonError = Exception
        with patch.dict(sys.modules, {"reedsolo": reedsolo_stub, module_name: module}):
            spec.loader.exec_module(module)
        corpus = "A car on a road. 一辆车。\n"
        packet_codec = module.SharedByteHuffmanCodec([corpus], pseudocount=1)
        oracle_codec = ByteHuffman(corpus.encode("utf-8"), pseudocount=1)
        for byte in range(256):
            self.assertEqual(packet_codec.codes[byte],
                             "".join(map(str, oracle_codec.codes[byte])))
        self.assertEqual(packet_codec.encode("红车".encode("utf-8")),
                         oracle_codec.encode("红车".encode("utf-8")))

    def test_awgn_is_seeded_and_reports_preclip_snr(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "map.png"
            pattern = np.tile(np.array([[[128, 64, 192], [64, 192, 128]]], dtype=np.uint8), (64, 32, 1))
            Image.fromarray(pattern).save(source)
            first, second = root / "first.png", root / "second.png"
            stats = awgn_rgb_map(source, first, snr_db=10, seed=7)
            awgn_rgb_map(source, second, snr_db=10, seed=7)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertAlmostEqual(stats["snr_db_realized_before_clipping"], 10, delta=0.3)
            self.assertIsNone(stats["map_channel_uses"])
            self.assertNotEqual(first.read_bytes(), source.read_bytes())

    def test_prepared_ids_survive_packet_batch_and_metric_reference_is_sender(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "data"
            run.mkdir()
            source = run / "source.png"
            condition = run / "condition.png"
            Image.new("RGB", (16, 8), "gray").save(source)
            Image.new("RGB", (16, 8), "blue").save(condition)
            sample_id = "city_000001_000002"
            (run / "manifest.jsonl").write_text(json.dumps({
                "id": sample_id, "source": str(source),
                "condition_map": str(condition), "model_size": [16, 8],
            }) + "\n", encoding="utf-8")
            (run / "captions.jsonl").write_text(json.dumps({
                "id": sample_id, "caption": "A blue car on the road.",
            }) + "\n", encoding="utf-8")
            corpus = root / "train.txt"
            contents = b"A car on the road.\n"
            corpus.write_bytes(contents)
            corpus.with_suffix(".txt.json").write_text(json.dumps({
                "split": "train", "sha256": hashlib.sha256(contents).hexdigest(),
            }), encoding="utf-8")
            transport = run / "transport_packet_awgn"
            prepare(run, transport, corpus)
            self.assertEqual((transport / "sender_lines.txt").read_text(),
                             "A blue car on the road.\n")
            (transport / "recovered_lines.txt").write_text("A [MASK] car on the road.\n", encoding="utf-8")
            (transport / "packet_summary.json").write_text(json.dumps({
                "codebook_corpus": str(corpus.resolve()), "caption_count": 1,
                "per_caption": [{"caption_index": 0, "huffman_source_bits": 20,
                                 "transmitted_packet_bits": 8192,
                                 "realized_packet_loss_rate": 0.25}],
            }), encoding="utf-8")
            finalize(run, transport, snr_db=10, seed=42)
            receiver = read_jsonl(transport / "receiver.jsonl")[0]
            self.assertEqual(receiver["caption"], "A [MASK] car on the road.")
            self.assertTrue(Path(receiver["condition_map"]).is_file())
            self.assertEqual(read_jsonl(transport / "stats.jsonl")[0]["transmitted_packet_bits"], 8192)

    def test_rejects_packet_result_with_wrong_count(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "manifest.jsonl").write_text('{"id":"x","condition_map":"unused"}\n')
            output = root / "transport"
            output.mkdir()
            (output / "sender_ids.jsonl").write_text('{"id":"x"}\n')
            (output / "recovered_lines.txt").write_text("")
            corpus = root / "train.txt"
            corpus.write_bytes(b"road\n")
            digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
            corpus.with_suffix(".txt.json").write_text(json.dumps({
                "split": "train", "sha256": digest,
            }))
            (output / "codebook.json").write_text(json.dumps({
                "corpus": str(corpus.resolve()), "sha256": digest,
            }))
            (output / "packet_summary.json").write_text(json.dumps({
                "codebook_corpus": str(corpus.resolve()), "caption_count": 0,
                "per_caption": [],
            }))
            with self.assertRaises(ValueError):
                finalize(root, output, 10, 42)

    def test_precomputed_swinjscc_map_pairs_with_packet_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "data"
            run.mkdir()
            source = run / "condition.png"
            Image.new("RGB", (16, 8), "blue").save(source)
            sample_id = "city_000001_000002"
            (run / "manifest.jsonl").write_text(json.dumps({
                "id": sample_id, "condition_map": str(source),
                "model_size": [16, 8],
            }) + "\n", encoding="utf-8")
            (run / "captions.jsonl").write_text(json.dumps({
                "id": sample_id, "caption": "A blue car.",
            }) + "\n", encoding="utf-8")
            corpus = root / "train.txt"
            corpus.write_bytes(b"A blue car.\n")
            corpus.with_suffix(".txt.json").write_text(json.dumps({
                "split": "train", "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            transport = run / "transport_packet_swinjscc"
            prepare(run, transport, corpus)
            (transport / "recovered_lines.txt").write_text("A [MASK] car.\n", encoding="utf-8")
            (transport / "packet_summary.json").write_text(json.dumps({
                "codebook_corpus": str(corpus.resolve()), "caption_count": 1,
                "per_caption": [{"caption_index": 0, "transmitted_packet_bits": 8192}],
            }), encoding="utf-8")
            maps = transport / "maps"
            maps.mkdir()
            received = maps / f"{sample_id}.png"
            Image.new("RGB", (16, 8), "red").save(received)
            map_stats = transport / "map_stats.jsonl"
            record = {
                "id": "wrong_id", "sender_map": str(source),
                "receiver_map": str(received),
                "map_channel_mode": "swinjscc_awgn_latent",
                "snr_db": 10.0, "map_seed": 42, "map_channel_uses": 24,
            }
            map_stats.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "IDs/order"):
                finalize(run, transport, 10, 42, map_stats)
            record["id"] = sample_id
            map_stats.write_text(json.dumps(record) + "\n", encoding="utf-8")
            finalize(run, transport, 10, 42, map_stats)
            receiver_row = read_jsonl(transport / "receiver.jsonl")[0]
            stats_row = read_jsonl(transport / "stats.jsonl")[0]
            self.assertEqual(receiver_row["caption"], "A [MASK] car.")
            self.assertEqual(Path(receiver_row["condition_map"]), received)
            self.assertEqual(receiver_row["map_link"], "swinjscc_awgn_latent")
            self.assertEqual(stats_row["map_channel_uses"], 24)
            self.assertEqual(stats_row["transmitted_packet_bits"], 8192)


if __name__ == "__main__":
    unittest.main()
