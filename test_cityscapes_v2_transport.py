import json
import argparse
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cityscapes_v2_transport import ByteHuffman, load_shared_corpus, read_jsonl, transport
import cityscapes_gt_smoke as smoke
from build_cityscapes_v2_codebook_corpus import build_corpus


class CityscapesV2TransportTests(unittest.TestCase):
    def test_huffman_roundtrip_unseen_utf8_and_padding(self) -> None:
        codec = ByteHuffman(b"road car sky road")
        raw = "车道，night 🚗".encode("utf-8")
        payload, bit_length = codec.encode(raw)
        self.assertEqual(codec.decode(payload, bit_length, len(raw)), raw)
        self.assertLessEqual(bit_length, len(payload) * 8)
        with self.assertRaises(ValueError):
            codec.decode(payload, bit_length, len(raw) + 1)

    def test_receiver_contract_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_dir = root / "run"
            run_dir.mkdir()
            map_path = run_dir / "condition_map.png"
            map_path.write_bytes(b"map bytes")
            (run_dir / "manifest.jsonl").write_text(json.dumps({
                "id": "city_000000_000001", "condition_map": str(map_path),
            }) + "\n", encoding="utf-8")
            captions_path = run_dir / "captions.jsonl"
            captions_path.write_text(json.dumps({
                "id": "city_000000_000001", "caption": "A red car near a sign.",
            }) + "\n", encoding="utf-8")
            corpus = root / "train_captions.txt"
            corpus.write_text("A car on a road. A blue sky.", encoding="utf-8")
            output = run_dir / "transport_oracle"
            transport(run_dir, corpus, output, 1)
            receiver = read_jsonl(output / "receiver.jsonl")[0]
            stats = read_jsonl(output / "stats.jsonl")[0]
            self.assertEqual(receiver["caption"], "A red car near a sign.")
            self.assertEqual(Path(receiver["condition_map"]).read_bytes(), b"map bytes")
            self.assertEqual(stats["text_wire_bits"],
                             Path(stats["text_wire_path"]).stat().st_size * 8)
            self.assertIsNone(stats["map_channel_uses"])
            with self.assertRaises(FileExistsError):
                transport(run_dir, corpus, output, 1)

    def test_rejects_eval_captions_as_codebook(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            captions = root / "captions.jsonl"
            captions.write_text('{"id":"x","caption":"road"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                transport(root, captions, root / "out", 1)

    def test_generate_consumes_receiver_map_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            sender_map = root / "sender_map.png"
            receiver_map = root / "receiver_map.png"
            Image.new("RGB", (8, 4), "black").save(source)
            Image.new("RGB", (8, 4), "red").save(sender_map)
            Image.new("RGB", (8, 4), "blue").save(receiver_map)
            (root / "manifest.jsonl").write_text(json.dumps({
                "id": "x", "source": str(source),
                "condition_map": str(sender_map), "model_size": [8, 4],
            }) + "\n", encoding="utf-8")
            receiver_jsonl = root / "receiver.jsonl"
            receiver_jsonl.write_text(json.dumps({
                "id": "x", "caption": "silver [MASK] hatchback",
                "condition_map": str(receiver_map),
            }) + "\n", encoding="utf-8")

            seen = []
            placements = []

            class FakePipe:
                def __init__(self):
                    parameter = types.SimpleNamespace(dtype="bf16", numel=lambda: 1)
                    component = types.SimpleNamespace(
                        dtype="bf16", parameters=lambda: [parameter])
                    self.transformer = component
                    self.text_encoder = component
                    self.vae = component

                def enable_model_cpu_offload(self):
                    placements.append("cpu_offload")
                    return None

                def to(self, device):
                    placements.append(device)
                    return self

                def __call__(self, **kwargs):
                    seen.append(kwargs)
                    return types.SimpleNamespace(images=[Image.new("RGB", (8, 4), "green")])

            fake_pipe = FakePipe()
            fake_diffusers = types.SimpleNamespace(
                QwenImage21Pipeline=types.SimpleNamespace(
                    from_pretrained=lambda *args, **kwargs: fake_pipe),
                __version__="test",
            )
            fake_torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: True),
                Generator=lambda device: types.SimpleNamespace(manual_seed=lambda seed: seed),
                bfloat16="bf16", __version__="test",
            )
            with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
                smoke.generate(argparse.Namespace(
                    run_dir=root, receiver_jsonl=receiver_jsonl,
                    output_dir=root / "generated_oracle", model=root / "model",
                    seed=42, steps=4, cpu_offload=True,
                ))
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["image"].getpixel((0, 0)), (0, 0, 255))
            self.assertIn("silver [MASK] hatchback", seen[0]["prompt"])
            self.assertIn("recovered after packet loss", seen[0]["prompt"])
            rows = read_jsonl(root / "generated_oracle" / "results.jsonl")
            self.assertEqual(rows[0]["condition_map"], str(receiver_map))
            self.assertEqual([row["caption_mask_count"] for row in rows], [1])
            self.assertEqual([row["mask_guidance_applied"] for row in rows], [True])
            self.assertEqual(rows[0]["fusion_mask_prompt_policy"], "conditional")
            with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
                smoke.generate(argparse.Namespace(
                    run_dir=root, receiver_jsonl=receiver_jsonl,
                    output_dir=root / "generated_resident", model=root / "model",
                    seed=42, steps=4, cpu_offload=False, placement="resident",
                ))
            self.assertEqual(placements, ["cpu_offload", "cuda"])
            resident_rows = read_jsonl(root / "generated_resident" / "results.jsonl")
            self.assertTrue(all(row["placement"] == "resident" for row in resident_rows))
            profile = json.loads((root / "generated_resident" /
                                  "worker_single_profile.json").read_text())
            self.assertEqual(profile["placement"], "resident")

    def test_generation_rejects_missing_receiver_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "receiver-jsonl is required"):
            smoke.generate(argparse.Namespace(receiver_jsonl=None))

    def test_parallel_image_workers_pin_gpus_and_merge_in_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample_ids = ["first", "second", "third"]
            (root / "manifest.jsonl").write_text(
                "".join(json.dumps({"id": sample_id}) + "\n" for sample_id in sample_ids),
                encoding="utf-8",
            )
            receiver = root / "receiver.jsonl"
            receiver.write_text(
                "".join(json.dumps({"id": sample_id}) + "\n" for sample_id in sample_ids),
                encoding="utf-8",
            )
            launched = []

            class FakeProcess:
                def poll(self):
                    return 0

            def fake_popen(command, **kwargs):
                worker = int(command[command.index("--worker-index") + 1])
                worker_count = int(command[command.index("--worker-count") + 1])
                output = Path(command[command.index("--output-dir") + 1])
                launched.append((worker, kwargs["env"]["CUDA_VISIBLE_DEVICES"], command))
                rows = []
                for sample_index, sample_id in enumerate(sample_ids):
                    if sample_index % worker_count != worker:
                        continue
                    sample_dir = output / sample_id
                    sample_dir.mkdir()
                    for branch in ("fusion",):
                        path = sample_dir / f"{branch}.png"
                        path.write_bytes(b"test image")
                        rows.append({"id": sample_id, "branch": branch, "path": str(path)})
                        kwargs["stdout"].write(f"Generated {sample_id}/{branch}: {path}\n")
                    (sample_dir / "comparison.png").write_bytes(b"test comparison")
                kwargs["stdout"].flush()
                smoke.save_jsonl_exclusive(output / f"worker_{worker}_results.jsonl", rows)
                return FakeProcess()

            terminal = io.StringIO()
            with (patch.object(smoke.subprocess, "Popen", side_effect=fake_popen),
                  redirect_stdout(terminal)):
                smoke.generate(argparse.Namespace(
                    run_dir=root, receiver_jsonl=receiver, output_dir=root / "generated",
                    model=root / "model", seed=42, steps=4, cpu_offload=True,
                    workers=2, gpu_devices="2,3", worker_index=None,
                    fusion_mask_prompt_policy="off",
                ))
            self.assertEqual([(worker, gpu) for worker, gpu, _ in launched],
                             [(0, "2"), (1, "3")])
            self.assertTrue(all(command[command.index("--placement") + 1] == "cpu_offload"
                                for _, _, command in launched))
            self.assertTrue(all(command[command.index("--fusion-mask-prompt-policy") + 1]
                                == "off" for _, _, command in launched))
            self.assertIn("生图进度", terminal.getvalue())
            self.assertIn("3/3 (100.0%)", terminal.getvalue())
            rows = read_jsonl(root / "generated" / "results.jsonl")
            self.assertEqual([(row["id"], row["branch"]) for row in rows], [
                (sample_id, branch) for sample_id in sample_ids
                for branch in ("fusion",)
            ])
            launched.clear()
            with patch.object(smoke.subprocess, "Popen", side_effect=fake_popen):
                smoke.generate(argparse.Namespace(
                    run_dir=root, receiver_jsonl=receiver,
                    output_dir=root / "generated_resident",
                    model=root / "model", seed=42, steps=4,
                    cpu_offload=False, placement="resident",
                    workers=2, gpu_devices="2,3", worker_index=None,
                    fusion_mask_prompt_policy="off",
                ))
            self.assertTrue(all(command[command.index("--placement") + 1] == "resident"
                                for _, _, command in launched))

    def test_worker_progress_counts_only_complete_generation_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "worker.log"
            log.write_bytes(b"Loading model\n\rGenerated x/mask_only: x.png\nGenerated x/fu")
            count, offset, pending = smoke.read_new_generation_count(log, 0, b"")
            self.assertEqual(count, 1)
            with log.open("ab") as handle:
                handle.write(b"sion: x.png\n")
            count, offset, pending = smoke.read_new_generation_count(log, offset, pending)
            self.assertEqual((count, pending), (1, b""))

    def test_parallel_result_merge_rejects_missing_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            image = output / "x.png"
            image.write_bytes(b"image")
            smoke.save_jsonl_exclusive(output / "worker_0_results.jsonl", [
                {"id": "x", "branch": "mask_only", "path": str(image)},
            ])
            with self.assertRaises(ValueError):
                smoke.merge_generation_results(output, [{"id": "x"}], 1)
            self.assertFalse((output / "results.jsonl").exists())

    def test_codebook_corpus_requires_train_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.jsonl"
            captions = root / "captions.jsonl"
            output = root / "frozen" / "corpus.txt"
            captions.write_text('{"id":"x","caption":"A road."}\n', encoding="utf-8")
            manifest.write_text('{"id":"x","split":"val"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                build_corpus(root, output)
            manifest.write_text('{"id":"x","split":"train"}\n', encoding="utf-8")
            build_corpus(root, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "A road.\n")
            self.assertEqual(json.loads(output.with_suffix(".txt.json").read_text())[
                "sample_count"], 1)
            self.assertEqual(load_shared_corpus(output, True)[0], b"A road.\n")
            output.write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_shared_corpus(output, True)


if __name__ == "__main__":
    unittest.main()
