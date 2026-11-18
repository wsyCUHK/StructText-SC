import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cityscapes_v2_blip_caption as blip


class FakeProcessor:
    def __init__(self):
        self.seen_pixels = []

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        self.seen_pixels.append(images.getpixel((0, 0)))
        return {"pixel_values": self.seen_pixels[-1]}

    def decode(self, _tokens, *, skip_special_tokens):
        assert skip_special_tokens
        return "a red car on a city street"


class FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 40
        assert kwargs["num_beams"] == 3
        assert kwargs["do_sample"] is False
        return [[1]]


class BlipCaptionTests(unittest.TestCase):
    def make_inputs(self, root: Path):
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type":"blip"}', encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"test fixture")
        run_dir = root / "run"
        run_dir.mkdir()
        original = root / "original.png"
        resized = root / "resized.png"
        Image.new("RGB", (8, 4), "red").save(original)
        Image.new("RGB", (8, 4), "blue").save(resized)
        (run_dir / "manifest.jsonl").write_text(json.dumps({
            "id": "sample_1", "source_original": str(original),
            "source": str(resized), "condition_map": str(root / "unused_map.png"),
        }) + "\n", encoding="utf-8")
        args = argparse.Namespace(run_dir=run_dir, model=model_dir, device="cpu",
                                  max_new_tokens=40, num_beams=3, cpu_threads=4)
        return args

    def test_uses_original_image_only_and_writes_workflow_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.make_inputs(Path(temp))
            processor = FakeProcessor()
            blip.generate_captions(args, processor=processor, model=FakeModel())
            self.assertEqual(processor.seen_pixels, [(255, 0, 0)])
            row = json.loads((args.run_dir / "captions.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["id"], "sample_1")
            self.assertEqual(row["caption"], "a red car on a city street")
            self.assertEqual(row["input"], "original_image_only")
            self.assertEqual((args.run_dir / "sender_text" / "sample_1.txt")
                             .read_text(encoding="utf-8"), row["caption"] + "\n")
            profile = json.loads((args.run_dir / "blip_caption_profile.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(profile["input"], "source_original_only")
            self.assertEqual(profile["sample_count"], 1)
            with self.assertRaises(FileExistsError):
                blip.generate_captions(args, processor=processor, model=FakeModel())

    def test_rejects_non_blip_checkpoint_before_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.make_inputs(Path(temp))
            (args.model / "config.json").write_text(
                '{"model_type":"bert"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Expected a BLIP checkpoint"):
                blip.generate_captions(args, processor=FakeProcessor(), model=FakeModel())

    def test_four_workers_merge_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.make_inputs(Path(temp))
            first = json.loads((args.run_dir / "manifest.jsonl").read_text(encoding="utf-8"))
            manifest = [dict(first, id=f"sample_{index}") for index in range(4)]
            (args.run_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
            self.assertEqual(blip.worker_devices("0,1,2,3", 4, 4),
                             ["0", "1", "2", "3"])
            for index in range(4):
                args.worker_index = index
                args.worker_count = 4
                blip.generate_captions(args, processor=FakeProcessor(), model=FakeModel())
            self.assertFalse((args.run_dir / "captions.jsonl").exists())
            blip.merge_worker_captions(args.run_dir, manifest,
                                       ["0", "1", "2", "3"],
                                       blip.check_model(args.model), args)
            rows = [json.loads(line) for line in
                    (args.run_dir / "captions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in rows],
                             [row["id"] for row in manifest])
            self.assertTrue(all((args.run_dir / "sender_text" / f"{row['id']}.txt").is_file()
                                for row in rows))
            profile = json.loads((args.run_dir / "blip_caption_profile.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(profile["worker_count"], 4)
            self.assertEqual(profile["gpu_devices"], ["0", "1", "2", "3"])

    def test_rejects_invalid_parallel_gpu_assignment(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            blip.worker_devices("0,0,2,3", 4, 4)
        with self.assertRaisesRegex(ValueError, "available"):
            blip.worker_devices("0,1,2,3", 4, 3)


if __name__ == "__main__":
    unittest.main()
