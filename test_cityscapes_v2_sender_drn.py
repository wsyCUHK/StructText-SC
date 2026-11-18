import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cityscapes_gt_smoke as prepare_module
import cityscapes_v2_sender_drn as sender
from swinjscc_common import PALETTE


class SenderDRNTests(unittest.TestCase):
    def make_dataset(self, root: Path, split: str) -> None:
        city = "aachen" if split == "val" else "berlin"
        sample_id = f"{city}_000000_000001"
        image_dir = root / "leftImg8bit" / split / city
        image_dir.mkdir(parents=True)
        Image.new("RGB", (8, 4), (20, 30, 40)).save(
            image_dir / f"{sample_id}_leftImg8bit.png")
        if split == "val":
            gt_dir = root / "gtFine" / split / city
            gt_dir.mkdir(parents=True)
            Image.new("L", (8, 4), 26).save(gt_dir / f"{sample_id}_gtFine_labelIds.png")
            Image.new("RGB", (8, 4), (0, 0, 142)).save(
                gt_dir / f"{sample_id}_gtFine_color.png")

    def prepare(self, root: Path, split: str, mask_source: str) -> tuple[Path, dict]:
        run = root.parent / f"run_{split}_{mask_source}"
        prepare_module.prepare(argparse.Namespace(
            dataset_root=root, split=split, limit=1, width=8, height=4,
            run_dir=run, mask_source=mask_source,
        ))
        row = json.loads((run / "manifest.jsonl").read_text(encoding="utf-8"))
        return run, row

    def test_gt_default_remains_gt_and_drn_uses_predicted_palette(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_dataset(root, "val")
            gt_run, gt = self.prepare(root, "val", "gt")
            self.assertEqual(gt["mask_source"], "Cityscapes gtFine ground truth")
            self.assertTrue(Path(gt["condition_map"]).is_file())
            self.assertNotIn("sender_train_ids", gt)

            drn_run, drn = self.prepare(root, "val", "drn")
            self.assertFalse(Path(drn["condition_map"]).exists())
            self.assertTrue(Path(drn["label_ids"]).exists())
            calls = []

            def predict(path):
                calls.append(path)
                return np.full((4, 8), 13, dtype=np.uint8)

            self.assertEqual(sender.materialize_sender_maps(drn_run, predict), 1)
            self.assertEqual(calls, [Path(drn["source"])])
            with Image.open(drn["sender_train_ids"]) as image:
                self.assertEqual(image.mode, "L")
                self.assertEqual(image.getpixel((0, 0)), 13)
            with Image.open(drn["condition_map"]) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.getpixel((0, 0)), tuple(PALETTE[13]))
            self.assertEqual(drn["mask_source"], "DRN-D-105 predicted")
            with self.assertRaisesRegex(ValueError, "existing DRN sender output"):
                sender.materialize_sender_maps(drn_run, predict)

    def test_official_test_needs_predicted_mask_and_no_gt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_dataset(root, "test")
            with self.assertRaises(ValueError):
                prepare_module.cityscapes_pairs(root, "test", 1, "gt")
            run, row = self.prepare(root, "test", "drn")
            self.assertNotIn("label_ids", row)
            self.assertNotIn("label_original", row)
            sender.materialize_sender_maps(
                run, lambda _: np.full((4, 8), 13, dtype=np.uint8))
            self.assertTrue(Path(row["condition_map"]).is_file())

if __name__ == "__main__":
    unittest.main()
