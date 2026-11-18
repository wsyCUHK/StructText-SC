import csv
import errno
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from organize_cityscapes_v2_outputs import link_or_copy, prepare_view


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class CityscapesV2OutputViewTests(unittest.TestCase):
    def test_permission_denied_hardlink_falls_back_to_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root / "source.txt", root / "view/copy.txt"
            source.write_bytes(b"unchanged result bytes")
            with patch("organize_cityscapes_v2_outputs.os.link",
                       side_effect=PermissionError(errno.EACCES, "hard links unavailable")):
                self.assertEqual(link_or_copy(source, destination), "copy")
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertTrue(source.is_file())

    def make_run(self, root: Path) -> tuple[Path, Path, Path]:
        data = root / "data"
        sample = data / "samples" / "x"
        sample.mkdir(parents=True)
        transport = data / "transport_packet_swinjscc"
        (transport / "maps").mkdir(parents=True)
        (transport / "maps_raw").mkdir()
        (transport / "receiver_text").mkdir()
        generated = data / "generated_packet_swinjscc" / "x"
        generated.mkdir(parents=True)
        (data / "sender_text").mkdir()
        for path in (sample / "source.png", sample / "condition_map.png",
                     sample / "label_ids.png", transport / "maps" / "x.png",
                     transport / "maps_raw" / "x.png",
                     generated / "fusion.png", generated / "comparison.png"):
            Image.new("RGB", (8, 4), "blue").save(path)
        (data / "sender_text" / "x.txt").write_text("A car.\n", encoding="utf-8")
        (transport / "receiver_text" / "x.txt").write_text("A car.\n", encoding="utf-8")
        (transport / "packet_summary.json").write_text("{}", encoding="utf-8")
        (transport / "map_summary.json").write_text("{}", encoding="utf-8")
        write_jsonl(data / "manifest.jsonl", [{
            "id": "x", "source": str(sample / "source.png"),
            "condition_map": str(sample / "condition_map.png"),
            "label_ids": str(sample / "label_ids.png"),
        }])
        write_jsonl(data / "captions.jsonl", [{"id": "x", "caption": "A car."}])
        write_jsonl(transport / "receiver.jsonl", [{
            "id": "x", "caption": "A car.",
            "caption_path": str(transport / "receiver_text" / "x.txt"),
            "condition_map": str(transport / "maps" / "x.png"),
        }])
        write_jsonl(generated.parent / "results.jsonl", [
            {"id": "x", "branch": branch, "path": str(generated / f"{branch}.png"),
             "condition_map": str(transport / "maps" / "x.png"),
             "prompt": "Shared segmentation-to-photo prompt" +
             (" A car." if branch == "fusion" else "")}
            for branch in ("fusion",)
        ])
        return data, transport, generated.parent

    def test_view_groups_images_text_and_metrics_without_moving_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, transport, generated = self.make_run(root)
            view = prepare_view(root, data, transport, generated)
            self.assertTrue((view / "01_source/rgb/x.png").is_file())
            self.assertTrue((view / "01_source/gt_color_mask/x.png").is_file())
            self.assertTrue((view / "02_sender_text/captions.jsonl").is_file())
            self.assertEqual(json.loads((view / "03_text_channel/recovered_captions.jsonl")
                                        .read_text(encoding="utf-8")),
                             {"id": "x", "caption": "A car."})
            self.assertTrue((view / "04_mask_channel/received/x.png").is_file())
            self.assertTrue((view / "05_generation/comparison/x.png").is_file())
            self.assertTrue((generated / "x" / "fusion.png").is_file())
            with self.assertRaises(FileExistsError):
                prepare_view(root, data, transport, generated)

    def test_view_rejects_generated_image_with_wrong_receiver_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, transport, generated = self.make_run(root)
            rows = [json.loads(line) for line in (generated / "results.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            rows[0]["condition_map"] = str(data / "samples" / "x" / "condition_map.png")
            write_jsonl(generated / "results.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "different map"):
                prepare_view(root, data, transport, generated)
            self.assertFalse((root / "artifacts").exists())

    def test_view_identifies_drn_sender_map_separately_from_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data, transport, generated = self.make_run(root)
            sender_ids = data / "sender_drn_train_ids" / "x.png"
            sender_ids.parent.mkdir()
            Image.new("L", (8, 4), 13).save(sender_ids)
            rows = [json.loads(line) for line in (data / "manifest.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
            rows[0].update(mask_source="DRN-D-105 predicted",
                           sender_train_ids=str(sender_ids))
            write_jsonl(data / "manifest.jsonl", rows)
            (data / "sender_drn_profile.json").write_text("{}", encoding="utf-8")
            view = prepare_view(root, data, transport, generated)
            self.assertTrue((view / "01_source/drn_color_mask/x.png").is_file())
            self.assertTrue((view / "01_source/drn_train_ids/x.png").is_file())
            self.assertTrue((view / "01_source/reference_gt_label_ids/x.png").is_file())
            self.assertFalse((view / "01_source/gt_color_mask/x.png").exists())

if __name__ == "__main__":
    unittest.main()
