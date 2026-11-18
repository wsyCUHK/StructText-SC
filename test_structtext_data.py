import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_structtext_data import validate_data


def sample(image="Dataset/sample.jpg"):
    return {"messages": [
        {"role": "system", "content": "Describe the image."},
        {"role": "user", "content": "<image>Describe this scene."},
        {"role": "assistant", "content": "A red car on a road."},
    ], "images": [image]}


class StructTextDataTests(unittest.TestCase):
    def test_release_file_has_expected_integrity_and_unique_images(self):
        root = Path(__file__).resolve().parents[1]
        metadata = json.loads((root / "data/provenance.json").read_text(encoding="utf-8"))
        report = validate_data(root / "train_20260428.jsonl",
                               expected_sha256=metadata["sha256"])
        self.assertEqual(report["records"], 1234)
        self.assertEqual(report["unique_images"], 1234)

    def test_valid_data_can_check_local_image_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "train.jsonl"
            data.write_text(json.dumps(sample()) + "\n", encoding="utf-8")
            (root / "Dataset").mkdir()
            (root / "Dataset/sample.jpg").write_bytes(b"reference existence check")
            report = validate_data(data, image_root=root)
            self.assertTrue(report["image_files_checked"])
            self.assertEqual(report["records"], 1)
            (root / "Dataset/sample.jpg").unlink()
            with self.assertRaisesRegex(ValueError, "Missing 1"):
                validate_data(data, image_root=root)

    def test_duplicate_images_are_rejected_even_with_different_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = sample(), sample()
            second["messages"][2]["content"] = "A different target."
            data = Path(directory) / "train.jsonl"
            data.write_text("\n".join(json.dumps(row) for row in [first, second]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate image"):
                validate_data(data)

    def test_unsafe_paths_and_invalid_schema_are_rejected(self):
        cases = [sample("../outside.jpg"), sample("C:/private.jpg"), sample("/outside.jpg"),
                 sample("Dataset\\sample.jpg"), sample()]
        cases[-1]["messages"][2]["role"] = "user"
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            for row in cases:
                with self.subTest(row=row):
                    data.write_text(json.dumps(row), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate_data(data)

    def test_changed_release_bytes_fail_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "train.jsonl"
            data.write_text(json.dumps(sample()), encoding="utf-8")
            original_hash = hashlib.sha256(data.read_bytes()).hexdigest()
            data.write_bytes(data.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_data(data, expected_sha256=original_hash)


if __name__ == "__main__":
    unittest.main()
