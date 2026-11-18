"""Check the BF16 guard without loading multi-gigabyte model files."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cityscapes_v2_weight_precision import verify_bf16_weights  # noqa: E402


def fake_safetensors(path: Path, dtype: str) -> None:
    header = json.dumps({"weight": {"dtype": dtype, "shape": [2],
                                    "data_offsets": [0, 4]}}).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)


class WeightPrecisionTests(unittest.TestCase):
    def test_all_three_bf16_components_pass(self) -> None:
        with TemporaryDirectory() as root:
            files = [Path(root) / f"{name}.safetensors"
                     for name in ("diffusion", "text_encoder", "vae")]
            for path in files:
                fake_safetensors(path, "BF16")
            report = verify_bf16_weights(*files)
            self.assertEqual([report[name]["elements_by_dtype"] for name in
                              ("diffusion_model", "llm", "vae")], [{"BF16": 2}] * 3)

    def test_int8_text_encoder_is_rejected(self) -> None:
        with TemporaryDirectory() as root:
            files = [Path(root) / f"{name}.safetensors"
                     for name in ("diffusion", "text_encoder", "vae")]
            for path in files:
                fake_safetensors(path, "BF16")
            fake_safetensors(files[1], "I8")
            with self.assertRaisesRegex(ValueError, "llm is not BF16-aligned"):
                verify_bf16_weights(*files)

    def test_gguf_diffusion_is_rejected(self) -> None:
        with TemporaryDirectory() as root:
            files = [Path(root) / f"{name}.safetensors"
                     for name in ("diffusion", "text_encoder", "vae")]
            for path in files:
                fake_safetensors(path, "BF16")
            gguf = Path(root) / "diffusion-Q8_0.gguf"
            gguf.write_bytes(b"GGUF")
            with self.assertRaisesRegex(ValueError, "requires safetensors"):
                verify_bf16_weights(gguf, files[1], files[2])


if __name__ == "__main__":
    unittest.main()
