import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cityscapes_gt_smoke as smoke


class CityscapesGTSmokeTests(unittest.TestCase):
    def make_sample(self, root: Path, city: str, frame: str) -> None:
        image_dir = root / "leftImg8bit" / "val" / city
        gt_dir = root / "gtFine" / "val" / city
        image_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 4), (100, 120, 140)).save(
            image_dir / f"{frame}_leftImg8bit.png"
        )
        labels = Image.new("L", (8, 4), 7)
        for x in range(4, 8):
            for y in range(4):
                labels.putpixel((x, y), 26)
        labels.save(gt_dir / f"{frame}_gtFine_labelIds.png")
        colors = Image.new("RGB", (8, 4), (128, 64, 128))
        for x in range(4, 8):
            for y in range(4):
                colors.putpixel((x, y), (0, 0, 142))
        colors.save(gt_dir / f"{frame}_gtFine_color.png")

    def test_prepare_pairs_cities_and_preserves_discrete_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_sample(root, "aachen", "aachen_000000_000001")
            self.make_sample(root, "bochum", "bochum_000000_000001")
            run_dir = Path(temp) / "run"
            smoke.prepare(
                argparse.Namespace(
                    dataset_root=root,
                    split="val",
                    limit=2,
                    width=128,
                    height=64,
                    run_dir=run_dir,
                )
            )
            rows = smoke.read_jsonl(run_dir / "manifest.jsonl")
            self.assertEqual({row["city"] for row in rows}, {"aachen", "bochum"})
            for row in rows:
                with Image.open(row["label_ids"]) as labels:
                    self.assertEqual(labels.size, (128, 64))
                    self.assertEqual(set(labels.tobytes()), {7, 26})
                    self.assertEqual(labels.getpixel((0, 0)), 7)
                    self.assertEqual(labels.getpixel((127, 0)), 26)
                messages = smoke.caption_messages(row)
                content = messages[1]["content"]
                self.assertEqual([item["type"] for item in content], [
                    "image_url", "text"
                ])
                self.assertFalse(content[0]["image_url"]["url"].startswith("data:"))
                data_url_messages = smoke.caption_messages(row, "data_url")
                data_url_images = [item["image_url"]["url"] for item in
                                   data_url_messages[1]["content"]
                                   if item["type"] == "image_url"]
                self.assertEqual(len(data_url_images), 1)
                self.assertTrue(all(url.startswith("data:image/png;base64,")
                                    for url in data_url_images))
                gemma_messages = smoke.caption_messages(
                    row, "data_url", model_family="gemma4")
                gemma_content = gemma_messages[1]["content"]
                self.assertEqual([item["type"] for item in gemma_content],
                                 ["image_url", "text"])
                self.assertIn("Original Cityscapes", gemma_content[-1]["text"])
                self.assertNotIn("Image 2", str(gemma_messages))
                self.assertEqual(gemma_messages[0]["role"], "system")
            with self.assertRaises(FileExistsError):
                smoke.prepare(
                    argparse.Namespace(
                        dataset_root=root,
                        split="val",
                        limit=2,
                        width=128,
                        height=64,
                        run_dir=run_dir,
                    )
                )

    def test_prepare_native_size_keeps_original_and_rejects_resize(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_sample(root, "aachen", "aachen_000000_000001")
            run_dir = Path(temp) / "native"
            args = argparse.Namespace(dataset_root=root, split="val", limit=1,
                                      width=8, height=4, run_dir=run_dir,
                                      mask_source="gt", require_native_size=True)
            smoke.prepare(args)
            row = smoke.read_jsonl(run_dir / "manifest.jsonl")[0]
            self.assertEqual(row["original_size"], row["model_size"])
            with Image.open(row["source"]) as image:
                self.assertEqual(image.size, (8, 4))

            args.run_dir = Path(temp) / "mismatch"
            args.width = 4
            with self.assertRaisesRegex(ValueError, "refusing to resize"):
                smoke.prepare(args)

    def test_missing_gt_and_test_split_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_sample(root, "aachen", "aachen_000000_000001")
            color = next((root / "gtFine").rglob("*_color.png"))
            color.unlink()
            with self.assertRaises(FileNotFoundError):
                smoke.cityscapes_pairs(root, "val", 1)
            with self.assertRaises(ValueError):
                smoke.cityscapes_pairs(root, "test", 1)

    def test_generation_prompt_does_not_include_source_image(self) -> None:
        self.assertIn("Additional factual scene description: red car", smoke.generation_prompt("red car"))
        self.assertEqual(smoke.generation_prompt("red car"),
                         smoke.GENERATION_PREFIX + "\nAdditional factual scene description: red car")
        self.assertNotIn("red car", smoke.generation_prompt(None))
        self.assertIn("semantic segmentation map", smoke.generation_prompt(None))
        self.assertIn("photorealistic", smoke.generation_prompt(None))
        self.assertIn("Do not output a colored segmentation map", smoke.generation_prompt(None))
        self.assertNotIn("recovered after packet loss", smoke.generation_prompt("red car"))
        self.assertNotIn("recovered after packet loss", smoke.generation_prompt(None))
        masked = smoke.generation_prompt("red [MASK] car")
        self.assertIn("recovered after packet loss", masked)
        self.assertIn("[MASK] is not an object", masked)
        self.assertIn("Additional factual scene description: red [MASK] car", masked)
        self.assertNotIn("recovered after packet loss",
                         smoke.generation_prompt("red [MASK] car", "off"))
        with self.assertRaises(ValueError):
            smoke.generation_prompt("red car", "invalid")

    def test_source_only_never_reads_a_map_for_either_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.png"
            Image.new("RGB", (8, 4), "white").save(source)
            # Intentionally omit condition_map: the caption stage must not need it.
            record = {"source": str(source)}
            for family in ("qwen3_vl", "gemma4"):
                with self.subTest(family=family):
                    with patch.object(smoke, "image_path_to_api_url",
                                      wraps=smoke.image_path_to_api_url) as encode:
                        messages = smoke.caption_messages(record, "data_url",
                                                          model_family=family)
                    self.assertEqual(encode.call_count, 1)
                    self.assertEqual(encode.call_args.args[0], str(source))
                    parts = messages[1]["content"]
                    self.assertEqual([part["type"] for part in parts], ["image_url", "text"])
                    text = json.dumps(messages)
                    self.assertNotIn("Image 2", text)
                    self.assertNotIn("Legend:", text)

    def test_explicit_map_mode_sends_two_images_for_either_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source, condition = Path(temp) / "source.png", Path(temp) / "map.png"
            Image.new("RGB", (8, 4), "white").save(source)
            Image.new("RGB", (8, 4), "blue").save(condition)
            for family in ("qwen3_vl", "gemma4"):
                with self.subTest(family=family):
                    messages = smoke.caption_messages(
                        {"source": str(source), "condition_map": str(condition)},
                        "data_url", model_family=family, input_mode="source_and_map")
                    self.assertEqual(sum(part["type"] == "image_url"
                                         for part in messages[1]["content"]), 2)
                    self.assertIn("Image 2: aligned semantic segmentation map", str(messages))
                    self.assertIn(smoke.CLASS_LEGEND, str(messages))

    def test_prompt_input_contract_and_legacy_custom_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prompt.json"
            prompt = {"name": "custom", "system_prompt": "Caption the photograph.",
                      "user_instruction": "Describe visible objects."}
            path.write_text(json.dumps(prompt), encoding="utf-8")
            self.assertEqual(smoke.load_caption_prompt(path)["input_mode"], "source_only")
            self.assertEqual(smoke.load_caption_prompt(path, "source_and_map")["input_mode"],
                             "source_and_map")
            prompt["input_mode"] = "source_and_map"
            path.write_text(json.dumps(prompt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "conflicts"):
                smoke.load_caption_prompt(path)
            self.assertEqual(smoke.load_caption_prompt(path, "source_and_map")["input_mode"],
                             "source_and_map")
            for wording in ("Use Image 2.", "Use the second image.", "Use Image two."):
                prompt.pop("input_mode", None)
                prompt["user_instruction"] = wording
                path.write_text(json.dumps(prompt), encoding="utf-8")
                with self.subTest(wording=wording):
                    with self.assertRaisesRegex(ValueError, "second image"):
                        smoke.load_caption_prompt(path)
            with self.assertRaisesRegex(ValueError, "Unsupported caption input mode"):
                smoke.load_caption_prompt(None, "unknown")

    def test_shipped_prompts_declare_source_only_inputs(self) -> None:
        prompt_paths = sorted((Path(__file__).resolve().parents[1] / "prompt").glob("*.json"))
        self.assertEqual(len(prompt_paths), 6)
        for path in prompt_paths:
            with self.subTest(path=path.name):
                prompt = smoke.load_caption_prompt(path)
                self.assertEqual(prompt["input_mode"], "source_only")
                self.assertEqual(len(prompt["sha256"]), 64)
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    smoke.load_caption_prompt(path, "source_and_map")

    def test_source_only_caption_writes_input_provenance_without_a_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            source = run_dir / "source.png"
            Image.new("RGB", (8, 4), "white").save(source)
            smoke.save_jsonl_exclusive(run_dir / "manifest.jsonl",
                                       [{"id": "sample", "source": str(source)}])
            with patch.object(smoke, "request_openai_compatible", return_value="A road.") as request:
                with redirect_stdout(io.StringIO()):
                    smoke.caption(argparse.Namespace(
                        run_dir=run_dir, llm_api="local", api_model_name="qwen-model",
                        qwen_image_format="data_url", max_new_tokens=256,
                        temperature=0.0, request_timeout=300, max_retries=0,
                        api_extra_body_json="{}"))
            parts = request.call_args.kwargs["messages"][1]["content"]
            self.assertEqual(sum(part["type"] == "image_url" for part in parts), 1)
            row = smoke.read_jsonl(run_dir / "captions.jsonl")[0]
            self.assertEqual(row["caption_input_mode"], "source_only")
            self.assertEqual(row["prompt_name"], "default_source_only")

    def test_caption_calls_sglang_openai_api_with_two_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_sample(root, "aachen", "aachen_000000_000001")
            run_dir = Path(temp) / "run"
            smoke.prepare(
                argparse.Namespace(
                    dataset_root=root,
                    split="val",
                    limit=1,
                    width=128,
                    height=64,
                    run_dir=run_dir,
                )
            )
            response = Mock()
            response.raise_for_status.return_value = None
            response.json.return_value = {
                "choices": [
                    {"message": {"content": [{"type": "text", "text": "A road and a car."}]}}
                ]
            }
            terminal = io.StringIO()
            with (
                patch("requests.post", return_value=response) as post,
                redirect_stdout(terminal),
            ):
                smoke.caption(
                    argparse.Namespace(
                        run_dir=run_dir,
                        llm_api="http://127.0.0.1:30000/v1/chat/completions",
                        api_model_name="/models/qwen-32b-instruct",
                        qwen_image_format="legacy_base64",
                        max_new_tokens=256,
                        temperature=0.0,
                        request_timeout=300,
                        max_retries=0,
                        api_extra_body_json="{}",
                        caption_input_mode="source_and_map",
                    )
                )

            payload = post.call_args.kwargs["json"]
            self.assertEqual(payload["model"], "/models/qwen-32b-instruct")
            image_parts = [
                item
                for item in payload["messages"][1]["content"]
                if item["type"] == "image_url"
            ]
            self.assertEqual(len(image_parts), 2)
            rows = smoke.read_jsonl(run_dir / "captions.jsonl")
            self.assertEqual(rows[0]["caption"], "A road and a car.")
            self.assertEqual(rows[0]["backend"], "sglang_openai_compatible_api")
            self.assertEqual(rows[0]["caption_input_mode"], "source_and_map")
            self.assertIn("Processing progress: 1/1 (100.0%)", terminal.getvalue())
            self.assertNotIn("A road and a car.", terminal.getvalue())

    def test_prompt_file_changes_messages_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_sample(root, "aachen", "aachen_000000_000001")
            run_dir = Path(temp) / "run"
            smoke.prepare(argparse.Namespace(dataset_root=root, split="val", limit=1,
                                             width=128, height=64, run_dir=run_dir))
            prompt_path = Path(temp) / "prompt.json"
            prompt_path.write_text(json.dumps({"name": "trial", "system_prompt": "System trial",
                                               "user_instruction": "User trial"}), encoding="utf-8")
            with patch.object(smoke, "request_openai_compatible", return_value="Visible car.") as request:
                with redirect_stdout(io.StringIO()):
                    smoke.caption(argparse.Namespace(
                        run_dir=run_dir, llm_api="local", api_model_name="Qwen3-VL-32B",
                        qwen_image_format="legacy_base64", max_new_tokens=256,
                        temperature=0.0, request_timeout=300, max_retries=0,
                        api_extra_body_json="{}", prompt_file=prompt_path))
            messages = request.call_args.kwargs["messages"]
            self.assertEqual(messages[0]["content"], "System trial")
            self.assertTrue(messages[1]["content"][-1]["text"].endswith("User trial"))
            self.assertEqual(sum(x["type"] == "image_url" for x in messages[1]["content"]), 1)
            row = smoke.read_jsonl(run_dir / "captions.jsonl")[0]
            self.assertEqual(row["prompt_name"], "trial")
            self.assertEqual(row["prompt_sha256"], hashlib.sha256(prompt_path.read_bytes()).hexdigest())
            self.assertEqual(row["caption_input_mode"], "source_only")

            prompt_path.write_text('{"name":"trial","system_prompt":"System trial"}',
                                   encoding="utf-8")
            with self.assertRaises(ValueError):
                smoke.load_caption_prompt(prompt_path)

    def test_gemma4_caption_uses_two_data_url_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "dataset"
            self.make_sample(root, "aachen", "aachen_000000_000001")
            run_dir = Path(temp) / "run"
            smoke.prepare(argparse.Namespace(dataset_root=root, split="val", limit=1,
                                             width=128, height=64, run_dir=run_dir))
            args = argparse.Namespace(
                run_dir=run_dir, llm_api="local", api_model_name="/models/gemma-4-31B-it",
                i2t_model_family="gemma4", qwen_image_format="legacy_base64",
                max_new_tokens=256, temperature=0.0, request_timeout=300,
                max_retries=0, api_extra_body_json="{}", prompt_file=None,
                caption_input_mode="source_and_map")
            with self.assertRaisesRegex(ValueError, "data_url"):
                smoke.caption(args)
            args.qwen_image_format = "data_url"
            with patch.object(smoke, "request_openai_compatible", return_value="A car on a road.") as request:
                with redirect_stdout(io.StringIO()):
                    smoke.caption(args)
            content = request.call_args.kwargs["messages"][1]["content"]
            self.assertEqual([part["type"] for part in content],
                             ["image_url", "image_url", "text"])
            self.assertTrue(all(part["image_url"]["url"].startswith("data:image/png;base64,")
                                for part in content[:2]))
            row = smoke.read_jsonl(run_dir / "captions.jsonl")[0]
            self.assertEqual(row["model_family"], "gemma4")

    def test_caption_http_error_omits_response_body(self) -> None:
        response = Mock(status_code=500, text="private caption or prompt")
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        with patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500") as caught:
                smoke.request_openai_compatible(
                    llm_api="http://127.0.0.1:30000/v1/chat/completions",
                    api_model_name="model",
                    messages=[],
                    max_tokens=1,
                    temperature=0.0,
                    timeout_seconds=1,
                    max_retries=0,
                )
        self.assertNotIn("private caption or prompt", str(caught.exception))


@unittest.skipIf(os.name == "nt", "Workflow shell integration requires the supported Linux platform")
class CaptionShellIntegrationTests(unittest.TestCase):
    """Exercise actual shell argument/config propagation with model services stubbed."""

    @classmethod
    def setUpClass(cls) -> None:
        git_bash = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        cls.bash = str(git_bash) if os.name == "nt" and git_bash.is_file() else shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("Bash is required for shell integration checks")
        cls.repo = Path(__file__).resolve().parents[1]

    def shell_path(self, path: Path) -> str:
        if os.name != "nt":
            return str(path)
        result = subprocess.run(
            [self.bash, "--noprofile", "--norc", "-c", 'cygpath -u "$1"', "_", str(path)],
            capture_output=True, text=True, check=True)
        return result.stdout.strip()

    def write_executable(self, path: Path, text: str) -> None:
        path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + text,
                        encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def base_env(self, bin_dir: Path) -> dict:
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join((str(bin_dir), str(Path(self.bash).parent), env["PATH"]))
        # Prevent a developer's configured model/env from entering these stubbed checks.
        for name in ("I2T_ENV", "BLIP_ENV", "DRN_ENV", "PACKET_ENV", "SWINJSCC_ENV",
                     "IMAGE_ENV", "CONDA_SH", "I2T_PROMPT_FILE", "SERVER_EXTRA_ARGS",
                     "DOCKER_IMAGE", "QWEN_IMAGE_FORMAT"):
            env.pop(name, None)
        return env

    def test_sglang_wrapper_forwards_default_and_explicit_input_modes(self) -> None:
        for mode in (None, "source_and_map"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                bin_dir, run_dir, model_dir = root / "bin", root / "run", root / "model"
                for directory in (bin_dir, run_dir, model_dir, root / "scripts"):
                    directory.mkdir()
                (root / "scripts/cityscapes_gt_smoke.py").write_text("", encoding="utf-8")
                (run_dir / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
                args_path, service_name, docker_state = root / "args.txt", root / "service", root / "docker"
                self.write_executable(bin_dir / "python", '''
if [[ "${1:-}" == -u ]]; then printf '%s\\n' "$@" > "$CAPTION_ARGS_FILE"; fi
''')
                self.write_executable(bin_dir / "docker", '''
case "$1" in
  run) touch "$DOCKER_STATE"; printf 'stub-container-id\\n' ;;
  ps) if [[ -f "$DOCKER_STATE" ]]; then cat "$SERVICE_NAME_FILE"; fi ;;
  rm) rm -f "$DOCKER_STATE" ;;
  stop|logs) ;;
  *) exit 9 ;;
esac
''')
                self.write_executable(bin_dir / "curl", "exit 0\n")
                env = self.base_env(bin_dir)
                env.update(ROOT_DIR=self.shell_path(root), MODEL_PATH=self.shell_path(model_dir),
                           I2T_MODEL_FAMILY="qwen3_vl", MODEL_PORT="65423",
                           CAPTION_ARGS_FILE=self.shell_path(args_path),
                           SERVICE_NAME_FILE=self.shell_path(service_name),
                           DOCKER_STATE=self.shell_path(docker_state))
                env.pop("I2T_CAPTION_INPUT_MODE", None)
                if mode:
                    env["I2T_CAPTION_INPUT_MODE"] = mode
                result = subprocess.run(
                    [self.bash, "--noprofile", "--norc",
                     self.shell_path(self.repo / "scripts/run_cityscapes_gt_caption_sglang.sh"),
                     self.shell_path(run_dir)], env=env, capture_output=True,
                    encoding="utf-8", errors="replace", timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                args = args_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(args[args.index("--caption-input-mode") + 1], mode or "source_only")
                self.assertFalse(docker_state.exists(), "temporary server was not cleaned up")

    def test_workflow_exports_and_records_input_mode(self) -> None:
        for mode in (None, "source_and_map"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                bin_dir, scripts, model_dir = root / "bin", root / "scripts", root / "model"
                dataset = root / "dataset"
                for directory in (bin_dir, scripts, model_dir, dataset / "leftImg8bit/val"):
                    directory.mkdir(parents=True)
                codebook = root / "codebook.txt"
                codebook.write_text("training captions\n", encoding="utf-8")
                Path(str(codebook) + ".json").write_text("{}\n", encoding="utf-8")
                self.write_executable(bin_dir / "python", "exit 0\n")
                self.write_executable(scripts / "run_cityscapes_gt_caption_sglang.sh",
                                      'printf "%s" "$I2T_CAPTION_INPUT_MODE" > "$CAPTION_MODE_FILE"\n')
                run_dir, mode_path = root / "run", root / "mode.txt"
                env = self.base_env(bin_dir)
                env.update(ROOT_DIR=self.shell_path(root), DATASET_ROOT=self.shell_path(dataset),
                           RUN_DIR=self.shell_path(run_dir), MODEL_PATH=self.shell_path(model_dir),
                           IMAGE_MODEL=self.shell_path(model_dir), CODEBOOK_CORPUS=self.shell_path(codebook),
                           I2T_MODEL_FAMILY="qwen3_vl", MASK_SOURCE="gt", SPLIT="val",
                           CHANNEL_MODE="oracle", IMAGE_BACKEND="diffusers",
                           CAPTION_MODE_FILE=self.shell_path(mode_path))
                env.pop("I2T_CAPTION_INPUT_MODE", None)
                if mode:
                    env["I2T_CAPTION_INPUT_MODE"] = mode
                result = subprocess.run(
                    [self.bash, "--noprofile", "--norc",
                     self.shell_path(self.repo / "work_flow_GenSemComv2.sh")],
                    env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                expected = mode or "source_only"
                self.assertEqual(mode_path.read_text(encoding="utf-8"), expected)
                config = (run_dir / "config/run_config.env").read_text(encoding="utf-8")
                self.assertIn(f"I2T_CAPTION_INPUT_MODE={expected}\n", config)


if __name__ == "__main__":
    unittest.main()
