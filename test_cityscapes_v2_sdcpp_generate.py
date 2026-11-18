"""Contract checks for the sd.cpp image-conditioned stage without a GPU."""

from __future__ import annotations

import base64
import argparse
import io
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cityscapes_v2_sdcpp_generate as sdcpp  # noqa: E402


class SdCppContractTests(unittest.TestCase):
    def test_parallel_workers_use_distinct_gpus_and_ports(self) -> None:
        self.assertEqual(sdcpp.worker_devices("0,1", 2, 2), ["0", "1"])
        self.assertEqual(sdcpp.worker_devices("0,1", 2, 1), ["0"])
        with self.assertRaises(ValueError):
            sdcpp.worker_devices("0,0", 2, 2)
        args = argparse.Namespace(
            run_dir=Path("/run"), output_dir=Path("/generated"),
            server_bin=Path("/bin/sd-server"), diffusion_model=Path("/dit"),
            vae=Path("/vae"), llm=Path("/llm"), llm_vision=None,
            precision_profile="bf16", backend_spec="diffusion=cuda0,te=cuda0,vae=cuda0",
            auto_fit="on", seed=42, steps=40, cfg_scale=1.0,
            startup_timeout=900, capabilities_timeout=30, job_timeout=1200,
            fusion_mask_prompt_policy="conditional", receiver_jsonl=Path("/receiver"),
            max_vram="cuda0=46", diffusion_fa=True,
            verbose=True, offload_to_cpu=False,
        )
        command = sdcpp.worker_command(args, 1, 2, "1", 12346)
        self.assertEqual(command[command.index("--gpu-device") + 1], "1")
        self.assertEqual(command[command.index("--port") + 1], "12346")
        self.assertEqual(command[command.index("--worker-index") + 1], "1")
        self.assertEqual(command[command.index("--capabilities-timeout") + 1], "30")
        self.assertIn("--receiver-jsonl", command)
        self.assertIn("--diffusion-fa", command)

    def test_health_check_allows_a_slow_capabilities_response(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        capabilities = {
            "model": {"name": "Qwen-Image-2.1"},
            "supported_modes": ["img_gen"],
            "features_by_mode": {"img_gen": {"ref_images": True}},
        }
        with patch.object(sdcpp, "http_json", return_value=capabilities) as request:
            result = sdcpp.wait_for_server("http://127.0.0.1:12345", process,
                                           timeout=1800, capabilities_timeout=30)
        self.assertEqual(result, capabilities)
        self.assertEqual(request.call_args.kwargs["timeout"], 30)

    def test_merge_parallel_outputs_preserves_manifest_order(self) -> None:
        with TemporaryDirectory() as root:
            output = Path(root)
            manifest = [{"id": name, "model_size": [64, 32]}
                        for name in ("second", "first")]
            for worker, record in enumerate(manifest):
                sample_dir = output / record["id"]
                sample_dir.mkdir()
                rows = []
                for branch in ("fusion",):
                    image_path = sample_dir / f"{branch}.png"
                    Image.new("RGB", (64, 32), "green").save(image_path)
                    rows.append({"id": record["id"], "branch": branch,
                                 "path": str(image_path.resolve()),
                                 "worker_index": worker})
                (sample_dir / "comparison.png").touch()
                (output / f"worker_{worker}_results.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            sdcpp.merge_worker_results(output, manifest, 2)
            rows = sdcpp.read_jsonl(output / "results.jsonl")
            self.assertEqual([(row["id"], row["branch"]) for row in rows], [
                ("second", "fusion"), ("first", "fusion")])

    def test_parallel_fusion_starts_two_services_and_merges_outputs(self) -> None:
        with TemporaryDirectory() as root:
            output = Path(root) / "generated"
            manifest = [{"id": name, "model_size": [64, 32]}
                        for name in ("a", "b")]
            args = argparse.Namespace(
                run_dir=Path(root), output_dir=output,
                server_bin=Path("/bin/sd-server"), diffusion_model=Path("/dit"),
                vae=Path("/vae"), llm=Path("/llm"), llm_vision=None,
                precision_profile="bf16", backend_spec="diffusion=cuda0,te=cuda0,vae=cuda0",
                auto_fit="on", seed=42, steps=40, cfg_scale=1.0,
                startup_timeout=900, job_timeout=1200,
                fusion_mask_prompt_policy="conditional", receiver_jsonl=Path("/receiver"),
                max_vram="cuda0=46", diffusion_fa=True,
                verbose=True, offload_to_cpu=False, gpu_device="0",
                gpu_devices="0,1", workers=2, worker_index=None, port=12345,
            )
            commands = []

            class FinishedProcess:
                def poll(self):
                    return 0

            def fake_popen(command, **_kwargs):
                commands.append(command)
                index = int(command[command.index("--worker-index") + 1])
                record = manifest[index]
                sample_dir = output / record["id"]
                sample_dir.mkdir()
                image_path = sample_dir / "fusion.png"
                Image.new("RGB", (64, 32), "blue").save(image_path)
                row = {"id": record["id"], "branch": "fusion",
                       "path": str(image_path.resolve()), "worker_index": index}
                (sample_dir / "comparison.png").touch()
                (output / f"worker_{index}_results.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8")
                (output / f"worker_{index}_sdcpp_profile.json").write_text(
                    json.dumps({"command": command, "precision_profile": "bf16"}),
                    encoding="utf-8")
                return FinishedProcess()

            with patch.object(sdcpp, "validate_inputs", return_value=(manifest, {})), \
                    patch.object(sdcpp, "check_port_available") as check_port, \
                    patch.object(sdcpp.subprocess, "Popen", side_effect=fake_popen):
                sdcpp.generate_parallel(args)
            self.assertEqual(check_port.call_count, 2)
            self.assertEqual([command[command.index("--port") + 1]
                              for command in commands], ["12345", "12346"])
            self.assertTrue(all("--receiver-jsonl" in command for command in commands))
            rows = sdcpp.read_jsonl(output / "results.jsonl")
            self.assertEqual([row["id"] for row in rows], ["a", "b"])
            profile = json.loads((output / "sdcpp_profile.json").read_text())
            self.assertEqual(profile["gpu_devices"], ["0", "1"])
            self.assertEqual(profile["worker_count"], 2)

    def test_server_job_timestamps_are_seconds(self) -> None:
        self.assertEqual(sdcpp.server_job_duration_seconds({
            "started": 1_700_000_000, "completed": 1_700_000_027,
        }), 27.0)
        self.assertIsNone(sdcpp.server_job_duration_seconds({
            "started": 10, "completed": 9,
        }))

    def test_image_edit_request_contains_exact_map_and_sampling_knobs(self) -> None:
        with TemporaryDirectory() as root:
            map_path = Path(root) / "receiver.png"
            Image.new("RGB", (64, 32), (5, 25, 90)).save(map_path)
            request = sdcpp.request_payload("preserve street geometry", map_path,
                                            64, 32, 17, 25, 6.0)
            self.assertEqual(base64.b64decode(request["ref_images"][0]), map_path.read_bytes())
            self.assertEqual((request["width"], request["height"]), (64, 32))
            self.assertEqual((request["seed"], request["sample_params"]["sample_steps"]),
                             (17, 25))
            self.assertEqual(request["sample_params"]["guidance"]["txt_cfg"], 6.0)
            self.assertEqual(request["sample_params"]["guidance"]["img_cfg"], 1.0)

    def test_async_job_extracts_one_rgb_png_and_preserves_job_id(self) -> None:
        image_stream = io.BytesIO()
        Image.new("RGBA", (64, 32), (1, 2, 3, 255)).save(image_stream, "PNG")
        png = base64.b64encode(image_stream.getvalue()).decode("ascii")
        replies = [
            {"id": "job_1", "poll_url": "/sdcpp/v1/jobs/job_1"},
            {"id": "job_1", "status": "generating"},
            {"id": "job_1", "status": "completed", "started": 1000,
             "completed": 2000, "result": {"images": [{"b64_json": png}]}},
        ]
        with patch.object(sdcpp, "http_json", side_effect=replies) as http, \
                patch.object(sdcpp.time, "sleep"):
            image, state = sdcpp.submit_and_wait("http://127.0.0.1:12345", {"prompt": "x"}, 10)
        self.assertEqual(http.call_count, 3)
        self.assertEqual((image.mode, image.size), ("RGB", (64, 32)))
        self.assertEqual(state["id"], "job_1")

    def test_fusion_output_uses_received_text_and_map(self) -> None:
        with TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source.png"
            condition = root_path / "map.png"
            Image.new("RGB", (64, 32), "gray").save(source)
            Image.new("RGB", (64, 32), "blue").save(condition)
            data_dir = root_path / "data"
            data_dir.mkdir()
            manifest = {"id": "sample_1", "model_size": [64, 32],
                        "source": str(source), "condition_map": str(condition)}
            (data_dir / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            receiver_jsonl = root_path / "receiver.jsonl"
            receiver_jsonl.write_text(json.dumps({"id": "sample_1", "caption": "a bus",
                                                  "condition_map": str(condition)}) + "\n",
                                      encoding="utf-8")
            weight = root_path / "weight.safetensors"
            weight.write_bytes(b"weight")
            args = argparse.Namespace(
                run_dir=data_dir, receiver_jsonl=receiver_jsonl,
                output_dir=root_path / "generated", server_bin=Path(sys.executable),
                diffusion_model=weight, vae=weight, llm=weight, llm_vision=None,
                gpu_device="2", port=12345, seed=42, steps=2, cfg_scale=6.0,
                startup_timeout=1, job_timeout=1, offload_to_cpu=False,
                backend_spec="diffusion=cuda0,te=cuda0,vae=cuda0", auto_fit="on",
                max_vram="cuda0=46", diffusion_fa=True, verbose=True,
                precision_profile="mixed",
                fusion_mask_prompt_policy="conditional",
            )

            class FakeProcess:
                def __init__(self):
                    self.command = None
                    self.env = None
                    self.stopped = False

                def poll(self):
                    return 0 if self.stopped else None

                def terminate(self):
                    self.stopped = True

                def wait(self, timeout=None):
                    return 0

            # A missing receiver map must never fall back to the clean sender map.
            receiver_jsonl.write_text(json.dumps({"id": "sample_1", "caption": "a bus"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Receiver condition map missing"):
                sdcpp.validate_inputs(args)
            receiver_jsonl.write_text(json.dumps({"id": "sample_1", "caption": "a bus",
                                                  "condition_map": str(condition)}) + "\n", encoding="utf-8")
            fake_process = FakeProcess()
            def fake_popen(command, **kwargs):
                fake_process.command = command
                fake_process.env = kwargs["env"]
                return fake_process

            replies = [
                (Image.new("RGB", (64, 32), "red"),
                 {"id": "job_1", "started": 1000, "completed": 1002}),
            ]
            with patch.object(sdcpp, "check_port_available"), \
                    patch.object(sdcpp.subprocess, "Popen", side_effect=fake_popen), \
                    patch.object(sdcpp, "wait_for_server", return_value={
                        "model": {"name": "Qwen-Image-2.1"},
                        "supported_modes": ["img_gen"],
                        "features_by_mode": {"img_gen": {"ref_images": True}},
                    }), \
                    patch.object(sdcpp, "submit_and_wait", side_effect=replies) as submit:
                sdcpp.generate(args)
            rows = sdcpp.read_jsonl(args.output_dir / "results.jsonl")
            self.assertEqual([row["branch"] for row in rows], ["fusion"])
            self.assertTrue((args.output_dir / "sample_1" / "comparison.png").is_file())
            self.assertEqual([row["condition_map"] for row in rows], [str(condition)])
            self.assertEqual([row["placement"] for row in rows], ["auto_fit"])
            self.assertEqual([row["server_generation_seconds"] for row in rows], [2.0])
            self.assertIsNone(rows[0]["model_load_seconds"])
            profile = json.loads((args.output_dir / "sdcpp_profile.json").read_text())
            self.assertIn("server_startup_seconds", profile)
            self.assertIn("first_request_wall_seconds", profile)
            self.assertIsNone(profile["model_load_seconds"])
            self.assertTrue(fake_process.stopped)
            self.assertIn("--backend", fake_process.command)
            self.assertIn("--diffusion-fa", fake_process.command)
            self.assertIn("cuda0=46", fake_process.command)
            self.assertEqual(fake_process.env["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(submit.call_count, 1)
            payload = submit.call_args.args[1]
            self.assertEqual(len(payload["ref_images"]), 1)
            self.assertIn("a bus", payload["prompt"])



if __name__ == "__main__":
    unittest.main()
