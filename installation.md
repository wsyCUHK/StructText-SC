# Installation and external assets

## CPU environment

Use Python 3.10 or later. Install the workflow's lightweight requirements:

```bash
python -m pip install -r requirements.txt
python scripts/validate_structtext_data.py
python -m unittest discover -s tests -v
```

The base requirements are `numpy`, `Pillow`, `requests` and `reedsolo`. The data validator itself only needs the Python standard library. Versions used for this package's CPU validation are recorded in [validation.md](validation.md); this does not define a validated GPU stack.

## Model execution

Use Linux, Bash and a suitable GPU environment for the full workflow. Obtain the dataset, third-party source repositories and model files separately. No model files are bundled.

| Stage | Required asset or environment |
| --- | --- |
| Source images and GT reference | Cityscapes data in the directory structure in the [workflow guide](cityscapes-workflow.md) |
| BLIP captions | Compatible PyTorch/Transformers stack and BLIP checkpoint; see `requirements-blip.txt` |
| Qwen3-VL or Gemma4 captions | Compatible model checkpoint, Docker GPU runtime and SGLang service; see `scripts/run_cityscapes_gt_caption_sglang.sh` |
| DRN sender maps | External DRN repository with `drn.py` and compatible `segment_compat.py`, plus DRN-D-105 checkpoint |
| SwinJSCC maps | Compatible external SwinJSCC repository and the exact checkpoint format described below |
| Diffusers reconstruction | Compatible PyTorch/CUDA, Transformers, Accelerate and Diffusers exposing `QwenImage21Pipeline`; see `requirements-diffusers.txt` |
| `sd-server` reconstruction | A compatible build of `stable-diffusion.cpp` with image conditioning, plus corresponding model components |

The family names supported by the code are not substitutes for exact experiment checkpoint IDs. Record the selected model versions, revisions, precision, resolution and library versions for each run. `sd-server` and Diffusers are alternative backends; this release does not establish numerical equivalence between them.

## SwinJSCC checkpoint contract

The adapter requires a payload with `model_state`, `config` and a positive integer `step`. It validates the model architecture, `model_size`, `channel`, `distortion`, `C`, `snr_db`, `width`, `height`, `crop_size` and `network_sha256`. In particular, the hash must match the external repository's `net/network.py`.

An arbitrary upstream checkpoint is not a drop-in replacement. A matching training recipe and checkpoint are not supplied here. The adapter should only be used with assets satisfying its checks; see `scripts/cityscapes_v2_swinjscc.py` for the exact contract.

## Configuration

Edit [`configs/cityscapes.example.env`](../configs/cityscapes.example.env), then source it from the repository root. Populate all paths required by your selected stages. The shared codebook corpus and its `.json` metadata are required by the top-level entry point in every channel mode.

The defaults in that example are for a small GT-map / BLIP / pixel-AWGN / Diffusers demonstration. They are not the paper's final experimental settings. Follow the [workflow guide](cityscapes-workflow.md) for corpus preparation and other modes.
