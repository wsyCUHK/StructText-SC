# Cityscapes joint reconstruction workflow

GenSemCom V2 provides an experimental workflow for **semantic image transmission and reconstruction** using the [Cityscapes](https://www.cityscapes-dataset.com/) dataset. The pipeline combines semantic segmentation, image-to-text generation, simulated communication channels, and conditional image generation.

The release generates one image per sample using both the received segmentation map and received text.

> [!NOTE]
> This is a research workflow. Some components depend on external repositories, pretrained checkpoints, and hardware-specific environments. The included tests do not establish end-to-end compatibility with every supported configuration.

## 1. Workflow Overview

1. **Extract semantic information.** Load an original Cityscapes image and generate a segmentation map using **DRN-D-105** (`MASK_SOURCE=drn`), or use the dataset's ground-truth segmentation labels when available (`MASK_SOURCE=gt`).
2. **Generate a text description.** **Qwen3-VL** and **Gemma4** use the original image and a prompt to produce captions by default (`I2T_CAPTION_INPUT_MODE=source_only`). An explicit `source_and_map` extension can also supply the sender map. **BLIP** uses only the original image.
3. **Simulate transmission.** Text is encoded using a Huffman codebook derived from a separate training corpus, followed by Reed–Solomon (RS) coding and a packet-loss channel. Segmentation maps are transmitted through either a pixel-level additive white Gaussian noise (AWGN) simulation or a trained **SwinJSCC** pipeline. The `oracle` mode provides a noise-free reference.
4. **Reconstruct images.** Feed the received text and segmentation map to the image-generation backend. Save one `fusion` output per sample.
5. **Save reproducible artifacts.** Store manifests, captions, channel outputs, reconstructed images, prompts, configuration snapshots, and logs. An `artifacts/` index organizes outputs for inspection.

**Release scope:** Qwen/Gemma now default to source-image-only captioning. Existing prompt variants under `prompt/` declare `input_mode: source_only`. For the optional map-assisted extension, set `I2T_CAPTION_INPUT_MODE=source_and_map` and leave `I2T_PROMPT_FILE` empty to use the built-in map prompt, or supply a matching custom prompt. The input mode is recorded in caption provenance and the run configuration. Regenerate captions and the training-caption codebook after changing the mode or prompt. Use `packet_swinjscc` only with a matching checkpoint. The release does not include SwinJSCC training, its checkpoint, I2T fine-tuning, or the paper's metric evaluation pipeline; see [reproduction status](reproduction.md).

### Important Channel Modeling Notes

- In `packet_awgn` mode, **AWGN is applied directly to the RGB pixels of the segmentation map**. This is a simplified simulation, **not** SwinJSCC or a complete wireless modulation/demodulation chain.
- Text and segmentation-map channels use different representations and transmission mechanisms. **Do not interpret their SNR and packet-loss settings as equivalent communication-resource budgets.**

## 2. Requirements and External Assets

**Recommended platform:** Linux with Python 3.10 or later.

Base Python dependencies are listed in `requirements.txt`. Optional model dependencies are listed in `requirements-blip.txt` and `requirements-diffusers.txt`; select a PyTorch build compatible with your CUDA setup. External repositories may need additional dependencies. Depending on the selected components:

| Component | Additional requirements |
| --- | --- |
| GPU execution | Compatible PyTorch and CUDA versions |
| BLIP image-to-text | Transformers and the appropriate model dependencies |
| Qwen3-VL / Gemma4 image-to-text | Docker with GPU support and an SGLang image |
| Diffusers image generation | Diffusers and Accelerate |
| `sd-server` image generation | A user-built, compatible version of `stable-diffusion.cpp` with image-conditioning support |
| DRN segmentation / SwinJSCC | External source repositories and corresponding checkpoints |

Dependency versions are **not pinned**. CPU validation is documented in [validation.md](validation.md); a clean GPU environment and full model execution have not been verified for this snapshot. Validate model compatibility on your hardware.

**Not included in this repository:** Cityscapes source dataset, model weights, training outputs, or third-party project source code. One archived qualitative illustration is included separately; the original COCO SFT file is documented in [dataset.md](dataset.md). Obtain and use each asset in accordance with its respective license and terms.

### Expected Cityscapes Directory Layout

```text
DATASET_ROOT/
  leftImg8bit/{train,val,test}/<city>/<city>_<sequence>_<frame>_leftImg8bit.png
  gtFine/{train,val}/<city>/<city>_<sequence>_<frame>_gtFine_labelIds.png
  gtFine/{train,val}/<city>/<city>_<sequence>_<frame>_gtFine_color.png
```

- `MASK_SOURCE=gt` requires the corresponding ground-truth annotations. Official Cityscapes test labels are generally not publicly available.
- `MASK_SOURCE=drn` also supports test images for which ground-truth annotations are unavailable.
- By default, the pipeline expects the native Cityscapes resolution of **2048 × 1024**. Images that do not match the configured `WIDTH` and `HEIGHT` are rejected.
- The input resolution must match the resolution expected by the SwinJSCC checkpoint.

### DRN and SwinJSCC Setup

For DRN-based segmentation, provide:

- `DRN_REPO_ROOT`: directory of the external DRN repository, containing `drn.py` and `segment_compat.py`.
- `DRN_CHECKPOINT`: path to the DRN pretrained checkpoint.

`segment_compat.py` is an environment-compatible version of the DRN `segment.py`. In older code, replace `cuda(async=True)` with `cuda(non_blocking=True)` to avoid incompatibility with current Python syntax.

For SwinJSCC-based transmission, provide:

- `SWINJSCC_ROOT`: directory of the external SwinJSCC repository.
- `SWINJSCC_CHECKPOINT`: path to the trained SwinJSCC checkpoint.

The workflow also verifies that the SHA-256 hash of `net/network.py` in the SwinJSCC repository matches the hash recorded in the training checkpoint.

## 3. Prepare a Shared Text Codebook Corpus

**A separate training-split caption corpus and its metadata are required in all channel modes.** Do not build the codebook from the evaluation or transmission samples themselves.

First, prepare a manifest of training samples and run an image-to-text model to populate a training-caption run directory with:

```text
TRAIN_CAPTION_RUN_DIR/
  manifest.jsonl
  captions.jsonl
```

You can prepare the manifest with `scripts/cityscapes_gt_smoke.py prepare`, then generate captions using either `scripts/run_cityscapes_gt_caption_sglang.sh` or `scripts/cityscapes_v2_blip_caption.py`.

Build the shared codebook corpus:

```bash
python scripts/build_cityscapes_v2_codebook_corpus.py \
  --train-run-dir "$TRAIN_CAPTION_RUN_DIR" \
  --output "$CODEBOOK_CORPUS"
```

This command produces the corpus at `CODEBOOK_CORPUS` and a companion `.json` metadata file. Every manifest entry must belong to the **training split**. The workflow validates the metadata and SHA-256 hash. Configure the actual paths to both files, and ensure that the transmitter and receiver use the **same corpus and coding parameters**.

### Example: Prepare Training Captions with BLIP

After configuring `DATASET_ROOT`, `MODEL_PATH`, and `TRAIN_CAPTION_RUN_DIR`, run:

```bash
python scripts/cityscapes_gt_smoke.py prepare \
  --dataset-root "$DATASET_ROOT" --run-dir "$TRAIN_CAPTION_RUN_DIR" \
  --split train --limit 100 --mask-source gt \
  --width 2048 --height 1024 --require-native-size

python scripts/cityscapes_v2_blip_caption.py \
  --run-dir "$TRAIN_CAPTION_RUN_DIR" --model "$MODEL_PATH" \
  --device cuda --workers 1 --gpu-devices 0
```

Then execute the codebook-building command above.

The 100-image limit is **only an illustrative, runnable setup**. Select the actual corpus size and captioning model according to your experimental design. Training captions and logs may contain local paths or dataset-derived information; review them before public distribution.

## 4. Configuration and Execution

The workflow does not assume any fixed local path for datasets, models, checkpoints, third-party repositories, or the codebook. Configure the relevant environment variables before execution:

```bash
export DATASET_ROOT=""          # Root directory of the Cityscapes dataset
export MODEL_PATH=""            # Image-to-text model directory
export CODEBOOK_CORPUS=""       # Shared training-caption corpus file
export DRN_REPO_ROOT=""         # Required only for MASK_SOURCE=drn
export DRN_CHECKPOINT=""        # Required only for MASK_SOURCE=drn
export SWINJSCC_ROOT=""         # Required only for CHANNEL_MODE=packet_swinjscc
export SWINJSCC_CHECKPOINT=""   # Required only for CHANNEL_MODE=packet_swinjscc
export SDCPP_SERVER_BIN=""      # Absolute path to the sd-server executable
export SDCPP_DIFFUSION_MODEL="" # Diffusion model file
export SDCPP_LLM=""             # Text encoder model file
export SDCPP_VAE=""             # VAE model file
export IMAGE_MODEL=""           # Required only for IMAGE_BACKEND=diffusers
```

Only populate variables required for the components you select.

### Minimal Example

Start with a small validation run:

```bash
MASK_SOURCE=drn I2T_MODEL_FAMILY=blip \
CHANNEL_MODE=packet_awgn IMAGE_BACKEND=sdcpp_server \
GPU_DEVICES=0 IMAGE_WORKERS=1 BLIP_WORKERS=1 \
SPLIT=val LIMIT=2 PACKET_LOSS_RATE=0.2 MAP_SNR_DB=10 \
SDCPP_PRECISION_PROFILE=bf16 \
bash work_flow_GenSemComv2.sh
```

You may set `RUN_DIR` to choose a custom output directory. **The directory must not already exist.**

### Supported Options

| Setting | Values / behavior |
| --- | --- |
| `I2T_MODEL_FAMILY` | `blip`, `qwen3_vl`, `gemma4` |
| `I2T_CAPTION_INPUT_MODE` | `source_only` (default), or `source_and_map` for the Qwen/Gemma extension |
| `I2T_PROMPT_FILE` | Path to a JSON prompt matching the selected caption input mode for Qwen3-VL or Gemma4 |
| `CHANNEL_MODE` | `packet_awgn`, `packet_swinjscc`, `oracle` |
| `IMAGE_BACKEND` | `sdcpp_server`, `diffusers` |
| `GPU_DEVICES` | Comma-separated GPU IDs, e.g. `0,1` |
| `IMAGE_WORKERS` | Number of image-generation workers, e.g. `2` for two GPUs |
| `SDCPP_PRECISION_PROFILE` | `bf16` validates the tensor precision of three model components; other precision configurations use `mixed` and require user verification |

**Additional notes:**

- Qwen3-VL and Gemma4 are accessed through SGLang services. Their execution requires Docker and a configured GPU runtime.
- BLIP does **not** use the segmentation map or a prompt as its input.
- In a multi-GPU `sd-server` configuration, the workflow starts **one independent server per GPU**. It does not shard a single image-generation model across devices.
- The workflow uses the active Python environment by default. To configure separate environments for pipeline stages, set `CONDA_SH`, `I2T_ENV`, `BLIP_ENV`, `DRN_ENV`, `PACKET_ENV`, `SWINJSCC_ENV`, or `IMAGE_ENV` as appropriate.
- Each environment must contain the dependencies for its corresponding stage. Validate API availability, GPU memory requirements, and checkpoint compatibility on the target server.

## 5. Outputs

Each run creates `output/<timestamp>/` or the user-specified `RUN_DIR`:

```text
config/       Configuration, prompt copies, codebook hash, workflow script snapshot
logs/         Workflow, service, and client logs
data/         Input manifest, transmitter captions, channel data, received and generated images
artifacts/    Read-only, workflow-organized index (using hard links where possible)
```

Each sample has one `fusion` entry in `results.jsonl`. `comparison.png` displays the source, received map, and generated image.

The original image is used at the **transmitter** and is **not** provided as a reconstruction input at the receiver.

Generated images and logs may contain dataset content and local filesystem paths. Review all artifacts before publishing them.

## 6. Tests

Run the included unit tests with:

```bash
python -m unittest discover -s tests -v
```

Some tests use mocked interfaces. Passing the tests does **not** imply that the pipeline has been validated end to end on the intended GPUs, real checkpoints, or production servers.

## 7. License

The original code in this repository is available under the [MIT License](../LICENSE).
Datasets, model weights, and external software remain subject to their own licenses and terms.


