# Validation of this snapshot

Validation date: **2026-11-18**.

## Checks performed

| Check | Result |
| --- | --- |
| Original public SFT file preservation | Exact SHA-256 matches `data/provenance.json` |
| Dataset schema and references | 1,234 valid records; 1,234 unique image paths; no duplicate records; image binaries not checked |
| Python/JSON syntax | All package Python and JSON files parse |
| Bash syntax | Main workflow, SGLang wrapper and example environment file pass `bash -n` |
| CPU test suite | 55 tests discovered: 53 passed, 2 skipped, 0 failed |
| Linux shell integration tests | Included, skipped on the local Windows platform |

The tests cover source-image-only captioning without map access, explicit map-assisted requests, prompt compatibility, caption provenance, packet/map processing, mocked generator interfaces, output organization, and dataset integrity. The output organizer includes a tested copy fallback when a platform denies hard links.

## Local CPU environment

- Windows; CPython 3.12.14.
- NumPy 2.3.5; Pillow 13.0.0.
- Requests 2.34.2; reedsolo 1.7.0.

These are recorded test versions, not a complete GPU environment lock. The repository requirements remain unpinned.

## Not established by these checks

No model weights were loaded, no real GPU reconstruction was run, and the paper's tables/figures were not reproduced. Model-serving and generation tests use mocks where required. Source-image availability, dataset licensing, training convergence, exact model revisions and GPU dependency compatibility are outside the CPU checks.

The GitHub Actions configuration in `.github/workflows/cpu-checks.yml` runs CPU tests and shell checks on Linux when this repository is pushed. Its hosted execution has not been observed as part of preparing this local package.

## Run the checks

From the repository root, after installing `requirements.txt`:

```bash
python scripts/validate_structtext_data.py
python -m unittest discover -s tests -v
bash -n work_flow_GenSemComv2.sh
bash -n scripts/run_cityscapes_gt_caption_sglang.sh
bash -n configs/cityscapes.example.env
```
