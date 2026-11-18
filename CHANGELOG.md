# Changelog

## 2026-11-18

- Reorganized the repository overview around textual supervision and joint text/map reconstruction.
- Preserved the public `train_20260428.jsonl` file and its original path; measured 1,234 records and unique image references, replacing the imprecise historical sample count in the current overview.
- Added a dataset example, provenance, measured statistics and a standalone schema/integrity validator.
- Included the supplied Cityscapes workflow, retaining its entry-point names.
- Changed Qwen/Gemma captioning to source-image-only by default, with an explicit map-assisted extension, mode-aware prompts and provenance.
- Added installation, configuration, reproduction-scope and resource documentation.
- Included one unchanged archived qualitative comparison; omitted the other large PNGs from this compact source package.
- Added automated CPU validation. Fine-tuning, matching SwinJSCC weights, final paper evaluation manifests and paper metric reproduction remain outside this snapshot.
- Added a copy fallback when Windows denies hard links while organizing results.
