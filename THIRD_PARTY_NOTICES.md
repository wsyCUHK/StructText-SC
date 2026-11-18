# Resource provenance and notices

## Workflow code

The supplied workflow uses the [MIT license](LICENSE), with its original copyright attribution retained. Updates to that workflow and the new validation code in this snapshot use the same license.

## Legacy StructText-SC SFT file

`train_20260428.jsonl` is preserved byte for byte from the existing public [StructText-SC repository](https://github.com/wsyCUHK/StructText-SC). Its source and hash are recorded in `data/provenance.json`. The existing repository described academic research use and referred users to MS COCO terms for source images. This update does not assign a new blanket data license or relicense those images.

## COCO and Cityscapes

Image binaries referenced by the SFT data and the Cityscapes dataset are not bundled. Obtain and use source data according to the terms provided by [COCO](https://cocodataset.org/) and [Cityscapes](https://www.cityscapes-dataset.com/). The archived illustration in `assets/qualitative_example.png` contains a Cityscapes-derived source image and generated comparison panels; it is separate from the MIT-licensed code.

## External software and models

The workflow integrates externally obtained BLIP, Qwen/Gemma, Qwen-Image-2.1, DRN, SwinJSCC, SGLang, Diffusers and stable-diffusion.cpp components. Their code, model files, usage conditions and licenses remain with their respective upstream providers. They are not redistributed as model assets in this package.
