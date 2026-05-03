# StructText-SC

This repository provides the supervised fine-tuning split of **StructText-SC**, a dataset and benchmark for text-only generative semantic communication.

StructText-SC is designed for low-bitrate wireless image transmission, where an image is first converted into a textual description, the text is transmitted through a semantic communication module, and the receiver uses the decoded text as a prompt for image regeneration. Unlike conventional image--text datasets that mainly emphasize semantic alignment, StructText-SC focuses on whether a textual description can support structure-preserving image regeneration after wireless semantic transmission.

It contains 1500+ human-aligned image--text pairs selected from a closed-loop wireless generation process. Each retained sample is selected by combining multi-dimensional objective assessment and human agreement auditing.

Each sample consists of:

a source image path,
a system prompt,
a user prompt for communication-oriented image description,
a target assistant response, i.e., the retained textual description.

The retained textual descriptions are intended to be:

semantically meaningful,
structurally informative,
robust to semantic-channel distortion,
suitable for supervised fine-tuning of image-to-text multimodal models.

## Dataset Overview

The released training file is:

```text
train_20260428.jsonl

## Data Format
Example:
{
  "messages": [
    {
      "role": "system",
      "content": "# Role\nYou are a helpful assistant specializing in image understanding..."
    },
    {
      "role": "user",
      "content": "<image># Task\nYour task is to generate a single, fluent, and coherent paragraph describing the visual content of an image..."
    },
    {
      "role": "assistant",
      "content": "A centrally positioned white, fluffy teddy bear with a red ribbon around its neck serves as the focal point..."
    }
  ],
  "images": [
    "Dataset/coco3k_source_data_train/000000000491.jpg"
  ]
}


## Intended Use

StructText-SC is intended for research on:

text-only generative semantic communication,
low-bitrate wireless image transmission,
communication-oriented image-to-text generation,
supervised fine-tuning of compact multimodal large language models,
benchmarking text-only image transmission over noisy wireless channels.

In our paper, StructText-SC is used to fine-tune compact image-to-text models at the transmitter side. The text semantic communication module and the receiver-side text-to-image generator are kept fixed, so that the benchmark focuses on whether the transmitter can generate compact and structure-preserving textual descriptions.

## Construction Pipeline

StructText-SC is constructed through a closed-loop wireless generation process:

Source images are selected from MS COCO.
Multiple candidate textual descriptions are generated for each image using an image-to-text multimodal model.
Each candidate description is transmitted through a text semantic communication module.
The clean and post-transmission texts are used to regenerate images.
Candidate descriptions are evaluated using semantic, perceptual, and structural metrics.
The metric-selected candidates are further audited by human annotators.
The retained human-aligned pairs are used for supervised fine-tuning.
## Notes on Images

The released JSONL file stores image paths relative to the expected dataset directory, for example:

Dataset/coco3k_source_data_train/000000000491.jpg

Please make sure that the corresponding image files are placed under the same relative directory structure before training.

If the original images come from MS COCO, users should follow the MS COCO license and terms of use. This repository is intended for academic research only.
