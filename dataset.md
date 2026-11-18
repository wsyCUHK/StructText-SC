# StructText-SC data

## Released file

`train_20260428.jsonl` is the existing public COCO-based supervised fine-tuning file. This snapshot preserves its exact bytes and filename for compatibility with existing users.

| Measured property | Value |
| --- | --- |
| Non-empty JSONL records | 1,234 |
| Unique image references | 1,234 |
| Images per record | 1 |
| Duplicate complete records | 0 |
| Message roles | `system`, `user`, `assistant`, in that order |
| File size | 3,428,995 bytes |
| SHA-256 | `35fe589d246c5098cb6e5fb34a76e253b7129a2f61055174f156230385679662` |

These counts describe this released file. The older repository overview said “1500+”; that statement is not the measured size of the available JSONL. No additional samples are synthesized or inferred in this update. The paper's 4,800-reconstruction analysis must not be interpreted as the SFT record count.

## Format

Every line is a JSON object with `messages` and `images`. The system and user messages specify the captioning task; the assistant message is the retained target description. The user text contains an `<image>` marker. Training integrations must preserve the relationship between that marker and the image input expected by their model/template.

The complete first record is available as [`data/example.json`](../data/example.json). The image reference format is:

```text
Dataset/coco3k_source_data_train/000000000491.jpg
```

The JSONL contains image references, not image binaries or segmentation maps. A training/validation/test manifest and an exact COCO source archive mapping are not provided in the legacy file. Do not infer a new evaluation split from this file or use training records as held-out results.

## Prepare images

Obtain the source images separately under their applicable terms. If you already have the original image directory, preserve this layout relative to a chosen dataset parent:

```text
DATA_PARENT/
  Dataset/
    coco3k_source_data_train/
      000000000491.jpg
      ...
```

Check schema and duplicates with the standard library only:

```bash
python scripts/validate_structtext_data.py
```

Also check that all referenced images exist beneath the selected parent:

```bash
python scripts/validate_structtext_data.py --image-root /path/to/DATA_PARENT
```

Write a new measured report when auditing another file:

```bash
python scripts/validate_structtext_data.py --data /path/to/train.jsonl --output /path/to/stats.json
```

This validator checks the available data structure and file references. It does not establish the original human-audit process or image content quality.

## Relationship to Cityscapes

The Cityscapes workflow uses its own dataset, manifests and captions. Its text codebook must be built from a separate **Cityscapes training-caption run**. The COCO SFT JSONL is not a ready-made Cityscapes manifest or codebook corpus. This snapshot does not claim the COCO records are a newly released Cityscapes SFT dataset.

The distinction between textual training targets and receiver conditioning is also important: the updated runtime defaults to source-image-only captioning, while reconstruction uses received text and a segmentation map.
