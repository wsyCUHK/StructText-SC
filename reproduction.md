# Reproduction status

This release includes the original public SFT file and an executable Cityscapes workflow. It does not yet reproduce every stage or result of the paper.

| Stage | Available | Required for complete paper reproduction |
| --- | --- | --- |
| Legacy textual supervision | Original 1,234-record COCO JSONL, schema, example and integrity report | Final dataset inventory, source-image mapping, held-out manifests and construction provenance |
| Candidate evaluation and selection | Method overview | Exact candidate generation, quality metrics, selection rule and human-audit protocol |
| I2T adaptation | SFT data format | Model-specific training recipe, hyperparameters, fine-tuned weights and checkpoint IDs |
| Cityscapes runtime | Preparation, captioning, channel and joint reconstruction stages | Final manuscript settings and a verified end-to-end environment |
| Map channel | Pixel-AWGN demonstration, oracle and SwinJSCC adapter | Matching SwinJSCC checkpoint and training recipe |
| Results | One archived qualitative illustration | Fixed evaluation IDs, metric implementation, raw results, and table/figure generation |

## Input contracts

Qwen/Gemma captioning defaults to `source_only`. The source photograph is the captioner's only image input. The optional `source_and_map` setting is an extension and must be labeled separately in comparisons. BLIP uses the source image only.

Image reconstruction uses the received text and received segmentation map. The receiver does not receive the original photograph as a reconstruction condition. Ground-truth maps (`MASK_SOURCE=gt`) and DRN-predicted maps (`MASK_SOURCE=drn`) are separate experimental settings.

Changing a caption input mode, checkpoint or prompt changes the caption distribution. Regenerate the affected captions and a compatible training-caption Huffman corpus; do not reuse historical results as evidence for the modified configuration.

## Channel and resource accounting

`packet_awgn` applies additive noise directly to segmentation-map RGB values. `packet_swinjscc` sends maps through a learned representation. `oracle` provides a noise-free reference. These configurations must remain distinct in reports.

Text uses Huffman and Reed–Solomon coding with a packet-loss channel. A text packet-loss rate and a map SNR do not define an equal communication budget. A budget comparison should record at least image resolution, text payload and coding overhead, map representation or channel-use count, both channel settings, and the allocation to the two streams.

## Paper results

The 4,800 reconstructions discussed in the manuscript revision concern reconstruction analysis; that number is not a count of records in the legacy SFT file. The composition of that analysis and the final paper's numerical results are not reconstructed from this code snapshot.

Before adding a claim of complete paper reproduction, supply the final experiment manifests/configurations, exact model assets, metric scripts and raw results, then verify that they reproduce the manuscript tables and figures.
