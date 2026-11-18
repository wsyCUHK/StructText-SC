#!/usr/bin/env python3
"""Batch adapter for the Huffman-RS packet-loss channel.

The existing packet-loss implementation is a single-text demonstration that
prints its receiver output. This adapter applies the same channel to one
caption per input line and writes the output contract expected by merge_id.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hf_rs_packet_loss import (
    BertMaskFiller,
    SharedByteHuffmanCodec,
    decode_source_payloads_to_tokens,
    detokenize,
    encode_text_to_packets,
    format_tokens_with_masks,
    gilbert_elliott_packet_loss_channel,
    iid_packet_loss_channel,
    recover_source_payloads,
    token_recovery_accuracy,
    tokenize_text,
    tokenize_text_unicode,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch Huffman-RS packet-loss channel for caption files."
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--summary_file", required=True)
    parser.add_argument("--codebook_corpus", required=True)
    parser.add_argument("--text_encoding", default="utf-8")

    parser.add_argument("--loss_model", choices=["iid", "ge"], default="iid")
    parser.add_argument("--loss_rate", type=float, default=0.1)
    parser.add_argument("--channel_seed", type=int, default=42)
    parser.add_argument("--interleaver_seed", type=int, default=2025)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--p", type=int, default=4)
    parser.add_argument("--payload_size", type=int, default=128)
    parser.add_argument("--huffman_pseudocount", type=int, default=1)
    parser.add_argument("--tokenizer_mode", choices=["legacy", "unicode"], default="legacy",
                        help="Packet-channel tokenizer; legacy preserves earlier runs")

    parser.add_argument("--ge_good_to_bad", type=float, default=0.05)
    parser.add_argument("--ge_bad_to_good", type=float, default=0.20)
    parser.add_argument("--ge_loss_good", type=float, default=0.01)
    parser.add_argument("--ge_loss_bad", type=float, default=0.80)
    parser.add_argument(
        "--ge_initial_state",
        choices=["good", "bad"],
        default="good",
    )

    parser.add_argument("--use_bert", action="store_true")
    parser.add_argument("--bert_model", default="bert-base-cased")
    parser.add_argument("--bert_device", default="cpu")
    parser.add_argument("--bert_top_k", type=int, default=20)
    return parser


def read_caption_lines(path: Path, encoding: str) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding=encoding).splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(
            f"Input caption file must contain one non-empty caption per line: {path}"
        )
    return [line.strip() for line in lines]


def load_codebook(path: Path, encoding: str, pseudocount: int) -> SharedByteHuffmanCodec:
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared codebook corpus not found: {path}. "
            "This file is required to avoid building a codebook from the test caption."
        )
    corpus = path.read_text(encoding=encoding)
    if not corpus.strip():
        raise ValueError(f"Shared codebook corpus is empty: {path}")
    return SharedByteHuffmanCodec([corpus], pseudocount=pseudocount)


def transmit_one(
    text: str,
    codec: SharedByteHuffmanCodec,
    args: argparse.Namespace,
    index: int,
    filler: Optional[BertMaskFiller],
) -> Tuple[str, Dict[str, object]]:
    tokenizer_mode = getattr(args, "tokenizer_mode", "legacy")
    tokens = tokenize_text_unicode(text) if tokenizer_mode == "unicode" else tokenize_text(text)
    if not tokens:
        raise ValueError(f"Caption {index} contains no supported tokens.")
    if tokenizer_mode == "unicode" and "".join(tokens) != "".join(text.split()):
        raise ValueError(f"Unicode tokenizer dropped non-whitespace content in caption {index}")

    huffman_bits = sum(codec.encode(token.encode("utf-8"))[1] for token in tokens)
    packets, session = encode_text_to_packets(
        tokens=tokens,
        huffman_codec=codec,
        source_packets_per_block=args.k,
        parity_packets_per_block=args.p,
        packet_payload_size=args.payload_size,
        interleaver_seed=args.interleaver_seed + index,
    )

    channel_seed = args.channel_seed + index
    if args.loss_model == "iid":
        received, lost = iid_packet_loss_channel(
            packets=packets,
            packet_loss_probability=args.loss_rate,
            seed=channel_seed,
            shuffle_received_packets=True,
        )
    else:
        received, lost = gilbert_elliott_packet_loss_channel(
            packets=packets,
            good_to_bad_probability=args.ge_good_to_bad,
            bad_to_good_probability=args.ge_bad_to_good,
            loss_probability_good=args.ge_loss_good,
            loss_probability_bad=args.ge_loss_bad,
            initial_state=args.ge_initial_state,
            seed=channel_seed,
            shuffle_received_packets=True,
        )

    source_payloads, reports = recover_source_payloads(
        received_packets=received,
        session=session,
    )
    recovered_tokens = decode_source_payloads_to_tokens(
        source_payloads_by_block=source_payloads,
        session=session,
        huffman_codec=codec,
    )

    masked_tokens = format_tokens_with_masks(recovered_tokens)
    output_tokens = masked_tokens
    bert_accuracy: Optional[float] = None
    if filler is not None:
        output_tokens = filler.fill(recovered_tokens)
        bert_accuracy = token_recovery_accuracy(tokens, output_tokens)

    unresolved_tokens = sum(token is None for token in recovered_tokens)
    transmitted_bits = len(packets) * args.payload_size * 8
    received_bits = len(received) * args.payload_size * 8
    lost_bits = len(lost) * args.payload_size * 8
    source_packet_bits = (
        session.number_of_blocks
        * session.source_packets_per_block
        * args.payload_size
        * 8
    )

    stats: Dict[str, object] = {
        "caption_index": index,
        "token_count": len(tokens),
        "huffman_source_bits": huffman_bits,
        "transmitted_packet_count": len(packets),
        "received_packet_count": len(received),
        "lost_packet_count": len(lost),
        "realized_packet_loss_rate": len(lost) / len(packets),
        "source_packet_bits": source_packet_bits,
        "transmitted_packet_bits": transmitted_bits,
        "received_packet_bits": received_bits,
        "lost_packet_bits": lost_bits,
        "rs_blocks": session.number_of_blocks,
        "rs_k": session.source_packets_per_block,
        "rs_p": session.parity_packets_per_block,
        "rs_code_rate": session.code_rate,
        "unresolved_token_count": unresolved_tokens,
        "unresolved_token_rate": unresolved_tokens / len(tokens),
        "rs_successful_blocks": sum(report.rs_success for report in reports),
        "rs_failed_blocks": sum(not report.rs_success for report in reports),
    }
    if bert_accuracy is not None:
        stats["bert_token_accuracy"] = bert_accuracy

    return detokenize(output_tokens), stats


def aggregate_stats(
    per_caption: Sequence[Dict[str, object]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    def total(name: str) -> int:
        return sum(int(item[name]) for item in per_caption)

    total_packets = total("transmitted_packet_count")
    total_tokens = total("token_count")
    total_lost = total("lost_packet_count")
    total_huffman_bits = total("huffman_source_bits")
    total_transmitted_bits = total("transmitted_packet_bits")
    total_unresolved = total("unresolved_token_count")

    summary: Dict[str, object] = {
        "channel": "packet_loss",
        "loss_model": args.loss_model,
        "configured_loss_rate": args.loss_rate if args.loss_model == "iid" else None,
        "realized_packet_loss_rate": total_lost / total_packets,
        "caption_count": len(per_caption),
        "token_count": total_tokens,
        "huffman_source_bits": total_huffman_bits,
        "source_packet_bits": total("source_packet_bits"),
        "transmitted_packet_bits": total_transmitted_bits,
        "received_packet_bits": total("received_packet_bits"),
        "lost_packet_bits": total("lost_packet_bits"),
        "effective_huffman_rate": total_huffman_bits / total_transmitted_bits,
        "unresolved_token_count": total_unresolved,
        "unresolved_token_rate": total_unresolved / total_tokens,
        "rs_successful_blocks": total("rs_successful_blocks"),
        "rs_failed_blocks": total("rs_failed_blocks"),
        "rs": {
            "k": args.k,
            "p": args.p,
            "n": args.k + args.p,
            "code_rate": args.k / (args.k + args.p),
            "payload_size_bytes": args.payload_size,
        },
        "seeds": {
            "channel_seed": args.channel_seed,
            "interleaver_seed": args.interleaver_seed,
        },
    }
    if args.loss_model == "ge":
        summary["gilbert_elliott"] = {
            "good_to_bad": args.ge_good_to_bad,
            "bad_to_good": args.ge_bad_to_good,
            "loss_good": args.ge_loss_good,
            "loss_bad": args.ge_loss_bad,
            "initial_state": args.ge_initial_state,
        }
    return summary


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    summary_path = Path(args.summary_file)
    codebook_path = Path(args.codebook_corpus)

    captions = read_caption_lines(input_path, args.text_encoding)
    codec = load_codebook(codebook_path, args.text_encoding, args.huffman_pseudocount)
    filler = None
    if args.use_bert:
        filler = BertMaskFiller(
            model_name=args.bert_model,
            device=args.bert_device,
            top_k=args.bert_top_k,
        )

    recovered_captions: List[str] = []
    per_caption: List[Dict[str, object]] = []
    for index, caption in enumerate(captions):
        recovered, stats = transmit_one(
            text=caption,
            codec=codec,
            args=args,
            index=index,
            filler=filler,
        )
        recovered_captions.append(recovered)
        per_caption.append(stats)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(recovered_captions) + "\n", encoding="utf-8")

    summary = aggregate_stats(per_caption, args)
    summary["input_file"] = str(input_path)
    summary["output_file"] = str(output_path)
    summary["codebook_corpus"] = str(codebook_path)
    summary["tokenizer_mode"] = args.tokenizer_mode
    summary["per_caption"] = per_caption
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Packet-loss captions written to: {output_path}")
    print(f"Packet-loss summary written to: {summary_path}")
    print(
        "Realized packet loss rate: "
        f"{summary['realized_packet_loss_rate']:.4%}"
    )
    print(f"RS code rate: {args.k / (args.k + args.p):.4f}")
    print(f"Transmitted packet bits: {summary['transmitted_packet_bits']}")
    print(f"Unresolved token rate: {summary['unresolved_token_rate']:.4%}")


if __name__ == "__main__":
    main()
