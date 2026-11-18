#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_style_huffman_rs_textpc.py

A paper-aligned, engineering-stable implementation of:

Text
 -> word-level tokenization
 -> deterministic random interleaving in a sending window
 -> shared byte-level Huffman coding
 -> one word record per systematic packet
 -> systematic packet-level Reed-Solomon coding over GF(2^8)
 -> IID or Gilbert-Elliott packet loss
 -> RS erasure decoding
 -> partial retention of received systematic packets when RS fails
 -> inverse interleaving
 -> [MASK] insertion for unrecovered words
 -> optional pretrained/fine-tuned BERT mask filling

Important design choices
------------------------
1. One interleaving window is one RS block.
2. One word/punctuation token is stored in one systematic source packet.
3. Huffman codebook is shared by transmitter and receiver.
4. RS is performed column-wise across fixed-length packet payloads.
5. Packet headers/metadata are treated as reliable side information in this
   simulation. Only entire payload packets are erased.
6. The default K=4, P=4 gives RS code rate K/(K+P)=1/2.
7. The optional BERT module is an inference baseline. To reproduce TextPC
   more faithfully, pass a locally fine-tuned checkpoint with --bert_model.

Dependencies
------------
Required:
    pip install reedsolo

Optional BERT concealment:
    pip install torch transformers
"""

from __future__ import annotations

import argparse
import binascii
import heapq
import math
import random
import re
import unicodedata
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from reedsolo import RSCodec, ReedSolomonError


# =============================================================================
# Constants and simple tokenization
# =============================================================================

MASK_TOKEN = "[MASK]"
PAD_FLAG = 0x01

# English words/numbers, individual CJK characters, and punctuation.
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9_]+(?:['’\-][A-Za-z0-9_]+)*"
    r"|[\u3400-\u4dbf\u4e00-\u9fff]"
    r"|[^\w\s]",
    flags=re.UNICODE,
)

RIGHT_ATTACHED_PUNCT = set(
    ".,!?;:%)]}，。！？；：、）》】〉」』”’…"
)
LEFT_ATTACHED_PUNCT = set(
    "([{（【《〈「『“‘"
)


def is_cjk_token(token: str) -> bool:
    """Return True when token is a single CJK character."""
    return (
        len(token) == 1
        and (
            "\u3400" <= token <= "\u4dbf"
            or "\u4e00" <= token <= "\u9fff"
        )
    )


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text at word/punctuation level.

    For English, a word is one token.
    For Chinese, each CJK character is one token.
    Punctuation is a separate token.
    """
    return TOKEN_PATTERN.findall(text)


def tokenize_text_unicode(text: str) -> List[str]:
    """Tokenize all non-whitespace Unicode content without dropping characters.

    Keep letters, numbers, combining marks, and internal apostrophes/hyphens
    together. CJK characters remain individual tokens as in the legacy mode.
    Other non-whitespace characters become their own tokens.
    """
    tokens: List[str] = []
    word: List[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    for index, char in enumerate(text):
        if char.isspace():
            flush_word()
        elif is_cjk_token(char):
            flush_word()
            tokens.append(char)
        elif char.isalnum() or char == "_" or unicodedata.category(char).startswith("M"):
            word.append(char)
        elif (char in "'’−-" and word and index + 1 < len(text) and
              (text[index + 1].isalnum() or text[index + 1] == "_")):
            word.append(char)
        else:
            flush_word()
            tokens.append(char)
    flush_word()
    return tokens


def detokenize(tokens: Sequence[str]) -> str:
    """
    Convert tokens back to readable text.

    This is a deterministic formatting rule, not a lossless whitespace codec.
    Evaluation should therefore focus on token accuracy, BLEU, or semantic
    similarity rather than raw byte equality.
    """
    output = ""

    for token in tokens:
        if not output:
            output = token
            continue

        previous = output[-1]

        no_space = (
            token in RIGHT_ATTACHED_PUNCT
            or previous in LEFT_ATTACHED_PUNCT
            or is_cjk_token(token)
            or (
                len(output) >= 1
                and is_cjk_token(output[-1])
            )
            or token.startswith("##")
        )

        if token.startswith("##"):
            output += token[2:]
        elif no_space:
            output += token
        else:
            output += " " + token

    return output


# =============================================================================
# Shared byte-level Huffman codec
# =============================================================================

@dataclass
class HuffmanNode:
    frequency: int
    symbol: Optional[int] = None
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None


class SharedByteHuffmanCodec:
    """
    Shared byte-level Huffman codec.

    The codebook is built from a shared corpus. A pseudocount is added to every
    byte value 0..255 so that previously unseen UTF-8 bytes remain encodable.
    Both transmitter and receiver must construct/load the same codebook.
    """

    def __init__(
        self,
        corpus_texts: Sequence[str],
        pseudocount: int = 1,
    ) -> None:
        if pseudocount <= 0:
            raise ValueError("pseudocount must be positive.")

        frequencies = Counter({value: pseudocount for value in range(256)})

        for text in corpus_texts:
            frequencies.update(text.encode("utf-8"))

        self.root = self._build_tree(dict(frequencies))
        self.codes = self._build_codes(self.root)

    @staticmethod
    def _build_tree(frequencies: Dict[int, int]) -> HuffmanNode:
        heap: List[Tuple[int, int, HuffmanNode]] = []
        order = 0

        for symbol in sorted(frequencies):
            node = HuffmanNode(
                frequency=frequencies[symbol],
                symbol=symbol,
            )
            heapq.heappush(heap, (node.frequency, order, node))
            order += 1

        while len(heap) > 1:
            freq_1, _, node_1 = heapq.heappop(heap)
            freq_2, _, node_2 = heapq.heappop(heap)

            parent = HuffmanNode(
                frequency=freq_1 + freq_2,
                left=node_1,
                right=node_2,
            )

            heapq.heappush(
                heap,
                (parent.frequency, order, parent),
            )
            order += 1

        return heap[0][2]

    @staticmethod
    def _build_codes(root: HuffmanNode) -> Dict[int, str]:
        codes: Dict[int, str] = {}

        def visit(node: HuffmanNode, prefix: str) -> None:
            if node.symbol is not None:
                codes[node.symbol] = prefix or "0"
                return

            if node.left is None or node.right is None:
                raise ValueError("Invalid Huffman tree.")

            visit(node.left, prefix + "0")
            visit(node.right, prefix + "1")

        visit(root, "")
        return codes

    @staticmethod
    def _pack_bits(bit_string: str) -> bytes:
        packed = bytearray((len(bit_string) + 7) // 8)

        for index, bit in enumerate(bit_string):
            if bit == "1":
                packed[index // 8] |= 1 << (7 - index % 8)

        return bytes(packed)

    @staticmethod
    def _iter_bits(data: bytes, bit_length: int) -> Iterable[int]:
        for index in range(bit_length):
            yield (
                data[index // 8]
                >> (7 - index % 8)
            ) & 1

    def encode(self, raw: bytes) -> Tuple[bytes, int]:
        bit_string = "".join(self.codes[value] for value in raw)
        return self._pack_bits(bit_string), len(bit_string)

    def decode(
        self,
        packed: bytes,
        bit_length: int,
        expected_byte_length: int,
    ) -> bytes:
        if expected_byte_length == 0:
            return b""

        if bit_length > len(packed) * 8:
            raise ValueError("Huffman bit length exceeds packed payload.")

        output = bytearray()
        node = self.root

        for bit in self._iter_bits(packed, bit_length):
            node = node.left if bit == 0 else node.right

            if node is None:
                raise ValueError("Invalid Huffman bit stream.")

            if node.symbol is not None:
                output.append(node.symbol)

                if len(output) == expected_byte_length:
                    break

                node = self.root

        if len(output) != expected_byte_length:
            raise ValueError(
                "Decoded byte length mismatch: "
                f"expected={expected_byte_length}, actual={len(output)}"
            )

        return bytes(output)


# =============================================================================
# Fixed-size word record
# =============================================================================

TOKEN_MAGIC = b"TK01"

# magic:4, flags:1, original UTF-8 byte length:2,
# Huffman bit length:2, CRC32:4 -> 13 bytes.
TOKEN_HEADER = struct.Struct(">4sBHHI")


@dataclass(frozen=True)
class DecodedRecord:
    token: Optional[str]
    is_padding: bool


def pack_token_record(
    token: str,
    codec: SharedByteHuffmanCodec,
    packet_payload_size: int,
    is_padding: bool = False,
) -> bytes:
    """
    Pack one token into one fixed-length systematic packet payload.
    """
    if packet_payload_size <= TOKEN_HEADER.size:
        raise ValueError(
            f"packet_payload_size must exceed {TOKEN_HEADER.size} bytes."
        )

    if is_padding:
        raw = b""
        packed = b""
        bit_length = 0
        crc32 = 0
        flags = PAD_FLAG
    else:
        raw = token.encode("utf-8")
        packed, bit_length = codec.encode(raw)
        crc32 = binascii.crc32(raw) & 0xFFFFFFFF
        flags = 0

    if len(raw) > 0xFFFF:
        raise ValueError("Token UTF-8 representation is too long.")

    if bit_length > 0xFFFF:
        raise ValueError("Token Huffman representation is too long.")

    record = (
        TOKEN_HEADER.pack(
            TOKEN_MAGIC,
            flags,
            len(raw),
            bit_length,
            crc32,
        )
        + packed
    )

    if len(record) > packet_payload_size:
        raise ValueError(
            "Encoded token does not fit in one source packet. "
            f"token={token!r}, record_size={len(record)}, "
            f"packet_payload_size={packet_payload_size}. "
            "Increase --payload_size."
        )

    return record.ljust(packet_payload_size, b"\x00")


def unpack_token_record(
    payload: bytes,
    codec: SharedByteHuffmanCodec,
) -> DecodedRecord:
    if len(payload) < TOKEN_HEADER.size:
        raise ValueError("Token payload is too short.")

    magic, flags, raw_length, bit_length, expected_crc = (
        TOKEN_HEADER.unpack_from(payload, 0)
    )

    if magic != TOKEN_MAGIC:
        raise ValueError("Token record magic mismatch.")

    if flags & PAD_FLAG:
        return DecodedRecord(token=None, is_padding=True)

    encoded_byte_length = (bit_length + 7) // 8
    start = TOKEN_HEADER.size
    end = start + encoded_byte_length

    if end > len(payload):
        raise ValueError("Truncated token record.")

    raw = codec.decode(
        payload[start:end],
        bit_length=bit_length,
        expected_byte_length=raw_length,
    )

    actual_crc = binascii.crc32(raw) & 0xFFFFFFFF

    if actual_crc != expected_crc:
        raise ValueError("Token CRC32 verification failed.")

    return DecodedRecord(
        token=raw.decode("utf-8"),
        is_padding=False,
    )


# =============================================================================
# RTP-like packet and session metadata
# =============================================================================

@dataclass(frozen=True)
class RTPPacket:
    sequence_number: int
    timestamp: int
    fec_block_id: int
    symbol_index: int
    is_parity: bool
    payload: bytes


@dataclass(frozen=True)
class SessionInfo:
    number_of_tokens: int
    number_of_blocks: int
    source_packets_per_block: int
    parity_packets_per_block: int
    packet_payload_size: int
    interleaver_seed: int

    @property
    def total_packets_per_block(self) -> int:
        return (
            self.source_packets_per_block
            + self.parity_packets_per_block
        )

    @property
    def code_rate(self) -> float:
        return (
            self.source_packets_per_block
            / self.total_packets_per_block
        )


@dataclass
class BlockDecodeReport:
    block_id: int
    total_missing_packets: int
    missing_source_packets_before_rs: int
    rs_attempted: bool
    rs_success: bool
    unresolved_source_packets: int


# =============================================================================
# Deterministic word-level random interleaving
# =============================================================================

def make_permutation(
    size: int,
    shared_seed: int,
    block_id: int,
) -> List[int]:
    """
    Generate a deterministic permutation known to both endpoints.

    The receiver can reproduce it from (shared_seed, block_id), so no explicit
    permutation vector is transmitted in this simulation.
    """
    mixed_seed = (
        (shared_seed & 0xFFFFFFFFFFFFFFFF)
        ^ (
            (block_id + 1)
            * 0x9E3779B97F4A7C15
        )
    ) & 0xFFFFFFFFFFFFFFFF

    rng = random.Random(mixed_seed)
    permutation = list(range(size))
    rng.shuffle(permutation)
    return permutation


def apply_permutation(
    items: Sequence[Tuple[str, bool]],
    permutation: Sequence[int],
) -> List[Tuple[str, bool]]:
    return [items[original_index] for original_index in permutation]


def invert_permutation(
    interleaved_items: Sequence[Optional[str]],
    interleaved_padding: Sequence[bool],
    permutation: Sequence[int],
) -> Tuple[List[Optional[str]], List[bool]]:
    size = len(permutation)
    original_items: List[Optional[str]] = [None] * size
    original_padding: List[bool] = [False] * size

    for interleaved_position, original_position in enumerate(permutation):
        original_items[original_position] = interleaved_items[
            interleaved_position
        ]
        original_padding[original_position] = interleaved_padding[
            interleaved_position
        ]

    return original_items, original_padding


# =============================================================================
# Packet-level systematic Reed-Solomon
# =============================================================================

def encode_text_to_packets(
    tokens: Sequence[str],
    huffman_codec: SharedByteHuffmanCodec,
    source_packets_per_block: int = 4,
    parity_packets_per_block: int = 4,
    packet_payload_size: int = 128,
    interleaver_seed: int = 2025,
) -> Tuple[List[RTPPacket], SessionInfo]:
    """
    One sending window = one RS block.

    Each interleaved token occupies one systematic source packet. Parity
    packets are generated column-wise across fixed-size source payloads.
    """
    k = source_packets_per_block
    p = parity_packets_per_block
    n = k + p

    if k <= 0:
        raise ValueError("source_packets_per_block must be positive.")

    if p <= 0:
        raise ValueError("parity_packets_per_block must be positive.")

    if n > 255:
        raise ValueError("GF(2^8) requires K + P <= 255.")

    number_of_blocks = max(1, math.ceil(len(tokens) / k))
    rs_codec = RSCodec(p)

    packets: List[RTPPacket] = []
    sequence_number = 0

    for block_id in range(number_of_blocks):
        start = block_id * k
        source_window = list(tokens[start : start + k])

        slots: List[Tuple[str, bool]] = [
            (token, False)
            for token in source_window
        ]

        while len(slots) < k:
            slots.append(("", True))

        permutation = make_permutation(
            size=k,
            shared_seed=interleaver_seed,
            block_id=block_id,
        )

        interleaved_slots = apply_permutation(
            slots,
            permutation,
        )

        source_payloads = [
            bytearray(
                pack_token_record(
                    token=token,
                    codec=huffman_codec,
                    packet_payload_size=packet_payload_size,
                    is_padding=is_padding,
                )
            )
            for token, is_padding in interleaved_slots
        ]

        parity_payloads = [
            bytearray(packet_payload_size)
            for _ in range(p)
        ]

        # One RS codeword per byte column.
        for column in range(packet_payload_size):
            message = bytes(
                source_payloads[source_index][column]
                for source_index in range(k)
            )

            codeword = bytes(rs_codec.encode(message))

            if len(codeword) != n:
                raise RuntimeError(
                    "Unexpected RS codeword length: "
                    f"expected={n}, actual={len(codeword)}"
                )

            for parity_index in range(p):
                parity_payloads[parity_index][column] = (
                    codeword[k + parity_index]
                )

        timestamp = start

        for symbol_index, payload in enumerate(source_payloads):
            packets.append(
                RTPPacket(
                    sequence_number=sequence_number,
                    timestamp=timestamp,
                    fec_block_id=block_id,
                    symbol_index=symbol_index,
                    is_parity=False,
                    payload=bytes(payload),
                )
            )
            sequence_number += 1

        for parity_index, payload in enumerate(parity_payloads):
            packets.append(
                RTPPacket(
                    sequence_number=sequence_number,
                    timestamp=timestamp,
                    fec_block_id=block_id,
                    symbol_index=k + parity_index,
                    is_parity=True,
                    payload=bytes(payload),
                )
            )
            sequence_number += 1

    session = SessionInfo(
        number_of_tokens=len(tokens),
        number_of_blocks=number_of_blocks,
        source_packets_per_block=k,
        parity_packets_per_block=p,
        packet_payload_size=packet_payload_size,
        interleaver_seed=interleaver_seed,
    )

    return packets, session


# =============================================================================
# Packet loss channels
# =============================================================================

def iid_packet_loss_channel(
    packets: Sequence[RTPPacket],
    packet_loss_probability: float,
    seed: int = 42,
    shuffle_received_packets: bool = True,
) -> Tuple[List[RTPPacket], List[RTPPacket]]:
    """
    Independent packet erasure channel.
    """
    if not 0.0 <= packet_loss_probability <= 1.0:
        raise ValueError(
            "packet_loss_probability must be in [0, 1]."
        )

    rng = random.Random(seed)
    received: List[RTPPacket] = []
    lost: List[RTPPacket] = []

    for packet in packets:
        if rng.random() < packet_loss_probability:
            lost.append(packet)
        else:
            received.append(packet)

    if shuffle_received_packets:
        rng.shuffle(received)

    return received, lost


def gilbert_elliott_packet_loss_channel(
    packets: Sequence[RTPPacket],
    good_to_bad_probability: float = 0.05,
    bad_to_good_probability: float = 0.20,
    loss_probability_good: float = 0.01,
    loss_probability_bad: float = 0.80,
    initial_state: str = "good",
    seed: int = 42,
    shuffle_received_packets: bool = True,
) -> Tuple[List[RTPPacket], List[RTPPacket]]:
    """
    Two-state Gilbert-Elliott burst packet erasure channel.
    """
    probabilities = [
        good_to_bad_probability,
        bad_to_good_probability,
        loss_probability_good,
        loss_probability_bad,
    ]

    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("All GE probabilities must be in [0, 1].")

    if initial_state not in {"good", "bad"}:
        raise ValueError("initial_state must be 'good' or 'bad'.")

    rng = random.Random(seed)
    state = initial_state
    received: List[RTPPacket] = []
    lost: List[RTPPacket] = []

    for packet in packets:
        loss_probability = (
            loss_probability_good
            if state == "good"
            else loss_probability_bad
        )

        if rng.random() < loss_probability:
            lost.append(packet)
        else:
            received.append(packet)

        if state == "good":
            if rng.random() < good_to_bad_probability:
                state = "bad"
        else:
            if rng.random() < bad_to_good_probability:
                state = "good"

    if shuffle_received_packets:
        rng.shuffle(received)

    return received, lost


# =============================================================================
# Receiver: RS erasure decoding and partial source retention
# =============================================================================

def recover_source_payloads(
    received_packets: Sequence[RTPPacket],
    session: SessionInfo,
) -> Tuple[List[List[Optional[bytes]]], List[BlockDecodeReport]]:
    """
    Recover systematic source packet payloads for every RS block.

    If total block erasures <= P, all source packets are reconstructed through
    RS erasure decoding.

    If total block erasures > P, the function does not abort the sentence.
    It retains every directly received systematic source packet and returns
    None only for unresolved source packet positions.
    """
    k = session.source_packets_per_block
    p = session.parity_packets_per_block
    n = session.total_packets_per_block
    payload_size = session.packet_payload_size

    rs_codec = RSCodec(p)

    packet_map_by_block: Dict[int, Dict[int, RTPPacket]] = defaultdict(dict)

    for packet in received_packets:
        if not 0 <= packet.fec_block_id < session.number_of_blocks:
            raise ValueError(
                f"Invalid block id: {packet.fec_block_id}"
            )

        if not 0 <= packet.symbol_index < n:
            raise ValueError(
                f"Invalid RS symbol index: {packet.symbol_index}"
            )

        if len(packet.payload) != payload_size:
            raise ValueError("Inconsistent packet payload size.")

        block_map = packet_map_by_block[packet.fec_block_id]

        if packet.symbol_index in block_map:
            raise ValueError(
                "Duplicate packet: "
                f"block={packet.fec_block_id}, "
                f"symbol={packet.symbol_index}"
            )

        block_map[packet.symbol_index] = packet

    all_source_payloads: List[List[Optional[bytes]]] = []
    reports: List[BlockDecodeReport] = []

    for block_id in range(session.number_of_blocks):
        block_map = packet_map_by_block[block_id]

        missing_positions = [
            index
            for index in range(n)
            if index not in block_map
        ]

        missing_source_positions = [
            index
            for index in range(k)
            if index not in block_map
        ]

        source_payloads: List[Optional[bytes]] = [
            (
                block_map[index].payload
                if index in block_map
                else None
            )
            for index in range(k)
        ]

        rs_attempted = bool(missing_source_positions)
        rs_success = len(missing_source_positions) == 0

        if missing_source_positions and len(missing_positions) <= p:
            reconstructed = [
                bytearray(payload_size)
                for _ in range(k)
            ]

            try:
                for column in range(payload_size):
                    codeword = bytearray(n)

                    for symbol_index in range(n):
                        packet = block_map.get(symbol_index)
                        codeword[symbol_index] = (
                            packet.payload[column]
                            if packet is not None
                            else 0
                        )

                    decoded_result = rs_codec.decode(
                        bytes(codeword),
                        erase_pos=missing_positions,
                    )

                    decoded_message = (
                        decoded_result[0]
                        if isinstance(decoded_result, tuple)
                        else decoded_result
                    )

                    if len(decoded_message) != k:
                        raise RuntimeError(
                            "Unexpected decoded RS message length: "
                            f"expected={k}, actual={len(decoded_message)}"
                        )

                    for source_index in range(k):
                        reconstructed[source_index][column] = (
                            decoded_message[source_index]
                        )

                source_payloads = [
                    bytes(payload)
                    for payload in reconstructed
                ]
                rs_success = True

            except (ReedSolomonError, RuntimeError, ValueError):
                # Fall back to directly received systematic packets.
                rs_success = False

        unresolved = sum(
            payload is None
            for payload in source_payloads
        )

        reports.append(
            BlockDecodeReport(
                block_id=block_id,
                total_missing_packets=len(missing_positions),
                missing_source_packets_before_rs=(
                    len(missing_source_positions)
                ),
                rs_attempted=rs_attempted,
                rs_success=rs_success,
                unresolved_source_packets=unresolved,
            )
        )

        all_source_payloads.append(source_payloads)

    return all_source_payloads, reports


def decode_source_payloads_to_tokens(
    source_payloads_by_block: Sequence[Sequence[Optional[bytes]]],
    session: SessionInfo,
    huffman_codec: SharedByteHuffmanCodec,
) -> List[Optional[str]]:
    """
    Decode available token records and apply inverse interleaving.

    Unavailable or invalid source records become None and later map to [MASK].
    """
    k = session.source_packets_per_block
    reconstructed_tokens: List[Optional[str]] = []

    for block_id, source_payloads in enumerate(
        source_payloads_by_block
    ):
        if len(source_payloads) != k:
            raise ValueError("Unexpected number of source payloads.")

        interleaved_tokens: List[Optional[str]] = []
        interleaved_padding: List[bool] = []

        for payload in source_payloads:
            if payload is None:
                interleaved_tokens.append(None)
                interleaved_padding.append(False)
                continue

            try:
                record = unpack_token_record(
                    payload,
                    huffman_codec,
                )
                interleaved_tokens.append(record.token)
                interleaved_padding.append(record.is_padding)

            except (
                ValueError,
                UnicodeDecodeError,
            ):
                interleaved_tokens.append(None)
                interleaved_padding.append(False)

        permutation = make_permutation(
            size=k,
            shared_seed=session.interleaver_seed,
            block_id=block_id,
        )

        original_tokens, original_padding = invert_permutation(
            interleaved_items=interleaved_tokens,
            interleaved_padding=interleaved_padding,
            permutation=permutation,
        )

        for token, is_padding in zip(
            original_tokens,
            original_padding,
        ):
            if not is_padding:
                reconstructed_tokens.append(token)

    return reconstructed_tokens[: session.number_of_tokens]


# =============================================================================
# Optional BERT mask concealment
# =============================================================================

class BertMaskFiller:
    """
    Greedy, left-to-right BERT mask filler.

    This is a usable baseline, not a substitute for the paper's domain-finetuned
    TextPC model. A local fine-tuned checkpoint can be supplied as model_name.
    """

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        device: str = "cpu",
        top_k: int = 20,
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForMaskedLM,
                AutoTokenizer,
            )
        except ImportError as error:
            raise ImportError(
                "BERT concealment requires:\n"
                "    pip install torch transformers"
            ) from error

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name
        )
        self.model.eval()
        self.device = device
        self.top_k = max(1, top_k)

        self.model.to(device)

        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                f"Model {model_name!r} has no mask token."
            )

    def _select_candidate(self, logits) -> str:
        top_ids = self.torch.topk(
            logits,
            k=min(self.top_k, logits.shape[-1]),
        ).indices.tolist()

        special_ids = set(
            self.tokenizer.all_special_ids
        )

        for token_id in top_ids:
            if token_id in special_ids:
                continue

            token = self.tokenizer.convert_ids_to_tokens(
                token_id
            )

            if not token:
                continue

            # BERT WordPiece cleanup. RoBERTa-like prefixes are also handled.
            token = token.removeprefix("##")
            token = token.lstrip("Ġ▁")

            if token:
                return token

        return "unknown"

    def fill(
        self,
        tokens: Sequence[Optional[str]],
    ) -> List[str]:
        reconstructed: List[Optional[str]] = list(tokens)

        missing_indices = [
            index
            for index, token in enumerate(reconstructed)
            if token is None
        ]

        for target_index in missing_indices:
            visible_tokens = [
                (
                    token
                    if token is not None
                    else self.tokenizer.mask_token
                )
                for token in reconstructed
            ]

            masked_text = detokenize(visible_tokens)

            encoded = self.tokenizer(
                masked_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            mask_positions = (
                encoded["input_ids"]
                == self.tokenizer.mask_token_id
            ).nonzero(as_tuple=False)

            if mask_positions.numel() == 0:
                reconstructed[target_index] = "unknown"
                continue

            # Missing indices are filled from left to right, so the first
            # remaining [MASK] corresponds to target_index.
            mask_position = int(mask_positions[0, 1])

            with self.torch.no_grad():
                logits = self.model(**encoded).logits[
                    0,
                    mask_position,
                ]

            reconstructed[target_index] = (
                self._select_candidate(logits)
            )

        return [
            token if token is not None else "unknown"
            for token in reconstructed
        ]


# =============================================================================
# Metrics and reporting
# =============================================================================

def token_recovery_accuracy(
    reference: Sequence[str],
    estimate: Sequence[Optional[str]],
) -> float:
    if len(reference) != len(estimate):
        raise ValueError("Reference and estimate lengths differ.")

    if not reference:
        return 1.0

    correct = sum(
        estimated == target
        for target, estimated in zip(reference, estimate)
    )

    return correct / len(reference)


def masked_token_rate(
    tokens: Sequence[Optional[str]],
) -> float:
    if not tokens:
        return 0.0

    return sum(token is None for token in tokens) / len(tokens)


def format_tokens_with_masks(
    tokens: Sequence[Optional[str]],
) -> List[str]:
    return [
        token if token is not None else MASK_TOKEN
        for token in tokens
    ]


def print_block_reports(
    reports: Sequence[BlockDecodeReport],
) -> None:
    print("\nBlock-level decoding report")
    print("-" * 88)
    print(
        f"{'Block':>7} | {'Lost(all)':>9} | "
        f"{'Lost(src)':>9} | {'RS tried':>8} | "
        f"{'RS success':>10} | {'Unresolved':>10}"
    )
    print("-" * 88)

    for report in reports:
        print(
            f"{report.block_id:7d} | "
            f"{report.total_missing_packets:9d} | "
            f"{report.missing_source_packets_before_rs:9d} | "
            f"{str(report.rs_attempted):>8} | "
            f"{str(report.rs_success):>10} | "
            f"{report.unresolved_source_packets:10d}"
        )


# =============================================================================
# End-to-end experiment
# =============================================================================

def run_once(args: argparse.Namespace) -> None:
    text = load_input_text(args)
    tokens = tokenize_text(text)

    if not tokens:
        raise ValueError("No tokens were found in the input text.")

    corpus_texts = load_codebook_corpus(
        args=args,
        fallback_text=text,
    )

    huffman_codec = SharedByteHuffmanCodec(
        corpus_texts=corpus_texts,
        pseudocount=args.huffman_pseudocount,
    )

    packets, session = encode_text_to_packets(
        tokens=tokens,
        huffman_codec=huffman_codec,
        source_packets_per_block=args.k,
        parity_packets_per_block=args.p,
        packet_payload_size=args.payload_size,
        interleaver_seed=args.interleaver_seed,
    )

    if args.loss_model == "iid":
        received, lost = iid_packet_loss_channel(
            packets=packets,
            packet_loss_probability=args.loss_rate,
            seed=args.channel_seed,
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
            seed=args.channel_seed,
            shuffle_received_packets=True,
        )

    source_payloads, reports = recover_source_payloads(
        received_packets=received,
        session=session,
    )

    recovered_tokens = decode_source_payloads_to_tokens(
        source_payloads_by_block=source_payloads,
        session=session,
        huffman_codec=huffman_codec,
    )

    masked_tokens = format_tokens_with_masks(
        recovered_tokens
    )

    print("=" * 88)
    print("Paper-style Huffman-RS packet-loss experiment")
    print("=" * 88)
    print(f"Original text            : {text}")
    print(f"Number of tokens         : {len(tokens)}")
    print(
        "RS configuration         : "
        f"K={session.source_packets_per_block}, "
        f"P={session.parity_packets_per_block}, "
        f"N={session.total_packets_per_block}, "
        f"rate={session.code_rate:.4f}"
    )
    print(
        f"Packet payload           : "
        f"{session.packet_payload_size} bytes"
    )
    print(f"Number of RS blocks      : {session.number_of_blocks}")
    print(f"Transmitted packets      : {len(packets)}")
    print(f"Received packets         : {len(received)}")
    print(f"Lost packets             : {len(lost)}")
    print(
        f"Realized packet loss rate: "
        f"{len(lost) / len(packets):.4%}"
    )
    print(f"Loss model               : {args.loss_model}")

    print_block_reports(reports)

    before_accuracy = token_recovery_accuracy(
        reference=tokens,
        estimate=recovered_tokens,
    )

    print("\nReceiver output before BERT")
    print("-" * 88)
    print(detokenize(masked_tokens))
    print(
        f"Exact token recovery     : {before_accuracy:.4%}"
    )
    print(
        f"Unresolved token rate    : "
        f"{masked_token_rate(recovered_tokens):.4%}"
    )

    if args.use_bert:
        filler = BertMaskFiller(
            model_name=args.bert_model,
            device=args.bert_device,
            top_k=args.bert_top_k,
        )

        bert_tokens = filler.fill(recovered_tokens)

        after_accuracy = token_recovery_accuracy(
            reference=tokens,
            estimate=bert_tokens,
        )

        print("\nReceiver output after BERT concealment")
        print("-" * 88)
        print(detokenize(bert_tokens))
        print(
            f"Exact token accuracy     : {after_accuracy:.4%}"
        )
        print(
            "Note: this is an off-the-shelf/fine-tuned-checkpoint "
            "inference result, not the paper's unpublished checkpoint."
        )


def load_input_text(args: argparse.Namespace) -> str:
    if args.text_file is not None:
        path = Path(args.text_file)
        if not path.exists():
            raise FileNotFoundError(path)
        return path.read_text(
            encoding=args.text_encoding
        ).strip()

    return args.text


def load_codebook_corpus(
    args: argparse.Namespace,
    fallback_text: str,
) -> List[str]:
    if args.codebook_corpus is None:
        # Demonstration setting. In a real experiment, use a fixed training
        # corpus shared by both endpoints.
        return [fallback_text]

    path = Path(args.codebook_corpus)

    if not path.exists():
        raise FileNotFoundError(path)

    return [
        path.read_text(
            encoding=args.text_encoding
        )
    ]


# =============================================================================
# Deterministic boundary tests
# =============================================================================

def drop_exact_packet_positions(
    packets: Sequence[RTPPacket],
    positions_by_block: Dict[int, Sequence[int]],
) -> Tuple[List[RTPPacket], List[RTPPacket]]:
    loss_set = {
        (block_id, symbol_index)
        for block_id, symbol_indices in positions_by_block.items()
        for symbol_index in symbol_indices
    }

    received: List[RTPPacket] = []
    lost: List[RTPPacket] = []

    for packet in packets:
        key = (
            packet.fec_block_id,
            packet.symbol_index,
        )

        if key in loss_set:
            lost.append(packet)
        else:
            received.append(packet)

    return received, lost


def run_boundary_tests() -> None:
    """
    Validate the key behavior:
      - <= P total erasures: full RS recovery
      - > P erasures: retain received source packets and mask unresolved words
    """
    text = (
        "BERT predicts missing words while Reed Solomon "
        "recovers erased packets from redundant symbols."
    )

    tokens = tokenize_text(text)
    codec = SharedByteHuffmanCodec([text])

    packets, session = encode_text_to_packets(
        tokens=tokens,
        huffman_codec=codec,
        source_packets_per_block=4,
        parity_packets_per_block=4,
        packet_payload_size=128,
        interleaver_seed=2025,
    )

    test_cases = [
        (
            "No loss",
            {},
            0,
        ),
        (
            "Exactly P=4 erasures in block 0",
            {0: [0, 2, 4, 7]},
            0,
        ),
        (
            "P+1=5 erasures including 3 source packets",
            {0: [0, 1, 2, 4, 7]},
            3,
        ),
    ]

    print("=" * 88)
    print("Deterministic RS boundary tests")
    print("=" * 88)

    for name, loss_map, expected_masks_in_first_block in test_cases:
        received, _ = drop_exact_packet_positions(
            packets,
            loss_map,
        )

        payloads, reports = recover_source_payloads(
            received,
            session,
        )

        estimate = decode_source_payloads_to_tokens(
            payloads,
            session,
            codec,
        )

        actual_masks_first_block = sum(
            token is None
            for token in estimate[:4]
        )

        if actual_masks_first_block != expected_masks_in_first_block:
            raise AssertionError(
                f"{name}: expected "
                f"{expected_masks_in_first_block} unresolved tokens "
                f"in block 0, got {actual_masks_first_block}."
            )

        print(
            f"[PASS] {name}: "
            f"unresolved tokens in block 0 = "
            f"{actual_masks_first_block}, "
            f"RS success = {reports[0].rs_success}"
        )


# =============================================================================
# CLI
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Word-interleaved Huffman-RS packet-loss text "
            "transmission with optional BERT concealment."
        )
    )

    parser.add_argument(
        "--text",
        type=str,
        default=(
            "The transmitter randomly interleaves words before "
            "Huffman and Reed Solomon coding. The receiver "
            "uses context to predict words that remain missing."
        ),
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--text_encoding",
        type=str,
        default="utf-8",
    )
    parser.add_argument(
        "--codebook_corpus",
        type=str,
        default=None,
        help=(
            "Shared corpus used to construct the Huffman codebook. "
            "If omitted, the input text is used for demonstration."
        ),
    )
    parser.add_argument(
        "--huffman_pseudocount",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Systematic source packets/words per RS block.",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=4,
        help="Parity packets per RS block.",
    )
    parser.add_argument(
        "--payload_size",
        type=int,
        default=128,
        help="Fixed packet payload size in bytes.",
    )
    parser.add_argument(
        "--interleaver_seed",
        type=int,
        default=2025,
    )

    parser.add_argument(
        "--loss_model",
        choices=["iid", "ge"],
        default="iid",
    )
    parser.add_argument(
        "--loss_rate",
        type=float,
        default=0.5,
        help="IID packet-loss probability.",
    )
    parser.add_argument(
        "--channel_seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--ge_good_to_bad",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--ge_bad_to_good",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--ge_loss_good",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--ge_loss_bad",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--ge_initial_state",
        choices=["good", "bad"],
        default="good",
    )

    parser.add_argument(
        "--use_bert",
        action="store_true",
    )
    parser.add_argument(
        "--bert_model",
        type=str,
        default="bert-base-cased",
        help=(
            "Hugging Face model name or local fine-tuned "
            "TextPC checkpoint directory."
        ),
    )
    parser.add_argument(
        "--bert_device",
        type=str,
        default="cpu",
        help="Examples: cpu, cuda, cuda:0",
    )
    parser.add_argument(
        "--bert_top_k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--run_boundary_tests",
        action="store_true",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.run_boundary_tests:
        run_boundary_tests()
        print()

    run_once(args)


if __name__ == "__main__":
    main()
