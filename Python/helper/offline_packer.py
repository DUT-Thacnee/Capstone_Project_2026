#!/usr/bin/env python3
"""Offline packer for the command-driven CNN HLS core.

This script demonstrates how to:
- take int8 weights (mocked here) and int32 biases
- compress weights using (distance,value) RLE pairs (8-bit each)
- serialize a single binary blob with weights + biases
- emit metadata with byte offsets and nnz (pair count)

RLE format used here (per pair):
  uint8 distance   # number of zeros to skip before placing value
  int8  value      # non-zero weight value (may be 0 for long zero runs)

Decoder behavior (to reconstruct dense weights):
  pos = 0
  for (distance, value) in pairs:
      pos += distance
      dense[pos] = value
      pos += 1
  remaining elements up to dense_len are zeros

If a zero-run is longer than 255, this script emits extra pairs with
value=0 to advance the position safely while preserving correctness.

NOTE:
- This format is *not* the same as the current HLS skip-list parser in
  hardware_functions.cpp. Only use if your hardware decoder matches this
  pair-based RLE format.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


HELPER_DIR = Path(__file__).resolve().parent


def flatten_weights_for_1x1(weights: np.ndarray) -> np.ndarray:
    """Flatten 1x1 weights to the order expected by the HLS core.

    HLS expects:
      for oc in Cout:
        for ic in Cin:
          read weight

    Accepts weights shaped as:
      (Cout, Cin) or (Cout, Cin, 1, 1)
    """

    if weights.ndim == 4:
        if weights.shape[2:] != (1, 1):
            raise ValueError("Expected 1x1 weights with shape (Cout, Cin, 1, 1)")
        w = weights[:, :, 0, 0]
    elif weights.ndim == 2:
        w = weights
    else:
        raise ValueError("Expected weights with shape (Cout, Cin) or (Cout, Cin, 1, 1)")

    return np.asarray(w, dtype=np.int8).reshape(-1)


def flatten_weights_for_dw3x3(weights: np.ndarray) -> np.ndarray:
    """Flatten depthwise 3x3 weights to the order expected by the HLS core.

    HLS expects:
      for c in C:
        for k in 0..8 (r*3 + c)
          read weight

    Accepts weights shaped as:
      (C, 3, 3) or (C, 1, 3, 3)
    """

    if weights.ndim == 4:
        if weights.shape[1:] != (1, 3, 3):
            raise ValueError("Expected DW 3x3 weights with shape (C, 1, 3, 3)")
        w = weights[:, 0, :, :]
    elif weights.ndim == 3:
        if weights.shape[1:] != (3, 3):
            raise ValueError("Expected DW 3x3 weights with shape (C, 3, 3)")
        w = weights
    else:
        raise ValueError("Expected weights with shape (C, 3, 3) or (C, 1, 3, 3)")

    return np.asarray(w, dtype=np.int8).reshape(-1)


def rle_encode_pairs(values: np.ndarray) -> Tuple[List[Tuple[int, int]], int]:
    """Encode a 1D int8 array into (distance,value) RLE pairs.

    Returns:
      pairs: list of (distance, value)
      dense_len: number of original dense values
    """

    values = np.asarray(values, dtype=np.int8).reshape(-1)
    dense_len = int(values.size)

    pairs: List[Tuple[int, int]] = []
    zero_run = 0

    for v in values:
        if v == 0:
            zero_run += 1
            if zero_run == 255:
                # Emit a "dummy" zero value to advance by 256 positions.
                pairs.append((255, 0))
                zero_run = 0
            continue

        pairs.append((zero_run, int(v)))
        zero_run = 0

    # trailing zeros are implicit; dense_len tells decoder how many to fill.
    return pairs, dense_len


def pack_pairs_to_bytes(pairs: List[Tuple[int, int]]) -> bytes:
    """Pack (distance,value) pairs into a raw byte stream (little-endian).

    Each pair becomes 2 bytes: [distance][value].
    """

    out = bytearray()
    for dist, val in pairs:
        if dist < 0 or dist > 255:
            raise ValueError(f"distance out of range: {dist}")
        if val < -128 or val > 127:
            raise ValueError(f"value out of range: {val}")
        out.extend(struct.pack("<Bb", int(dist), int(val)))
    return bytes(out)


def pack_biases_i32(biases: np.ndarray) -> bytes:
    biases = np.asarray(biases, dtype=np.int32).reshape(-1)
    out = bytearray()
    for b in biases:
        out.extend(struct.pack("<i", int(b)))
    return bytes(out)


def align_offset(offset: int, alignment: int) -> int:
    if alignment <= 1:
        return offset
    return (offset + alignment - 1) // alignment * alignment


def build_blob_and_meta(
    *,
    rle_bytes: bytes,
    bias_bytes: bytes,
    dense_len: int,
    rle_pairs: int,
    alignment: int = 8,
) -> Tuple[bytes, Dict[str, int]]:
    """Return (binary_blob, metadata) for a single layer."""

    offset_w = 0
    offset_b = align_offset(len(rle_bytes), alignment)

    padding = b"\x00" * (offset_b - len(rle_bytes))
    blob = rle_bytes + padding + bias_bytes

    meta = {
        "weights_offset": int(offset_w),
        "weights_bytes": int(len(rle_bytes)),
        "bias_offset": int(offset_b),
        "bias_bytes": int(len(bias_bytes)),
        "rle_pairs": int(rle_pairs),
        "dense_len": int(dense_len),
        "format": "rle8x8",
        "alignment": int(alignment),
    }
    return blob, meta


def mock_conv1x1_weights(cout: int = 16, cin: int = 8, sparsity: float = 0.7, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.integers(-8, 8, size=(cout, cin), dtype=np.int8)
    mask = rng.random(w.shape) < float(sparsity)
    w[mask] = 0
    b = rng.integers(-5000, 5000, size=(cout,), dtype=np.int32)
    return w, b


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline packer for HLS CNN weights/biases")
    parser.add_argument("--out_bin", default=str(HELPER_DIR / "weights_and_biases.bin"), help="Output binary file")
    parser.add_argument("--out_meta", default=str(HELPER_DIR / "weights_and_biases.json"), help="Output metadata JSON")
    parser.add_argument("--alignment", type=int, default=8, help="Alignment (bytes) for bias offset")
    args = parser.parse_args()

    # ---- Dummy layer: 1x1 conv ----
    weights, biases = mock_conv1x1_weights(cout=16, cin=8, sparsity=0.75, seed=1)

    dense = flatten_weights_for_1x1(weights)
    pairs, dense_len = rle_encode_pairs(dense)
    rle_bytes = pack_pairs_to_bytes(pairs)
    bias_bytes = pack_biases_i32(biases)

    blob, layer_meta = build_blob_and_meta(
        rle_bytes=rle_bytes,
        bias_bytes=bias_bytes,
        dense_len=dense_len,
        rle_pairs=len(pairs),
        alignment=int(args.alignment),
    )

    # ---- Emit outputs ----
    with open(args.out_bin, "wb") as f:
        f.write(blob)

    meta = {
        "file": args.out_bin,
        "layers": {
            "dummy_conv1x1": {
                **layer_meta,
                "cout": int(weights.shape[0]),
                "cin": int(weights.shape[1]),
                "kernel": [1, 1],
            }
        },
    }

    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {args.out_bin} ({len(blob)} bytes)")
    print(f"Wrote {args.out_meta}")
    print(f"RLE pairs: {len(pairs)}, dense_len: {dense_len}")


if __name__ == "__main__":
    main()
