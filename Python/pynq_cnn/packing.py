from __future__ import annotations

from typing import Tuple

import numpy as np

from .commands import axis_words_exact


def axis_words_for_nhwc_i8(h: int, w: int, c: int, axis_bytes: int = 4) -> int:
    return axis_words_exact(int(h) * int(w) * int(c), axis_bytes)


def pack_nhwc_i8_to_u32_words(x_nhwc_i8: np.ndarray, out_words_u32: np.ndarray) -> None:
    if x_nhwc_i8.dtype != np.int8:
        raise TypeError(f"Expected int8 tensor, got {x_nhwc_i8.dtype}")
    if not x_nhwc_i8.flags["C_CONTIGUOUS"]:
        x_nhwc_i8 = np.ascontiguousarray(x_nhwc_i8)

    flat_u8 = x_nhwc_i8.view(np.uint8).reshape(-1)
    nbytes = int(flat_u8.size)
    nwords = axis_words_exact(nbytes, 4)

    if out_words_u32.dtype != np.uint32:
        raise TypeError(f"out_words_u32 must be uint32, got {out_words_u32.dtype}")
    if out_words_u32.ndim != 1 or int(out_words_u32.size) != nwords:
        raise ValueError(f"out_words_u32 must be uint32[{nwords}]")

    out_u8 = out_words_u32.view(np.uint8)
    out_u8[:nbytes] = flat_u8


def pack_nhwc_i8_to_u32_words_cpad(
    x_nhwc_i8: np.ndarray,
    *,
    c_padded: int,
    out_words_u32: np.ndarray,
) -> None:
    if x_nhwc_i8.dtype != np.int8:
        raise TypeError(f"Expected int8 tensor, got {x_nhwc_i8.dtype}")
    if x_nhwc_i8.ndim != 3:
        raise ValueError("Expected NHWC with shape (H,W,C)")

    H, W, C = (int(x_nhwc_i8.shape[0]), int(x_nhwc_i8.shape[1]), int(x_nhwc_i8.shape[2]))
    Cp = int(c_padded)
    if Cp < C:
        raise ValueError("c_padded must be >= C")

    nbytes = H * W * Cp
    nwords = axis_words_exact(nbytes, 4)
    if out_words_u32.dtype != np.uint32 or out_words_u32.ndim != 1 or int(out_words_u32.size) != nwords:
        raise ValueError(f"out_words_u32 must be uint32[{nwords}]")

    out_bytes = out_words_u32.view(np.uint8)[:nbytes]
    out_nhwc = out_bytes.view(np.int8).reshape(H, W, Cp)
    out_nhwc.fill(0)
    out_nhwc[:, :, :C] = x_nhwc_i8


def unpack_u32_words_to_nhwc_i8(words_u32: np.ndarray, shape_hwc: Tuple[int, int, int]) -> np.ndarray:
    if words_u32.dtype != np.uint32:
        raise TypeError(f"words_u32 must be uint32, got {words_u32.dtype}")

    h, w, c = (int(shape_hwc[0]), int(shape_hwc[1]), int(shape_hwc[2]))
    nbytes = h * w * c
    nwords = axis_words_exact(nbytes, 4)
    if int(words_u32.size) < nwords:
        raise ValueError("Not enough words for requested shape")

    b = words_u32[:nwords].view(np.uint8)[:nbytes]
    return b.view(np.int8).reshape(h, w, c)


def unpack_u32_words_to_nhwc_i8_clogical(
    words_u32: np.ndarray,
    *,
    h: int,
    w: int,
    c_padded: int,
    c_logical: int,
) -> np.ndarray:
    t = unpack_u32_words_to_nhwc_i8(words_u32, (int(h), int(w), int(c_padded)))
    return t[:, :, : int(c_logical)]


def align_channels_for_axis_and_simd(c: int, axis_bytes: int = 4, simd: int = 8) -> int:
    c = int(c)
    a = int(axis_bytes)
    s = int(simd)

    def round_up(x: int, r: int) -> int:
        return ((x + r - 1) // r) * r

    c4 = round_up(c, a)
    return round_up(c4, s)
