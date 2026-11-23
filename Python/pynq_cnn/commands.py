from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

# Constants mirrored from Hardware/HLS/src_hw/parameters.h
CNN_OP_CONV2D = 0
CNN_OP_AVGPOOL = 1
CNN_OP_NOP = 255

CNN_LAYOUT_NHWC = 0

CNN_W_DENSE = 0
CNN_W_SPARSE_RLE = 1

CNN_ACT_NONE = 0
CNN_ACT_RELU = 1
CNN_ACT_RELU6 = 2

CNN_FLAG_IN_AXIS = 1 << 0
CNN_FLAG_OUT_AXIS = 1 << 1
CNN_FLAG_FUSE_BIAS = 1 << 2
CNN_FLAG_FUSE_RESIDUAL = 1 << 3
CNN_FLAG_ENABLE_SPARSE = 1 << 4


def _u32(x: int) -> int:
    return int(x) & 0xFFFFFFFF


def _i32(x: int) -> int:
    x = int(x)
    if x < -(1 << 31) or x > (1 << 31) - 1:
        raise ValueError(f"i32 overflow: {x}")
    return x


def _i8(x: int) -> int:
    x = int(x)
    if x < -128 or x > 127:
        raise ValueError(f"i8 overflow: {x}")
    return x


def _u8_bits(x: int) -> int:
    return int(x) & 0xFF


def _u16_bits(x: int) -> int:
    return int(x) & 0xFFFF


def _u64_words(x: int) -> tuple[int, int]:
    x = int(x) & 0xFFFFFFFFFFFFFFFF
    return x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF


def _pack_u8s(*values: int) -> int:
    out = 0
    for i, value in enumerate(values):
        out |= _u8_bits(value) << (8 * i)
    return _u32(out)


def _pack_u16s(lo: int, hi: int) -> int:
    return _u32(_u16_bits(lo) | (_u16_bits(hi) << 16))


def axis_words_exact(byte_count: int, axis_bytes: int) -> int:
    if byte_count % axis_bytes != 0:
        raise ValueError(f"Byte count {byte_count} is not a multiple of axis_bytes={axis_bytes}.")
    return byte_count // axis_bytes


@dataclass
class TensorDesc:
    addr: int = 0
    h: int = 0
    w: int = 0
    c: int = 0
    stride_h: int = 0
    stride_w: int = 0
    layout: int = CNN_LAYOUT_NHWC
    elem_bytes: int = 1

    @staticmethod
    def nhwc_i8(h: int, w: int, c: int) -> "TensorDesc":
        return TensorDesc(h=h, w=w, c=c, stride_w=c, stride_h=w * c, layout=CNN_LAYOUT_NHWC, elem_bytes=1)

    def byte_count(self) -> int:
        return int(self.h) * int(self.w) * int(self.c) * int(self.elem_bytes)


@dataclass
class Conv2DDesc:
    kernel_h: int = 1
    kernel_w: int = 1
    stride_h: int = 1
    stride_w: int = 1
    pad_top: int = 0
    pad_left: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    groups: int = 1


@dataclass
class QuantDesc:
    out_multiplier: int = 1
    out_shift: int = 0
    in_zero_point: int = 0
    w_zero_point: int = 0
    out_zero_point: int = 0
    act_min: int = -128
    act_max: int = 127

    def validate(self) -> None:
        _i32(self.out_multiplier)
        _i8(self.out_shift)
        _i8(self.in_zero_point)
        _i8(self.w_zero_point)
        _i8(self.out_zero_point)
        _i8(self.act_min)
        _i8(self.act_max)


@dataclass
class SparseRleDesc:
    w_nz_addr: int = 0
    skip_addr: int = 0
    nnz: int = 0
    skip_bits: int = 16


@dataclass
class LayerCmd:
    opcode: int
    flags: int
    in_words: int
    out_words: int

    src: TensorDesc
    dst: TensorDesc
    residual: TensorDesc = field(default_factory=TensorDesc)

    w_addr: int = 0
    b_addr: int = 0
    w_bytes: int = 0
    b_bytes: int = 0

    w_format: int = CNN_W_DENSE
    activation: int = CNN_ACT_NONE

    conv: Conv2DDesc = field(default_factory=Conv2DDesc)
    q: QuantDesc = field(default_factory=QuantDesc)
    sparse: SparseRleDesc = field(default_factory=SparseRleDesc)

    user_tag: int = 0

    def validate_axis_counts(self, axis_bytes: int = 4) -> None:
        if self.src.elem_bytes != 1 or self.dst.elem_bytes != 1:
            raise ValueError("Expected int8 activations (elem_bytes=1)")
        exp_in = self.src.byte_count()
        exp_out = self.dst.byte_count()
        if self.in_words * axis_bytes != exp_in:
            raise ValueError(f"in_words mismatch: {self.in_words}*{axis_bytes} != {exp_in}")
        if self.out_words * axis_bytes != exp_out:
            raise ValueError(f"out_words mismatch: {self.out_words}*{axis_bytes} != {exp_out}")

    def requires_residual_stream(self) -> bool:
        return (self.flags & CNN_FLAG_FUSE_RESIDUAL) != 0

    def as_flat_register_dict(self, prefix: str = "cmd") -> Dict[str, int]:
        self.q.validate()

        p = prefix
        d: Dict[str, int] = {}

        d[f"{p}_opcode"] = _u32(self.opcode)
        d[f"{p}_flags"] = _u32(self.flags)
        d[f"{p}_in_words"] = _u32(self.in_words)
        d[f"{p}_out_words"] = _u32(self.out_words)

        for name, t in (("src", self.src), ("dst", self.dst), ("residual", self.residual)):
            d[f"{p}_{name}_addr"] = _u32(t.addr)
            d[f"{p}_{name}_h"] = _u32(t.h)
            d[f"{p}_{name}_w"] = _u32(t.w)
            d[f"{p}_{name}_c"] = _u32(t.c)
            d[f"{p}_{name}_stride_h"] = _u32(t.stride_h)
            d[f"{p}_{name}_stride_w"] = _u32(t.stride_w)
            d[f"{p}_{name}_layout"] = _u32(t.layout)
            d[f"{p}_{name}_elem_bytes"] = _u32(t.elem_bytes)

        d[f"{p}_w_addr"] = _u32(self.w_addr)
        d[f"{p}_b_addr"] = _u32(self.b_addr)
        d[f"{p}_w_bytes"] = _u32(self.w_bytes)
        d[f"{p}_b_bytes"] = _u32(self.b_bytes)
        d[f"{p}_w_format"] = _u32(self.w_format)
        d[f"{p}_activation"] = _u32(self.activation)

        d[f"{p}_conv_kernel_h"] = _u32(self.conv.kernel_h)
        d[f"{p}_conv_kernel_w"] = _u32(self.conv.kernel_w)
        d[f"{p}_conv_stride_h"] = _u32(self.conv.stride_h)
        d[f"{p}_conv_stride_w"] = _u32(self.conv.stride_w)
        d[f"{p}_conv_pad_top"] = _u32(self.conv.pad_top & 0xFFFF_FFFF)
        d[f"{p}_conv_pad_left"] = _u32(self.conv.pad_left & 0xFFFF_FFFF)
        d[f"{p}_conv_dilation_h"] = _u32(self.conv.dilation_h)
        d[f"{p}_conv_dilation_w"] = _u32(self.conv.dilation_w)
        d[f"{p}_conv_groups"] = _u32(self.conv.groups)

        d[f"{p}_q_out_multiplier"] = _u32(self.q.out_multiplier)
        d[f"{p}_q_out_shift"] = _u32(self.q.out_shift & 0xFF)
        d[f"{p}_q_in_zero_point"] = _u32(self.q.in_zero_point & 0xFF)
        d[f"{p}_q_w_zero_point"] = _u32(self.q.w_zero_point & 0xFF)
        d[f"{p}_q_out_zero_point"] = _u32(self.q.out_zero_point & 0xFF)
        d[f"{p}_q_act_min"] = _u32(self.q.act_min & 0xFF)
        d[f"{p}_q_act_max"] = _u32(self.q.act_max & 0xFF)

        d[f"{p}_sparse_w_nz_addr"] = _u32(self.sparse.w_nz_addr)
        d[f"{p}_sparse_skip_addr"] = _u32(self.sparse.skip_addr)
        d[f"{p}_sparse_nnz"] = _u32(self.sparse.nnz)
        d[f"{p}_sparse_skip_bits"] = _u32(self.sparse.skip_bits)

        d[f"{p}_user_tag"] = _u32(self.user_tag)
        return d

    def as_register_words(self) -> list[int]:
        """Pack cnn_layer_cmd_t exactly as exposed by the HLS AXI-Lite register map."""
        self.q.validate()

        def tensor_words(t: TensorDesc) -> list[int]:
            addr_lo, addr_hi = _u64_words(t.addr)
            return [
                addr_lo,
                addr_hi,
                _u32(t.h),
                _u32(t.w),
                _u32(t.c),
                _u32(t.stride_h),
                _u32(t.stride_w),
                _pack_u8s(t.layout, t.elem_bytes, 0, 0),
            ]

        w_addr_lo, w_addr_hi = _u64_words(self.w_addr)
        b_addr_lo, b_addr_hi = _u64_words(self.b_addr)
        w_nz_addr_lo, w_nz_addr_hi = _u64_words(self.sparse.w_nz_addr)
        skip_addr_lo, skip_addr_hi = _u64_words(self.sparse.skip_addr)

        words = [
            _u32(self.opcode),
            _u32(self.flags),
            _u32(self.in_words),
            _u32(self.out_words),
            *tensor_words(self.src),
            *tensor_words(self.dst),
            *tensor_words(self.residual),
            w_addr_lo,
            w_addr_hi,
            b_addr_lo,
            b_addr_hi,
            _u32(self.w_bytes),
            _u32(self.b_bytes),
            _pack_u8s(self.w_format, self.activation, 0, 0),
            _pack_u16s(self.conv.kernel_h, self.conv.kernel_w),
            _pack_u16s(self.conv.stride_h, self.conv.stride_w),
            _pack_u16s(self.conv.pad_top, self.conv.pad_left),
            _pack_u16s(self.conv.dilation_h, self.conv.dilation_w),
            _u32(self.conv.groups),
            _u32(self.q.out_multiplier),
            _pack_u8s(self.q.out_shift, self.q.in_zero_point, self.q.w_zero_point, self.q.out_zero_point),
            _pack_u8s(self.q.act_min, self.q.act_max, 0, 0),
            0,
            w_nz_addr_lo,
            w_nz_addr_hi,
            skip_addr_lo,
            skip_addr_hi,
            _u32(self.sparse.nnz),
            _u32(self.sparse.skip_bits),
            _u32(self.user_tag),
            0,
        ]

        if len(words) != 52:
            raise RuntimeError(f"Internal command packing error: expected 52 words, got {len(words)}")
        return words
