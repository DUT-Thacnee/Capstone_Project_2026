from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Tuple

import numpy as np

from .accelerator import AxisTensor, CnnAccelerator, PingPong
from .commands import (
    Conv2DDesc,
    LayerCmd,
    QuantDesc,
    TensorDesc,
    CNN_ACT_NONE,
    CNN_FLAG_FUSE_BIAS,
    CNN_FLAG_FUSE_RESIDUAL,
    CNN_FLAG_IN_AXIS,
    CNN_FLAG_OUT_AXIS,
    CNN_OP_CONV2D,
    axis_words_exact,
)
from .packing import align_channels_for_axis_and_simd


@dataclass
class IrBlockSpec:
    h: int
    w: int
    cin: int
    expansion_factor: int = 6
    stride: int = 1
    cin_hw: int = 0
    cexp_hw: int = 0

    @property
    def cin_padded(self) -> int:
        return int(self.cin_hw) if self.cin_hw else int(self.cin)

    @property
    def cexp_padded(self) -> int:
        if self.cexp_hw:
            return int(self.cexp_hw)
        return int(self.cin_padded) * int(self.expansion_factor)

    @property
    def cout_padded(self) -> int:
        return int(self.cin_padded)


@dataclass
class IrBlockWeights:
    w_exp: int
    b_exp: int
    w_dw: int
    b_dw: int
    w_proj: int
    b_proj: int


@dataclass
class IrBlockQuant:
    q_exp: QuantDesc
    q_dw: QuantDesc
    q_proj: QuantDesc


def build_ir_block_cmds(
    *,
    spec: IrBlockSpec,
    w: IrBlockWeights,
    q: IrBlockQuant,
    axis_bytes: int = 4,
) -> Tuple[LayerCmd, LayerCmd, LayerCmd]:
    H, W = int(spec.h), int(spec.w)
    Cin = int(spec.cin_padded)
    Cexp = int(spec.cexp_padded)
    Cout = int(spec.cout_padded)

    if int(spec.stride) != 1:
        raise NotImplementedError("Only stride=1 residual blocks are shown here")

    exp_src = TensorDesc.nhwc_i8(H, W, Cin)
    exp_dst = TensorDesc.nhwc_i8(H, W, Cexp)
    exp_cmd = LayerCmd(
        opcode=CNN_OP_CONV2D,
        flags=CNN_FLAG_IN_AXIS | CNN_FLAG_OUT_AXIS | CNN_FLAG_FUSE_BIAS,
        in_words=axis_words_exact(exp_src.byte_count(), axis_bytes),
        out_words=axis_words_exact(exp_dst.byte_count(), axis_bytes),
        src=exp_src,
        dst=exp_dst,
        w_addr=int(w.w_exp),
        b_addr=int(w.b_exp),
        w_format=0,
        activation=CNN_ACT_NONE,
        conv=Conv2DDesc(kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, groups=1),
        q=q.q_exp,
    )

    Hpad, Wpad = H + 2, W + 2
    dw_src = TensorDesc.nhwc_i8(Hpad, Wpad, Cexp)
    dw_dst = TensorDesc.nhwc_i8(H, W, Cexp)
    dw_cmd = LayerCmd(
        opcode=CNN_OP_CONV2D,
        flags=CNN_FLAG_IN_AXIS | CNN_FLAG_OUT_AXIS | CNN_FLAG_FUSE_BIAS,
        in_words=axis_words_exact(dw_src.byte_count(), axis_bytes),
        out_words=axis_words_exact(dw_dst.byte_count(), axis_bytes),
        src=dw_src,
        dst=dw_dst,
        w_addr=int(w.w_dw),
        b_addr=int(w.b_dw),
        w_format=0,
        activation=CNN_ACT_NONE,
        conv=Conv2DDesc(kernel_h=3, kernel_w=3, stride_h=1, stride_w=1, groups=Cexp),
        q=q.q_dw,
    )

    proj_src = TensorDesc.nhwc_i8(H, W, Cexp)
    proj_dst = TensorDesc.nhwc_i8(H, W, Cout)
    proj_cmd = LayerCmd(
        opcode=CNN_OP_CONV2D,
        flags=CNN_FLAG_IN_AXIS | CNN_FLAG_OUT_AXIS | CNN_FLAG_FUSE_BIAS | CNN_FLAG_FUSE_RESIDUAL,
        in_words=axis_words_exact(proj_src.byte_count(), axis_bytes),
        out_words=axis_words_exact(proj_dst.byte_count(), axis_bytes),
        src=proj_src,
        dst=proj_dst,
        residual=TensorDesc.nhwc_i8(H, W, Cout),
        w_addr=int(w.w_proj),
        b_addr=int(w.b_proj),
        w_format=0,
        activation=CNN_ACT_NONE,
        conv=Conv2DDesc(kernel_h=1, kernel_w=1, stride_h=1, stride_w=1, groups=1),
        q=q.q_proj,
    )

    exp_cmd.validate_axis_counts(axis_bytes=axis_bytes)
    dw_cmd.validate_axis_counts(axis_bytes=axis_bytes)
    proj_cmd.validate_axis_counts(axis_bytes=axis_bytes)

    return exp_cmd, dw_cmd, proj_cmd


def run_ir_block(
    *,
    accel: CnnAccelerator,
    spec: IrBlockSpec,
    cmds: Tuple[LayerCmd, LayerCmd, LayerCmd],
    pingpong: PingPong,
    scratch_exp_dw: AxisTensor,
    scratch_dwpad: AxisTensor,
    x_in_nhwc_i8: np.ndarray,
    profile: dict[str, float] | None = None,
    profile_prefix: str = "",
) -> np.ndarray:
    H, W = int(spec.h), int(spec.w)
    Cin = int(spec.cin)
    Cin_hw = int(spec.cin_padded)
    Cexp_hw = int(spec.cexp_padded)

    if x_in_nhwc_i8.dtype != np.int8 or x_in_nhwc_i8.shape != (H, W, Cin):
        raise ValueError(f"x_in_nhwc_i8 must be int8 shape {(H, W, Cin)}")

    # Pack input into ping (padded if required)
    pingpong.ping.write_from_nhwc_i8(x_in_nhwc_i8)

    # 1x1 expansion
    t0 = time.perf_counter()
    accel.run_layer_tensor(cmd=cmds[0], in_tensor=pingpong.ping, out_tensor=scratch_exp_dw)
    if profile is not None:
        profile[f"{profile_prefix}expand_ms"] = (time.perf_counter() - t0) * 1000.0

    # Host-side SAME padding for depthwise valid
    t0 = time.perf_counter()
    exp_view = scratch_exp_dw.view_nhwc_i8().reshape(H, W, Cexp_hw)
    pad_view = scratch_dwpad.view_nhwc_i8().reshape(H + 2, W + 2, Cexp_hw)
    pad_view.fill(0)
    pad_view[1 : 1 + H, 1 : 1 + W, :] = exp_view
    if profile is not None:
        profile[f"{profile_prefix}padding_ms"] = (time.perf_counter() - t0) * 1000.0

    # Depthwise
    t0 = time.perf_counter()
    accel.run_layer_tensor(cmd=cmds[1], in_tensor=scratch_dwpad, out_tensor=scratch_exp_dw)
    if profile is not None:
        profile[f"{profile_prefix}depthwise_ms"] = (time.perf_counter() - t0) * 1000.0

    # Projection + residual
    t0 = time.perf_counter()
    accel.run_layer_tensor(cmd=cmds[2], in_tensor=scratch_exp_dw, out_tensor=pingpong.pong, residual_tensor=pingpong.ping)
    if profile is not None:
        profile[f"{profile_prefix}project_residual_ms"] = (time.perf_counter() - t0) * 1000.0

    # Return logical channels only
    out_view = pingpong.pong.view_nhwc_i8().reshape(H, W, Cin_hw)
    return out_view[:, :, :Cin].copy()


def make_aligned_spec(*, h: int, w: int, cin: int, expansion_factor: int, simd: int = 8) -> IrBlockSpec:
    cin_hw = align_channels_for_axis_and_simd(cin, axis_bytes=4, simd=simd)
    return IrBlockSpec(h=h, w=w, cin=cin, expansion_factor=expansion_factor, cin_hw=cin_hw)
