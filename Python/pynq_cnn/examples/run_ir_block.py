"""Example: run one MobileNetV2 inverted residual block on PYNQ.

Edit BITFILE/CORE/DMA names to match your overlay.
"""

import numpy as np

from pynq_cnn import (
    CnnAccelerator,
    IrBlockWeights,
    IrBlockQuant,
    build_ir_block_cmds,
    make_qdesc_from_scales,
)
from pynq_cnn.mobilenetv2 import make_aligned_spec, run_ir_block


def main():
    BITFILE = "/home/xilinx/overlays/mobilenet/mobilenet.bit"
    CORE = "MobileNet_Stream_0"
    DMA_MAIN = "axi_dma_main_0"
    DMA_RES = "axi_dma_residual_0"

    accel = CnnAccelerator(
        bitfile=BITFILE,
        core_name=CORE,
        dma_main_name=DMA_MAIN,
        dma_residual_name=DMA_RES,
    )

    # Example block: H=W=56, Cin=24, expansion=6
    spec = make_aligned_spec(h=56, w=56, cin=24, expansion_factor=6, simd=8)

    # Allocate DDR tables (replace sizes with actual packed model sizes)
    accel.allocate_ddr_tables(w_bytes=8 * 1024 * 1024, b_bytes=512 * 1024, skip_bytes=512 * 1024)

    # Byte offsets into the packed DDR weights/bias buffers
    w_off = IrBlockWeights(w_exp=0, b_exp=0, w_dw=0, b_dw=0, w_proj=0, b_proj=0)

    # Quant bridge: convert PyTorch scales/zero-points into qdesc
    q_exp = make_qdesc_from_scales(scale_in=0.02, scale_w=0.03, scale_out=0.04, out_zero_point=0)
    q_dw = make_qdesc_from_scales(scale_in=0.04, scale_w=0.02, scale_out=0.05, out_zero_point=0)
    # Projection output must share quant domain with residual for correct fusion
    q_proj = make_qdesc_from_scales(scale_in=0.05, scale_w=0.025, scale_out=0.02, out_zero_point=0)
    q_block = IrBlockQuant(q_exp=q_exp, q_dw=q_dw, q_proj=q_proj)

    cmds = build_ir_block_cmds(spec=spec, w=w_off, q=q_block)

    # Ping-pong activations (Cin padded to SIMD/AXIS alignment)
    ping = accel.make_axis_tensor(h=spec.h, w=spec.w, c_padded=spec.cin_padded, c_logical=spec.cin)
    pong = accel.make_axis_tensor(h=spec.h, w=spec.w, c_padded=spec.cin_padded, c_logical=spec.cin)

    # Scratch for expansion/depthwise (Cexp padded)
    scratch_exp_dw = accel.make_axis_tensor(h=spec.h, w=spec.w, c_padded=spec.cexp_padded, c_logical=spec.cexp_padded)

    # Scratch for padded depthwise input (H+2, W+2)
    scratch_dwpad = accel.make_axis_tensor(h=spec.h + 2, w=spec.w + 2, c_padded=spec.cexp_padded, c_logical=spec.cexp_padded)

    from pynq_cnn.accelerator import PingPong
    pp = PingPong(ping=ping, pong=pong)

    x = np.random.randint(-128, 127, size=(spec.h, spec.w, spec.cin), dtype=np.int8)

    y = run_ir_block(
        accel=accel,
        spec=spec,
        cmds=cmds,
        pingpong=pp,
        scratch_exp_dw=scratch_exp_dw,
        scratch_dwpad=scratch_dwpad,
        x_in_nhwc_i8=x,
    )

    print("Output shape:", y.shape, "dtype:", y.dtype)


if __name__ == "__main__":
    main()
