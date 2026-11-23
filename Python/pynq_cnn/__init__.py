"""PYNQ runtime for the command-driven CNN HLS IP."""

from .commands import (
    TensorDesc,
    Conv2DDesc,
    QuantDesc,
    SparseRleDesc,
    LayerCmd,
    CNN_OP_CONV2D,
    CNN_OP_AVGPOOL,
    CNN_OP_NOP,
    CNN_LAYOUT_NHWC,
    CNN_W_DENSE,
    CNN_W_SPARSE_RLE,
    CNN_ACT_NONE,
    CNN_ACT_RELU,
    CNN_ACT_RELU6,
    CNN_FLAG_IN_AXIS,
    CNN_FLAG_OUT_AXIS,
    CNN_FLAG_FUSE_BIAS,
    CNN_FLAG_FUSE_RESIDUAL,
    CNN_FLAG_ENABLE_SPARSE,
    axis_words_exact,
)

from .packing import (
    axis_words_for_nhwc_i8,
    pack_nhwc_i8_to_u32_words,
    pack_nhwc_i8_to_u32_words_cpad,
    unpack_u32_words_to_nhwc_i8,
    unpack_u32_words_to_nhwc_i8_clogical,
    align_channels_for_axis_and_simd,
)

from .quant import (
    quantize_multiplier,
    make_qdesc_from_scales,
    extract_qparams_from_torch_quant_conv,
)

from .accelerator import CnnAccelerator, AxisTensor, PingPong

from .mobilenetv2 import (
    IrBlockSpec,
    IrBlockWeights,
    IrBlockQuant,
    build_ir_block_cmds,
    run_ir_block,
    make_aligned_spec,
)
