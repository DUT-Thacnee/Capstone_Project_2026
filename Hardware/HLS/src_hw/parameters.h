#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <stdint.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#ifndef CNN_AXIS_DATA_BITS
#define CNN_AXIS_DATA_BITS 32
#endif

#ifndef CNN_AXI_MM_DATA_BITS
#define CNN_AXI_MM_DATA_BITS 64
#endif

#ifndef CNN_ENGINE_SIMD
#define CNN_ENGINE_SIMD 8
#endif

#ifndef CNN_ENGINE_PE
#define CNN_ENGINE_PE 8
#endif

#ifndef CNN_TILE_H
#define CNN_TILE_H 30
#endif

#ifndef CNN_TILE_W
#define CNN_TILE_W 30
#endif

#ifndef CNN_TILE_CIN
#define CNN_TILE_CIN 32
#endif

#ifndef CNN_TILE_COUT
#define CNN_TILE_COUT 32
#endif

typedef ap_axiu<CNN_AXIS_DATA_BITS, 0, 0, 0> cnn_axis_word_t;
typedef ap_uint<CNN_AXI_MM_DATA_BITS> cnn_axi_word_t;

typedef ap_int<8> cnn_i8_t;
typedef ap_uint<8> cnn_u8_t;
typedef ap_int<16> cnn_i16_t;
typedef ap_uint<16> cnn_u16_t;
typedef ap_int<32> cnn_i32_t;
typedef ap_uint<32> cnn_u32_t;
typedef uint64_t cnn_addr_t;

enum cnn_op_t : uint8_t {
    CNN_OP_CONV2D = 0,
    CNN_OP_AVGPOOL = 1,
    CNN_OP_NOP = 255,
};

enum cnn_layout_t : uint8_t {
    CNN_LAYOUT_NHWC = 0,
};

enum cnn_weight_format_t : uint8_t {
    CNN_W_DENSE = 0,
    CNN_W_SPARSE_RLE = 1,
};

enum cnn_activation_t : uint8_t {
    CNN_ACT_NONE = 0,
    CNN_ACT_RELU = 1,
    CNN_ACT_RELU6 = 2,
};

enum cnn_flags_t : uint32_t {
    CNN_FLAG_IN_AXIS = 1u << 0,
    CNN_FLAG_OUT_AXIS = 1u << 1,
    CNN_FLAG_FUSE_BIAS = 1u << 2,
    CNN_FLAG_FUSE_RESIDUAL = 1u << 3,
    CNN_FLAG_ENABLE_SPARSE = 1u << 4,
};

struct cnn_tensor_desc_t {
    cnn_addr_t addr;
    uint32_t h;
    uint32_t w;
    uint32_t c;
    uint32_t stride_h;
    uint32_t stride_w;
    uint8_t layout;
    uint8_t elem_bytes;
    uint16_t reserved0;
};

struct cnn_conv2d_desc_t {
    uint16_t kernel_h;
    uint16_t kernel_w;
    uint16_t stride_h;
    uint16_t stride_w;
    int16_t pad_top;
    int16_t pad_left;
    uint16_t dilation_h;
    uint16_t dilation_w;
    uint32_t groups;
};

struct cnn_quant_desc_t {
    int32_t out_multiplier;
    int8_t out_shift;
    int8_t in_zero_point;
    int8_t w_zero_point;
    int8_t out_zero_point;
    int8_t act_min;
    int8_t act_max;
    uint8_t reserved0;
};

struct cnn_sparse_rle_desc_t {
    cnn_addr_t w_nz_addr;
    cnn_addr_t skip_addr;
    uint32_t nnz;
    uint32_t skip_bits;
};

struct cnn_layer_cmd_t {
    uint32_t opcode;
    uint32_t flags;
    uint32_t in_words;
    uint32_t out_words;
    cnn_tensor_desc_t src;
    cnn_tensor_desc_t dst;
    cnn_tensor_desc_t residual;
    cnn_addr_t w_addr;
    cnn_addr_t b_addr;
    uint32_t w_bytes;
    uint32_t b_bytes;
    uint8_t w_format;
    uint8_t activation;
    uint16_t reserved0;
    cnn_conv2d_desc_t conv;
    cnn_quant_desc_t q;
    cnn_sparse_rle_desc_t sparse;
    uint32_t user_tag;
};

enum cnn_status_err_t : uint32_t {
    CNN_ERR_OK = 0,
    CNN_ERR_BAD_OPCODE = 1,
    CNN_ERR_BAD_LAYOUT = 2,
    CNN_ERR_BAD_DIMS = 3,
    CNN_ERR_STREAM_UNDERRUN = 4,
    CNN_ERR_STREAM_OVERRUN = 5,
};

struct cnn_status_t {
    uint32_t done;
    uint32_t err;
    uint32_t words_in;
    uint32_t words_out;
    uint32_t user_tag;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t reserved2;
};

#endif
