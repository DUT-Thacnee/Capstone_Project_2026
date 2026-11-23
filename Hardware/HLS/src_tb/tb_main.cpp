#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

#include "../src_hw/parameters.h"

// Top-level kernel prototype
extern "C" void MobileNet_Stream(
        hls::stream<cnn_axis_word_t> &s_axis,
        hls::stream<cnn_axis_word_t> &s_axis_residual,
        hls::stream<cnn_axis_word_t> &m_axis,
        const volatile cnn_axi_word_t *gmem_w,
        const volatile cnn_axi_word_t *gmem_b,
        const volatile cnn_axi_word_t *gmem_skip,
        cnn_layer_cmd_t cmd);

static constexpr int kAxisBytes = CNN_AXIS_DATA_BITS / 8;
static constexpr int kAxiMmBytes = CNN_AXI_MM_DATA_BITS / 8;

// Must match the m_axi depth pragmas in hardware_functions.cpp
static constexpr uint32_t kGmemWDepth = 100000;
static constexpr uint32_t kGmemBDepth = 2000;
static constexpr uint32_t kGmemSkipDepth = 100000;

static inline void write_mem_u8(cnn_axi_word_t *mem, uint32_t byte_offset, uint8_t val) {
    const uint32_t word_idx = byte_offset / kAxiMmBytes;
    const uint32_t byte_idx = byte_offset % kAxiMmBytes;
    cnn_axi_word_t w = mem[word_idx];
    w.range(8 * byte_idx + 7, 8 * byte_idx) = (ap_uint<8>)val;
    mem[word_idx] = w;
}

static inline void write_mem_u16(cnn_axi_word_t *mem, uint32_t byte_offset, uint16_t val) {
    write_mem_u8(mem, byte_offset + 0, (uint8_t)(val & 0xFF));
    write_mem_u8(mem, byte_offset + 1, (uint8_t)((val >> 8) & 0xFF));
}

static inline void write_mem_i32(cnn_axi_word_t *mem, uint32_t byte_offset, int32_t val) {
    write_mem_u8(mem, byte_offset + 0, (uint8_t)(val & 0xFF));
    write_mem_u8(mem, byte_offset + 1, (uint8_t)((val >> 8) & 0xFF));
    write_mem_u8(mem, byte_offset + 2, (uint8_t)((val >> 16) & 0xFF));
    write_mem_u8(mem, byte_offset + 3, (uint8_t)((val >> 24) & 0xFF));
}

static inline int8_t clamp_i32_to_i8(int32_t v, const cnn_quant_desc_t &q) {
    if (v < (int32_t)q.act_min) {
        v = (int32_t)q.act_min;
    }
    if (v > (int32_t)q.act_max) {
        v = (int32_t)q.act_max;
    }
    return (int8_t)v;
}

static inline int8_t quantize_i32_to_i8(int32_t acc, const cnn_quant_desc_t &q) {
    int64_t prod = (int64_t)acc * (int64_t)q.out_multiplier;
    int32_t scaled;
    if (q.out_shift > 0) {
        int64_t rounding = (int64_t)1 << (q.out_shift - 1);
        scaled = (int32_t)((prod + rounding) >> q.out_shift);
    } else if (q.out_shift < 0) {
        scaled = (int32_t)(prod << (-q.out_shift));
    } else {
        scaled = (int32_t)prod;
    }
    scaled += (int32_t)q.out_zero_point;
    return clamp_i32_to_i8(scaled, q);
}

static void sw_golden_conv1x1(
        const std::vector<int8_t> &input,
        const std::vector<int8_t> &residual,
        const std::vector<int8_t> &weights,
        const std::vector<int32_t> &bias,
        int H,
        int W,
        int Cin,
        int Cout,
        const cnn_quant_desc_t &q,
        std::vector<int8_t> &output) {
    output.resize(H * W * Cout);
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < Cout; oc++) {
                int32_t acc = bias[oc];
                for (int ic = 0; ic < Cin; ic++) {
                    const int in_idx = (h * W + w) * Cin + ic;
                    const int w_idx = oc * Cin + ic;
                    acc += (int32_t)input[in_idx] * (int32_t)weights[w_idx];
                }
                const int out_idx = (h * W + w) * Cout + oc;
                const int8_t projection = quantize_i32_to_i8(acc, q);
                output[out_idx] = clamp_i32_to_i8(
                        (int32_t)projection + (int32_t)residual[out_idx] - (int32_t)q.out_zero_point,
                        q);
            }
        }
    }
}

struct RlePair {
    uint8_t distance;
    int8_t value;
};

static std::vector<RlePair> rle_encode_distance_value(const std::vector<int8_t> &dense) {
    std::vector<RlePair> pairs;
    int zero_run = 0;
    for (size_t i = 0; i < dense.size(); i++) {
        const int8_t v = dense[i];
        if (v == 0) {
            zero_run++;
            continue;
        }
        while (zero_run > 255) {
            pairs.push_back({255, 0});
            zero_run -= 256;
        }
        pairs.push_back({(uint8_t)zero_run, v});
        zero_run = 0;
    }
    return pairs;
}

static void pack_axis_stream(hls::stream<cnn_axis_word_t> &s_axis, const std::vector<uint8_t> &bytes) {
    const size_t total_bytes = bytes.size();
    const size_t total_words = total_bytes / kAxisBytes;
    for (size_t i = 0; i < total_words; i++) {
        ap_uint<CNN_AXIS_DATA_BITS> data = 0;
        for (int b = 0; b < kAxisBytes; b++) {
            data |= (ap_uint<CNN_AXIS_DATA_BITS>)bytes[i * kAxisBytes + b] << (8 * b);
        }
        cnn_axis_word_t w;
        w.data = data;
        w.keep = (decltype(w.keep))(-1);
        w.strb = (decltype(w.strb))(-1);
        w.last = (i + 1 == total_words);
        s_axis.write(w);
    }
}

static void unpack_axis_stream(hls::stream<cnn_axis_word_t> &m_axis, size_t total_words, std::vector<uint8_t> &bytes) {
    bytes.clear();
    bytes.reserve(total_words * kAxisBytes);
    for (size_t i = 0; i < total_words; i++) {
        cnn_axis_word_t w = m_axis.read();
        ap_uint<CNN_AXIS_DATA_BITS> data = w.data;
        for (int b = 0; b < kAxisBytes; b++) {
            uint8_t by = (uint8_t)((data >> (8 * b)) & 0xFF);
            bytes.push_back(by);
        }
    }
}

int main() {
    const int H = 4;
    const int W = 4;
    const int Cin = 32;
    const int Cout = 32;

    const int in_bytes = H * W * Cin;
    const int out_bytes = H * W * Cout;
    const int in_words = in_bytes / kAxisBytes;
    const int out_words = out_bytes / kAxisBytes;

    // Build dummy input with corner cases (min/max int8).
    std::vector<int8_t> input(in_bytes, 0);
    for (int i = 0; i < in_bytes; i++) {
        int v = (i % 9) - 4;
        input[i] = (int8_t)v;
    }
    input[0] = 127;
    input[1] = -128;
    input[2] = 127;
    input[3] = -128;

    // Dense weights with intentional zero sequences.
    std::vector<int8_t> weights(Cout * Cin, 0);
    for (int i = 0; i < (int)weights.size(); i++) {
        if (i < 300) {
            weights[i] = 0;
        } else if ((i % 11) == 0) {
            weights[i] = 0;
        } else {
            weights[i] = (int8_t)((i % 7) - 3);
        }
    }
    weights[300] = 127;
    weights[301] = -128;

    std::vector<int32_t> bias(Cout, 0);
    for (int oc = 0; oc < Cout; oc++) {
        bias[oc] = (oc % 5) - 2;
    }

    // RLE encoding into (distance,value) pairs, then map to HW skip list + NZ values.
    std::vector<RlePair> pairs = rle_encode_distance_value(weights);
    const uint32_t nnz = (uint32_t)pairs.size();

    std::vector<uint16_t> skip_list(nnz, 0);
    std::vector<int8_t> nz_values(nnz, 0);
    for (uint32_t i = 0; i < nnz; i++) {
        skip_list[i] = (uint16_t)pairs[i].distance;
        nz_values[i] = pairs[i].value;
    }

    // Allocate mock DDR memories.
    const uint32_t w_nz_bytes = nnz;
    const uint32_t skip_bytes = nnz * 2;
    const uint32_t b_bytes = Cout * 4;

    const uint32_t w_words = (w_nz_bytes + kAxiMmBytes - 1) / kAxiMmBytes;
    const uint32_t skip_words = (skip_bytes + kAxiMmBytes - 1) / kAxiMmBytes;
    const uint32_t b_words = (b_bytes + kAxiMmBytes - 1) / kAxiMmBytes;

    if (w_words > kGmemWDepth || skip_words > kGmemSkipDepth || b_words > kGmemBDepth) {
        std::cerr << "ERROR: Test vectors exceed configured m_axi depth."
                  << " w_words=" << w_words
                  << " b_words=" << b_words
                  << " skip_words=" << skip_words << "\n";
        return 1;
    }

    std::vector<cnn_axi_word_t> gmem_w(kGmemWDepth);
    std::vector<cnn_axi_word_t> gmem_b(kGmemBDepth);
    std::vector<cnn_axi_word_t> gmem_skip(kGmemSkipDepth);

    for (uint32_t i = 0; i < kGmemWDepth; i++) {
        gmem_w[i] = 0;
    }
    for (uint32_t i = 0; i < kGmemBDepth; i++) {
        gmem_b[i] = 0;
    }
    for (uint32_t i = 0; i < kGmemSkipDepth; i++) {
        gmem_skip[i] = 0;
    }

    // Pack sparse weights and bias into DDR.
    for (uint32_t i = 0; i < nnz; i++) {
        write_mem_u8(gmem_w.data(), i, (uint8_t)nz_values[i]);
        write_mem_u16(gmem_skip.data(), 2u * i, skip_list[i]);
    }
    for (uint32_t i = 0; i < (uint32_t)Cout; i++) {
        write_mem_i32(gmem_b.data(), 4u * i, bias[i]);
    }

    // Prepare AXI-Stream inputs.
    hls::stream<cnn_axis_word_t> s_axis("s_axis");
    hls::stream<cnn_axis_word_t> s_axis_residual("s_axis_residual");
    hls::stream<cnn_axis_word_t> m_axis("m_axis");

    std::vector<uint8_t> in_bytes_vec(in_bytes, 0);
    for (int i = 0; i < in_bytes; i++) {
        in_bytes_vec[i] = (uint8_t)input[i];
    }
    pack_axis_stream(s_axis, in_bytes_vec);

    // Non-zero residual values make the test distinguish clamp-before-add
    // from the incorrect add-before-clamp behavior.
    std::vector<uint8_t> res_bytes_vec(out_bytes, 0);
    std::vector<int8_t> residual(out_bytes, 0);
    for (int i = 0; i < out_bytes; i++) {
        residual[i] = (int8_t)((i % 5) - 2);
        res_bytes_vec[i] = (uint8_t)residual[i];
    }
    pack_axis_stream(s_axis_residual, res_bytes_vec);

    // Build command struct.
    cnn_layer_cmd_t cmd;
    std::memset(&cmd, 0, sizeof(cmd));

    cmd.opcode = CNN_OP_CONV2D;
    cmd.flags = CNN_FLAG_IN_AXIS | CNN_FLAG_OUT_AXIS | CNN_FLAG_FUSE_BIAS |
            CNN_FLAG_FUSE_RESIDUAL | CNN_FLAG_ENABLE_SPARSE;
    cmd.in_words = in_words;
    cmd.out_words = out_words;

    cmd.src.h = H;
    cmd.src.w = W;
    cmd.src.c = Cin;
    cmd.src.layout = CNN_LAYOUT_NHWC;
    cmd.src.elem_bytes = 1;
    cmd.src.stride_w = Cin;
    cmd.src.stride_h = W * Cin;

    cmd.dst.h = H;
    cmd.dst.w = W;
    cmd.dst.c = Cout;
    cmd.dst.layout = CNN_LAYOUT_NHWC;
    cmd.dst.elem_bytes = 1;
    cmd.dst.stride_w = Cout;
    cmd.dst.stride_h = W * Cout;

    cmd.w_addr = 0;
    cmd.b_addr = 0;
    cmd.w_bytes = w_nz_bytes;
    cmd.b_bytes = b_bytes;
    cmd.w_format = CNN_W_SPARSE_RLE;

    cmd.conv.kernel_h = 1;
    cmd.conv.kernel_w = 1;
    cmd.conv.stride_h = 1;
    cmd.conv.stride_w = 1;
    cmd.conv.groups = 1;

    // Quantization: ReLU6 clamp in int8 domain.
    cmd.q.out_multiplier = 1;
    cmd.q.out_shift = 0;
    cmd.q.in_zero_point = 0;
    cmd.q.w_zero_point = 0;
    cmd.q.out_zero_point = 0;
    cmd.q.act_min = 0;
    cmd.q.act_max = 6;

    cmd.sparse.w_nz_addr = 0;
    cmd.sparse.skip_addr = 0;
    cmd.sparse.nnz = nnz;
    cmd.sparse.skip_bits = 16;

    // Run hardware kernel.
    MobileNet_Stream(
        s_axis,
        s_axis_residual,
        m_axis,
        gmem_w.data(),
        gmem_b.data(),
        gmem_skip.data(),
        cmd);

    // Collect hardware output.
    std::vector<uint8_t> out_bytes_vec;
    unpack_axis_stream(m_axis, out_words, out_bytes_vec);

    // Golden reference.
    std::vector<int8_t> golden;
    sw_golden_conv1x1(input, residual, weights, bias, H, W, Cin, Cout, cmd.q, golden);

    // Compare.
    bool pass = true;
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < Cout; oc++) {
                const int idx = (h * W + w) * Cout + oc;
                const int8_t hw_val = (int8_t)out_bytes_vec[idx];
                const int8_t sw_val = golden[idx];
                if (hw_val != sw_val) {
                    std::cerr << "Mismatch at (h=" << h << ", w=" << w << ", c=" << oc
                              << ") hw=" << (int)hw_val << " sw=" << (int)sw_val << "\n";
                    pass = false;
                    break;
                }
            }
            if (!pass) {
                break;
            }
        }
        if (!pass) {
            break;
        }
    }

    if (pass) {
        std::cout << "TEST PASSED\n";
        return 0;
    }

    std::cout << "TEST FAILED\n";
    return 1;
}
