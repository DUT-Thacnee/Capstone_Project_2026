#include "parameters.h"

namespace {

static constexpr int kAxisBytes = CNN_AXIS_DATA_BITS / 8;
static constexpr int kAxiMmBytes = CNN_AXI_MM_DATA_BITS / 8;

static constexpr int kSIMD = CNN_ENGINE_SIMD;
static constexpr int kPE = CNN_ENGINE_PE;

static constexpr int kMaxCin = CNN_TILE_CIN;
static constexpr int kMaxCout = CNN_TILE_COUT;
static constexpr int kMaxH = CNN_TILE_H;
static constexpr int kMaxW = CNN_TILE_W;
static constexpr int kMaxGroups = (kMaxCin + kSIMD - 1) / kSIMD;

typedef ap_uint<8> u8;
typedef ap_int<8> i8;
typedef ap_int<16> i16;
typedef ap_int<32> i32;
typedef ap_int<64> i64;

typedef ap_int<kSIMD * 8> vec_i8;

static inline u8 mem_read_u8(const volatile cnn_axi_word_t *mem, uint32_t byte_offset) {
	#pragma HLS INLINE
	const uint32_t word_idx = byte_offset / kAxiMmBytes;
	const uint32_t byte_idx = byte_offset % kAxiMmBytes;
	const cnn_axi_word_t w = mem[word_idx];
	const cnn_axi_word_t shifted = w >> (byte_idx * 8);
	return (u8)(shifted & 0xFF);
}

static inline ap_uint<16> mem_read_u16(const volatile cnn_axi_word_t *mem, uint32_t byte_offset) {
	#pragma HLS INLINE
	ap_uint<16> v = 0;
	v |= (ap_uint<16>)mem_read_u8(mem, byte_offset + 0);
	v |= (ap_uint<16>)mem_read_u8(mem, byte_offset + 1) << 8;
	return v;
}

static inline i32 mem_read_i32(const volatile cnn_axi_word_t *mem, uint32_t byte_offset) {
	#pragma HLS INLINE
	ap_uint<32> v = 0;
	v |= (ap_uint<32>)mem_read_u8(mem, byte_offset + 0);
	v |= (ap_uint<32>)mem_read_u8(mem, byte_offset + 1) << 8;
	v |= (ap_uint<32>)mem_read_u8(mem, byte_offset + 2) << 16;
	v |= (ap_uint<32>)mem_read_u8(mem, byte_offset + 3) << 24;
	return (i32)v;
}

static inline bool cmd_enable_residual(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	return ((cmd.flags & CNN_FLAG_FUSE_RESIDUAL) != 0u);
}

static inline bool cmd_enable_sparse(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	return ((cmd.flags & CNN_FLAG_ENABLE_SPARSE) != 0u) || (cmd.w_format == CNN_W_SPARSE_RLE);
}

static inline uint32_t cmd_expected_in_bytes(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	return cmd.src.h * cmd.src.w * cmd.src.c * cmd.src.elem_bytes;
}

static inline uint32_t cmd_expected_out_bytes(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	return cmd.dst.h * cmd.dst.w * cmd.dst.c * cmd.dst.elem_bytes;
}

static inline uint32_t cmd_axis_in_bytes(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	return cmd.in_words * (uint32_t)kAxisBytes;
}

static inline uint32_t cmd_axis_out_bytes(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	return cmd.out_words * (uint32_t)kAxisBytes;
}

static inline uint32_t cmd_weight_count_dense_bytes(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	if (cmd.opcode == CNN_OP_CONV2D) {
		if (cmd.conv.kernel_h == 1 && cmd.conv.kernel_w == 1) {
			return cmd.dst.c * cmd.src.c;
		}
		if (cmd.conv.kernel_h == 3 && cmd.conv.kernel_w == 3 && cmd.conv.groups == cmd.src.c && cmd.dst.c == cmd.src.c) {
			return cmd.src.c * 9;
		}
	}
	return 0;
}

static inline uint32_t cmd_bias_count(const cnn_layer_cmd_t &cmd) {
	#pragma HLS INLINE
	if ((cmd.flags & CNN_FLAG_FUSE_BIAS) == 0u) {
		return 0;
	}
	if (cmd.opcode == CNN_OP_CONV2D) {
		if (cmd.conv.kernel_h == 1 && cmd.conv.kernel_w == 1) {
			return cmd.dst.c;
		}
		if (cmd.conv.kernel_h == 3 && cmd.conv.kernel_w == 3 && cmd.conv.groups == cmd.src.c && cmd.dst.c == cmd.src.c) {
			return cmd.src.c;
		}
	}
	return 0;
}

static inline i32 requantize_i32_to_i32(i32 acc, const cnn_quant_desc_t &q) {
	#pragma HLS INLINE
	i64 prod = (i64)acc * (i64)q.out_multiplier;
	i32 scaled;
	if (q.out_shift > 0) {
		const i64 rounding = (i64)1 << (q.out_shift - 1);
		scaled = (i32)((prod + rounding) >> q.out_shift);
	} else if (q.out_shift < 0) {
		scaled = (i32)(prod << (-q.out_shift));
	} else {
		scaled = (i32)prod;
	}
	scaled += (i32)q.out_zero_point;
	return scaled;
}

static inline i8 clamp_i32_to_i8(i32 v, const cnn_quant_desc_t &q) {
	#pragma HLS INLINE
	if (v < (i32)q.act_min) {
		v = (i32)q.act_min;
	}
	if (v > (i32)q.act_max) {
		v = (i32)q.act_max;
	}
	return (i8)v;
}

static inline i8 quantize_i32_to_i8(i32 acc, const cnn_quant_desc_t &q) {
	#pragma HLS INLINE
	return clamp_i32_to_i8(requantize_i32_to_i32(acc, q), q);
}

static void ReadInputAxis(hls::stream<cnn_axis_word_t> &s_axis, hls::stream<u8> &bytes_out, uint32_t in_words) {
	#pragma HLS INLINE off
	for (uint32_t i = 0; i < in_words; i++) {
		#pragma HLS PIPELINE II=1
		cnn_axis_word_t w = s_axis.read();
		ap_uint<CNN_AXIS_DATA_BITS> data = w.data;
		for (int b = 0; b < kAxisBytes; b++) {
			#pragma HLS UNROLL
			u8 by = (u8)((data >> (8 * b)) & 0xFF);
			bytes_out.write(by);
		}
	}
}

static void WriteOutputAxis(hls::stream<u8> &bytes_in, hls::stream<cnn_axis_word_t> &m_axis, uint32_t out_words) {
	#pragma HLS INLINE off
	for (uint32_t i = 0; i < out_words; i++) {
		#pragma HLS PIPELINE II=1
		ap_uint<CNN_AXIS_DATA_BITS> data = 0;
		for (int b = 0; b < kAxisBytes; b++) {
			#pragma HLS UNROLL
			u8 by = bytes_in.read();
			data |= (ap_uint<CNN_AXIS_DATA_BITS>)by << (8 * b);
		}
		cnn_axis_word_t w;
		w.data = data;
		w.keep = (decltype(w.keep))(-1);
		w.strb = (decltype(w.strb))(-1);
		w.last = (i + 1 == out_words);
		m_axis.write(w);
	}
}



static void ReadWeightsBias(
		const cnn_layer_cmd_t &cmd,
		const volatile cnn_axi_word_t *gmem_w,
		const volatile cnn_axi_word_t *gmem_b,
		const volatile cnn_axi_word_t *gmem_skip,
		hls::stream<u8> &w_bytes,
		hls::stream<i32> &b_words) {
	#pragma HLS INLINE off

	const uint32_t w_count = cmd_weight_count_dense_bytes(cmd);
	const uint32_t b_count = cmd_bias_count(cmd);

	if (w_count != 0) {
		if (cmd_enable_sparse(cmd)) {
			uint32_t out_pos = 0;
			for (uint32_t nz = 0; nz < cmd.sparse.nnz && out_pos < w_count; nz++) {
				#pragma HLS PIPELINE II=1
				ap_uint<16> skip = mem_read_u16(gmem_skip, (uint32_t)cmd.sparse.skip_addr + 2u * nz);
				for (ap_uint<16> z = 0; z < skip && out_pos < w_count; z++) {
					#pragma HLS PIPELINE II=1
					w_bytes.write((u8)0);
					out_pos++;
				}
				if (out_pos < w_count) {
					u8 val = mem_read_u8(gmem_w, (uint32_t)cmd.sparse.w_nz_addr + nz);
					w_bytes.write(val);
					out_pos++;
				}
			}
			for (; out_pos < w_count; out_pos++) {
				#pragma HLS PIPELINE II=1
				w_bytes.write((u8)0);
			}
		} else {
			for (uint32_t i = 0; i < w_count; i++) {
				#pragma HLS PIPELINE II=1
				u8 val = mem_read_u8(gmem_w, (uint32_t)cmd.w_addr + i);
				w_bytes.write(val);
			}
		}
	}

	for (uint32_t i = 0; i < b_count; i++) {
		#pragma HLS PIPELINE II=1
		i32 bias = mem_read_i32(gmem_b, (uint32_t)cmd.b_addr + 4u * i);
		b_words.write(bias);
	}
}

static void DrainAndZero(
		hls::stream<u8> &in_bytes,
		hls::stream<u8> &out_bytes,
		uint32_t in_count,
		uint32_t out_count) {
	#pragma HLS INLINE off
	for (uint32_t i = 0; i < in_count; i++) {
		#pragma HLS PIPELINE II=1
		(void)in_bytes.read();
	}
	for (uint32_t i = 0; i < out_count; i++) {
		#pragma HLS PIPELINE II=1
		out_bytes.write((u8)0);
	}
}

static void DrainOnly(hls::stream<u8> &bytes_in, uint32_t count) {
	#pragma HLS INLINE off
	for (uint32_t i = 0; i < count; i++) {
		#pragma HLS PIPELINE II=1
		(void)bytes_in.read();
	}
}

static void Compute_Conv1x1(
		const cnn_layer_cmd_t &cmd,
		hls::stream<u8> &in_bytes,
		hls::stream<u8> &w_bytes,
		hls::stream<i32> &b_words,
		hls::stream<u8> &residual_bytes,
		hls::stream<u8> &out_bytes) {
	#pragma HLS INLINE off

	const uint32_t H = cmd.src.h;
	const uint32_t W = cmd.src.w;
	const uint32_t Cin = cmd.src.c;
	const uint32_t Cout = cmd.dst.c;
	const bool fuse_residual = cmd_enable_residual(cmd);

	i8 w_buf[kMaxCout][kMaxCin];
	i32 b_buf[kMaxCout];
	i8 act_buf[kMaxCin];

	#pragma HLS ARRAY_PARTITION variable=w_buf dim=1 cyclic factor=kPE
	#pragma HLS ARRAY_PARTITION variable=w_buf dim=2 cyclic factor=kSIMD
	#pragma HLS ARRAY_PARTITION variable=b_buf dim=1 cyclic factor=kPE
	#pragma HLS ARRAY_PARTITION variable=act_buf dim=1 cyclic factor=kSIMD

	for (uint32_t oc = 0; oc < Cout; oc++) {
		for (uint32_t ic = 0; ic < Cin; ic++) {
			#pragma HLS PIPELINE II=1
			u8 wb = w_bytes.read();
			w_buf[oc][ic] = (i8)wb;
		}
	}
	for (uint32_t oc = 0; oc < Cout; oc++) {
		#pragma HLS PIPELINE II=1
		b_buf[oc] = ((cmd.flags & CNN_FLAG_FUSE_BIAS) != 0u) ? b_words.read() : (i32)0;
	}

	const uint32_t pixels = H * W;
	for (uint32_t p = 0; p < pixels; p++) {
		for (uint32_t ic = 0; ic < Cin; ic++) {
			#pragma HLS PIPELINE II=1
			u8 ab = in_bytes.read();
			act_buf[ic] = (i8)ab;
		}
		for (uint32_t oc_base = 0; oc_base < Cout; oc_base += kPE) {
			i32 acc[kPE];
			#pragma HLS ARRAY_PARTITION variable=acc complete
			for (int pe = 0; pe < kPE; pe++) {
				#pragma HLS UNROLL
				acc[pe] = b_buf[oc_base + pe];
			}
			for (uint32_t ic_base = 0; ic_base < Cin; ic_base += kSIMD) {
				i8 a_lane[kSIMD];
				#pragma HLS ARRAY_PARTITION variable=a_lane complete
				for (int lane = 0; lane < kSIMD; lane++) {
					#pragma HLS UNROLL
					a_lane[lane] = act_buf[ic_base + lane];
				}
				for (int pe = 0; pe < kPE; pe++) {
					#pragma HLS UNROLL
					for (int lane = 0; lane < kSIMD; lane++) {
						#pragma HLS UNROLL
						acc[pe] += (i32)((i16)a_lane[lane] * (i16)w_buf[oc_base + pe][ic_base + lane]);
					}
				}
			}
			for (int pe = 0; pe < kPE; pe++) {
				#pragma HLS PIPELINE II=1
				i32 qv = requantize_i32_to_i32(acc[pe], cmd.q);
				if (fuse_residual) {
					// The projection tensor is int8 before it enters the residual add.
					qv = (i32)clamp_i32_to_i8(qv, cmd.q);
					i8 res_q = (i8)residual_bytes.read();
					qv += (i32)res_q - (i32)cmd.q.out_zero_point;
				}
				i8 outv = clamp_i32_to_i8(qv, cmd.q);
				out_bytes.write((u8)outv);
			}
		}
	}
}

static void Compute_Depthwise3x3_Valid(
		const cnn_layer_cmd_t &cmd,
		hls::stream<u8> &in_bytes,
		hls::stream<u8> &w_bytes,
		hls::stream<i32> &b_words,
		hls::stream<u8> &out_bytes) {
	#pragma HLS INLINE off

	const uint32_t H = cmd.src.h;
	const uint32_t W = cmd.src.w;
	const uint32_t C = cmd.src.c;
	const uint32_t stride_h = cmd.conv.stride_h;
	const uint32_t stride_w = cmd.conv.stride_w;

	i8 w_dw[kMaxCin][9];
	i32 b_dw[kMaxCin];

	#pragma HLS ARRAY_PARTITION variable=w_dw dim=1 cyclic factor=kSIMD
	#pragma HLS ARRAY_PARTITION variable=w_dw dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=b_dw dim=1 cyclic factor=kSIMD

	for (uint32_t c = 0; c < C; c++) {
		for (int k = 0; k < 9; k++) {
			#pragma HLS PIPELINE II=1
			u8 wb = w_bytes.read();
			w_dw[c][k] = (i8)wb;
		}
	}
	for (uint32_t c = 0; c < C; c++) {
		#pragma HLS PIPELINE II=1
		b_dw[c] = ((cmd.flags & CNN_FLAG_FUSE_BIAS) != 0u) ? b_words.read() : (i32)0;
	}

	vec_i8 linebuf1[kMaxGroups][kMaxW];
	vec_i8 linebuf2[kMaxGroups][kMaxW];
	vec_i8 win[kMaxGroups][3][3];

	#pragma HLS ARRAY_PARTITION variable=linebuf1 dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=linebuf2 dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=win dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=win dim=2 complete
	#pragma HLS ARRAY_PARTITION variable=win dim=3 complete

	const uint32_t groups = C / (uint32_t)kSIMD;

	for (int g = 0; g < kMaxGroups; g++) {
		for (int x = 0; x < kMaxW; x++) {
			#pragma HLS PIPELINE II=1
			linebuf1[g][x] = 0;
			linebuf2[g][x] = 0;
		}
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				#pragma HLS PIPELINE II=1
				win[g][r][c] = 0;
			}
		}
	}

	for (uint32_t ih = 0; ih < H; ih++) {
		for (uint32_t iw = 0; iw < W; iw++) {
			for (uint32_t g = 0; g < groups; g++) {
				#pragma HLS PIPELINE II=1
				vec_i8 in_vec = 0;
				for (int lane = 0; lane < kSIMD; lane++) {
					#pragma HLS UNROLL
					u8 by = in_bytes.read();
					in_vec.range(8 * lane + 7, 8 * lane) = by;
				}

				const vec_i8 lb2 = linebuf2[g][iw];
				const vec_i8 lb1 = linebuf1[g][iw];

				for (int r = 0; r < 3; r++) {
					#pragma HLS UNROLL
					win[g][r][0] = win[g][r][1];
					win[g][r][1] = win[g][r][2];
				}
				win[g][0][2] = lb2;
				win[g][1][2] = lb1;
				win[g][2][2] = in_vec;

				linebuf2[g][iw] = lb1;
				linebuf1[g][iw] = in_vec;

				if (ih >= 2 && iw >= 2) {
					const uint32_t oh = ih - 2;
					const uint32_t ow = iw - 2;
					if ((oh % stride_h) == 0 && (ow % stride_w) == 0) {
						for (int lane = 0; lane < kSIMD; lane++) {
							#pragma HLS UNROLL
							const uint32_t ch = g * (uint32_t)kSIMD + (uint32_t)lane;
							i32 acc = b_dw[ch];
							for (int k = 0; k < 9; k++) {
								#pragma HLS UNROLL
								const int r = k / 3;
								const int c = k % 3;
								const i8 a = (i8)win[g][r][c].range(8 * lane + 7, 8 * lane);
								acc += (i32)((i16)a * (i16)w_dw[ch][k]);
							}
							i8 outv = quantize_i32_to_i8(acc, cmd.q);
							out_bytes.write((u8)outv);
						}
					}
				}
			}
		}
	}
}

static void Compute_AvgPool_Global(
		const cnn_layer_cmd_t &cmd,
		hls::stream<u8> &in_bytes,
		hls::stream<u8> &out_bytes) {
	#pragma HLS INLINE off

	const uint32_t H = cmd.src.h;
	const uint32_t W = cmd.src.w;
	const uint32_t C = cmd.src.c;

	i32 sum[kMaxGroups][kSIMD];
	#pragma HLS ARRAY_PARTITION variable=sum dim=1 complete
	#pragma HLS ARRAY_PARTITION variable=sum dim=2 complete

	const uint32_t groups = C / (uint32_t)kSIMD;
	for (uint32_t g = 0; g < groups; g++) {
		for (int lane = 0; lane < kSIMD; lane++) {
			#pragma HLS PIPELINE II=1
			sum[g][lane] = 0;
		}
	}

	const uint32_t pixels = H * W;
	for (uint32_t p = 0; p < pixels; p++) {
		for (uint32_t g = 0; g < groups; g++) {
			#pragma HLS PIPELINE II=1
			for (int lane = 0; lane < kSIMD; lane++) {
				#pragma HLS UNROLL
				i8 a = (i8)in_bytes.read();
				sum[g][lane] += (i32)a;
			}
		}
	}

	for (uint32_t g = 0; g < groups; g++) {
		for (int lane = 0; lane < kSIMD; lane++) {
			#pragma HLS PIPELINE II=1
			i8 outv = quantize_i32_to_i8(sum[g][lane], cmd.q);
			out_bytes.write((u8)outv);
		}
	}
}

static void MultiMode_Compute(
		const cnn_layer_cmd_t &cmd,
		hls::stream<u8> &in_bytes,
		hls::stream<u8> &w_bytes,
		hls::stream<i32> &b_words,
		hls::stream<u8> &residual_bytes,
		hls::stream<u8> &out_bytes) {
	#pragma HLS INLINE off

	const uint32_t axis_in = cmd_axis_in_bytes(cmd);
	const uint32_t axis_out = cmd_axis_out_bytes(cmd);
	const uint32_t exp_in = cmd_expected_in_bytes(cmd);
	const uint32_t exp_out = cmd_expected_out_bytes(cmd);
	const uint32_t res_axis = cmd_enable_residual(cmd) ? axis_out : 0;

	const uint32_t w_count = cmd_weight_count_dense_bytes(cmd);
	const uint32_t b_count = cmd_bias_count(cmd);

	const bool counts_ok = (axis_in == exp_in) && (axis_out == exp_out) && (cmd.src.elem_bytes == 1) && (cmd.dst.elem_bytes == 1);

	const bool is_pw = (cmd.opcode == CNN_OP_CONV2D) && (cmd.conv.kernel_h == 1) && (cmd.conv.kernel_w == 1);
	const bool is_dw = (cmd.opcode == CNN_OP_CONV2D) && (cmd.conv.kernel_h == 3) && (cmd.conv.kernel_w == 3) && (cmd.conv.groups == cmd.src.c) && (cmd.dst.c == cmd.src.c);
	const bool is_pool = (cmd.opcode == CNN_OP_AVGPOOL);

	const bool shape_ok_pw =
			is_pw &&
			(cmd.conv.stride_h == 1) &&
			(cmd.conv.stride_w == 1) &&
			(cmd.src.h == cmd.dst.h) &&
			(cmd.src.w == cmd.dst.w) &&
			(cmd.src.c <= (uint32_t)kMaxCin) &&
			(cmd.dst.c <= (uint32_t)kMaxCout) &&
			((cmd.src.c % (uint32_t)kSIMD) == 0) &&
			((cmd.dst.c % (uint32_t)kPE) == 0);

	const uint32_t dw_out_h = (cmd.src.h >= 3 && cmd.conv.stride_h != 0) ? ((cmd.src.h - 3) / cmd.conv.stride_h + 1) : 0;
	const uint32_t dw_out_w = (cmd.src.w >= 3 && cmd.conv.stride_w != 0) ? ((cmd.src.w - 3) / cmd.conv.stride_w + 1) : 0;
	const bool shape_ok_dw =
			is_dw &&
			(cmd.dst.h == dw_out_h) &&
			(cmd.dst.w == dw_out_w) &&
			(cmd.dst.c == cmd.src.c) &&
			(cmd.src.c <= (uint32_t)kMaxCin) &&
			((cmd.src.c % (uint32_t)kSIMD) == 0) &&
			(cmd.src.h <= (uint32_t)kMaxH) &&
			(cmd.src.w <= (uint32_t)kMaxW);

	const bool shape_ok_pool =
			is_pool &&
			(cmd.dst.h == 1) &&
			(cmd.dst.w == 1) &&
			(cmd.dst.c == cmd.src.c) &&
			(cmd.src.c <= (uint32_t)kMaxCin) &&
			((cmd.src.c % (uint32_t)kSIMD) == 0);

	const bool ok = counts_ok && (shape_ok_pw || shape_ok_dw || shape_ok_pool);

	if (ok && is_pw) {
		Compute_Conv1x1(cmd, in_bytes, w_bytes, b_words, residual_bytes, out_bytes);
		return;
	}
	if (ok && is_dw) {
		DrainOnly(residual_bytes, res_axis);
		Compute_Depthwise3x3_Valid(cmd, in_bytes, w_bytes, b_words, out_bytes);
		return;
	}
	if (ok && is_pool) {
		DrainOnly(residual_bytes, res_axis);
		for (uint32_t i = 0; i < w_count; i++) {
			#pragma HLS PIPELINE II=1
			(void)w_bytes.read();
		}
		for (uint32_t i = 0; i < b_count; i++) {
			#pragma HLS PIPELINE II=1
			(void)b_words.read();
		}
		Compute_AvgPool_Global(cmd, in_bytes, out_bytes);
		return;
	}

	for (uint32_t i = 0; i < w_count; i++) {
		#pragma HLS PIPELINE II=1
		(void)w_bytes.read();
	}
	for (uint32_t i = 0; i < b_count; i++) {
		#pragma HLS PIPELINE II=1
		(void)b_words.read();
	}
	DrainOnly(residual_bytes, res_axis);
	DrainAndZero(in_bytes, out_bytes, axis_in, axis_out);
}

} // namespace

extern "C" void MobileNet_Stream(
		hls::stream<cnn_axis_word_t> &s_axis,
		hls::stream<cnn_axis_word_t> &s_axis_residual,
		hls::stream<cnn_axis_word_t> &m_axis,
		const volatile cnn_axi_word_t *gmem_w,
		const volatile cnn_axi_word_t *gmem_b,
		const volatile cnn_axi_word_t *gmem_skip,
		cnn_layer_cmd_t cmd) {
	#pragma HLS INTERFACE s_axilite port=cmd bundle=CTRL
	#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
	#pragma HLS INTERFACE axis port=s_axis
	#pragma HLS INTERFACE axis port=s_axis_residual
	#pragma HLS INTERFACE axis port=m_axis
	#pragma HLS INTERFACE m_axi port=gmem_w offset=slave bundle=W_MEM depth=100000
	#pragma HLS INTERFACE m_axi port=gmem_b offset=slave bundle=B_MEM depth=2000
	#pragma HLS INTERFACE m_axi port=gmem_skip offset=slave bundle=SKIP_MEM depth=100000
	#pragma HLS INTERFACE s_axilite port=gmem_w bundle=CTRL
	#pragma HLS INTERFACE s_axilite port=gmem_b bundle=CTRL
	#pragma HLS INTERFACE s_axilite port=gmem_skip bundle=CTRL

	hls::stream<u8> in_bytes("in_bytes");
	hls::stream<u8> residual_bytes("residual_bytes");
	hls::stream<u8> w_bytes("w_bytes");
	hls::stream<i32> b_words("b_words");
	hls::stream<u8> out_bytes("out_bytes");

	#pragma HLS STREAM variable=in_bytes depth=128
	#pragma HLS STREAM variable=residual_bytes depth=128
	#pragma HLS STREAM variable=w_bytes depth=128
	#pragma HLS STREAM variable=b_words depth=64
	#pragma HLS STREAM variable=out_bytes depth=128

	#pragma HLS DATAFLOW

	ReadInputAxis(s_axis, in_bytes, cmd.in_words);
	ReadInputAxis(s_axis_residual, residual_bytes, cmd_enable_residual(cmd) ? cmd.out_words : 0);
	ReadWeightsBias(cmd, gmem_w, gmem_b, gmem_skip, w_bytes, b_words);
	MultiMode_Compute(cmd, in_bytes, w_bytes, b_words, residual_bytes, out_bytes);
	WriteOutputAxis(out_bytes, m_axis, cmd.out_words);
}
