from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "Python"))

from pynq_cnn import CnnAccelerator, make_qdesc_from_scales, quantize_multiplier
from pynq_cnn.accelerator import PingPong
from pynq_cnn.commands import (
    CNN_ACT_NONE,
    CNN_FLAG_IN_AXIS,
    CNN_FLAG_OUT_AXIS,
    CNN_OP_AVGPOOL,
    LayerCmd,
    QuantDesc,
    TensorDesc,
    axis_words_exact,
)
from pynq_cnn.mobilenetv2 import (
    IrBlockQuant,
    IrBlockWeights,
    build_ir_block_cmds,
    make_aligned_spec,
    run_ir_block,
)


BITFILE = BASE_DIR / "Hardware" / "output" / "design_1_wrapper.bit"
HWHFILE = BITFILE.with_suffix(".hwh")
LEGACY_HWHFILE = BASE_DIR / "design_1.hwh"
DEFAULT_MODEL_DIR = BASE_DIR / "model_export"
BENCH_CSV = BASE_DIR / "mobilenetv2_fpga_results.csv"
PROFILE_COLUMNS = [
    "block1_expand_ms",
    "block1_padding_ms",
    "block1_depthwise_ms",
    "block1_project_residual_ms",
    "block2_expand_ms",
    "block2_padding_ms",
    "block2_depthwise_ms",
    "block2_project_residual_ms",
    "gap_ms",
    "classifier_ms",
]

CORE = "MobileNet_Stream_0"
DMA_MAIN = "axi_dma_0"
DMA_RES = "axi_dma_1"
DMA_MAX_BYTES = (1 << 26) - 1
TILE_MAX_CHANNELS = 32
AXIS_BYTES = 4


@dataclass
class PackedModel:
    weight_blob: bytes
    bias_blob: bytes
    block_weights: list[IrBlockWeights]


def parse_args() -> argparse.Namespace:   # tạo argrument để cấu hình dự án
    parser = argparse.ArgumentParser(description="Run the exported Fashion-MNIST model on the PYNQ CNN accelerator.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--samples", type=int, default=100, help="Number of balanced exported test samples to run.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=BENCH_CSV)
    return parser.parse_args()


def require_files(model_dir: Path) -> tuple[Path, Path, Path]:  #kiểm tra các file đã có chưa
    if not BITFILE.exists():
        raise FileNotFoundError(f"Bitstream not found: {BITFILE}")
    if not HWHFILE.exists() and LEGACY_HWHFILE.exists():
        shutil.copyfile(LEGACY_HWHFILE, HWHFILE)
    if not HWHFILE.exists():
        raise FileNotFoundError(f"Hardware metadata not found: {HWHFILE}")

    meta_path = model_dir / "model_meta.json"
    weights_path = model_dir / "model_weights.npz"
    samples_path = model_dir / "reference_samples.npz"
    for path in (meta_path, weights_path, samples_path):
        if not path.exists():
            raise FileNotFoundError(f"Model export file not found: {path}")
    return meta_path, weights_path, samples_path


def load_json(path: Path) -> dict:  # đọc file json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def conv_layers(meta: dict) -> list[dict]:  # search trong meta để lấy các layer phù hợp
    layers = [layer for layer in meta["layers"] if layer["type"] in {"conv1x1", "depthwise3x3"}]
    if [layer["name"] for layer in layers] != [
        "block1_expand",
        "block1_depthwise",
        "block1_project",
        "block2_expand",
        "block2_depthwise",
        "block2_project",
    ]:
        raise ValueError("Exported convolution layer order does not match the FPGA scheduler.")
    return layers


def qdesc_from_layer(layer: dict) -> QuantDesc:  # lấy các tham số trong quantization để chuyển int32 về int8
    q = layer["quant"]
    return make_qdesc_from_scales(
        scale_in=float(q["scale_in"]),
        scale_w=float(q["scale_w"]),
        scale_out=float(q["scale_out"]),
        in_zero_point=int(q["zero_point_in"]),
        w_zero_point=int(q["zero_point_w"]),
        out_zero_point=int(q["zero_point_out"]),
        act_min=int(q["act_min"]),
        act_max=int(q["act_max"]),
    )


def pack_model(meta: dict, weights: np.lib.npyio.NpzFile) -> PackedModel:
    w_chunks: list[bytes] = []
    b_chunks: list[bytes] = []
    block_weights: list[IrBlockWeights] = []
    w_offset = 0
    b_offset = 0

    layers = conv_layers(meta)
    for block_start in (0, 3):  # 2 trường hợp là 0 và 3 
        offsets_w: list[int] = []
        offsets_b: list[int] = []
        for layer in layers[block_start : block_start + 3]:  # khi =0 thì lấy 0,1,2 , khi bằng 3 thì lấy 4,5,6
            weight = np.ascontiguousarray(weights[layer["weight_key"]], dtype=np.int8)  # cần dùng ascontiguousarray để đảm bảo dữ liệu được lưu trữ liên tục trong bộ nhớ, tránh lỗi khi chuyển sang bytes
            bias = np.ascontiguousarray(weights[layer["bias_key"]], dtype=np.int32)  # chuyển về 32int để tránh overflow khi tích lũy conv và depthwise
            raw_w = weight.tobytes()  # hàm để chuyển từ numpy array sang bytes (dữ liệu thô nên có thể ghi trực tiếp xuống DDR)
            raw_b = bias.tobytes()

            offsets_w.append(w_offset)  # lưu offset hiện tại
            offsets_b.append(b_offset)
            w_chunks.append(raw_w)  # lưu dữ liệu thô vào list
            b_chunks.append(raw_b)
            w_offset += len(raw_w)  # tăng offset lên bằng kích thước vừa thêm
            b_offset += len(raw_b)

        block_weights.append(
            IrBlockWeights(
                w_exp=offsets_w[0],
                b_exp=offsets_b[0],
                w_dw=offsets_w[1],
                b_dw=offsets_b[1],
                w_proj=offsets_w[2],
                b_proj=offsets_b[2],
            )
        )

    return PackedModel(weight_blob=b"".join(w_chunks), bias_blob=b"".join(b_chunks), block_weights=block_weights)


def build_block_commands(meta: dict, packed: PackedModel):
    layers = conv_layers(meta)
    specs = [
        make_aligned_spec(h=28, w=28, cin=8, expansion_factor=2, simd=8),
        make_aligned_spec(h=28, w=28, cin=8, expansion_factor=3, simd=8),
    ]
    commands = []
    for block_index, spec in enumerate(specs):  # trả về cả index và giá trị của specs
        if spec.cexp_padded > TILE_MAX_CHANNELS:
            raise ValueError(f"Block {block_index + 1} exceeds channel tile limit {TILE_MAX_CHANNELS}.")
        block_layers = layers[block_index * 3 : block_index * 3 + 3]  # lấy 3 layer cho mỗi block
        q = IrBlockQuant(*(qdesc_from_layer(layer) for layer in block_layers))  # dấu * để unpack list thành các tham số riêng lẻ
        commands.append(build_ir_block_cmds(spec=spec, w=packed.block_weights[block_index], q=q))  # cmd sẽ chứa spec của block , w và b của block đó và qdesc của block đó, kể cả là từng layer trong block đó
    return specs, commands


def build_gap_command(meta: dict) -> LayerCmd:
    layer = next(layer for layer in meta["layers"] if layer["type"] == "global_average_pool")
    src = TensorDesc.nhwc_i8(28, 28, 8)  # input
    dst = TensorDesc.nhwc_i8(1, 1, 8)  # output
    multiplier, shift = quantize_multiplier(1.0 / (28 * 28))  #
    q_meta = layer["quant"]  # khởi tạo lượng tử hóa cho GAP layer
    q = QuantDesc(
        out_multiplier=multiplier,
        out_shift=shift,
        in_zero_point=int(q_meta["zero_point_in"]),
        w_zero_point=0,
        out_zero_point=int(q_meta["zero_point_out"]),
        act_min=int(q_meta["act_min"]),
        act_max=int(q_meta["act_max"]),
    )
    cmd = LayerCmd(
        opcode=CNN_OP_AVGPOOL,
        flags=CNN_FLAG_IN_AXIS | CNN_FLAG_OUT_AXIS,  # input và output đều đi qua luồng AXIS
        in_words=axis_words_exact(src.byte_count(), AXIS_BYTES),
        out_words=axis_words_exact(dst.byte_count(), AXIS_BYTES),
        src=src,
        dst=dst,
        activation=CNN_ACT_NONE,  # không có activation riêng cho GAP layer
        q=q,
    )
    cmd.validate_axis_counts(axis_bytes=AXIS_BYTES)  # kiểm tra cmd (command.py)
    return cmd


class FpgaFashionRunner:
    def __init__(self, accel: CnnAccelerator, meta: dict, weights: np.lib.npyio.NpzFile) -> None:
        self.accel = accel
        self.meta = meta
        self.packed = pack_model(meta, weights)
        self.specs, self.block_cmds = build_block_commands(meta, self.packed)
        self.gap_cmd = build_gap_command(meta)

        max_transfer = max(30 * 30 * 24, 28 * 28 * 24, 28 * 28 * 8)
        if max_transfer > DMA_MAX_BYTES:
            raise ValueError(f"Required transfer {max_transfer} exceeds DMA limit {DMA_MAX_BYTES}.")

        accel.allocate_ddr_tables(
            w_bytes=max(len(self.packed.weight_blob), 8),
            b_bytes=max(len(self.packed.bias_blob), 8),
            skip_bytes=8,
        )
        accel.load_ddr_bytes(w_data=self.packed.weight_blob, b_data=self.packed.bias_blob)

        self.ping = accel.make_axis_tensor(h=28, w=28, c_padded=8, c_logical=8)
        self.pong = accel.make_axis_tensor(h=28, w=28, c_padded=8, c_logical=8)
        self.pingpong = PingPong(ping=self.ping, pong=self.pong)
        self.block_scratch = [
            (
                accel.make_axis_tensor(h=28, w=28, c_padded=spec.cexp_padded, c_logical=spec.cexp_padded),
                accel.make_axis_tensor(h=30, w=30, c_padded=spec.cexp_padded, c_logical=spec.cexp_padded),
            )
            for spec in self.specs
        ]
        self.gap_in = accel.make_axis_tensor(h=28, w=28, c_padded=8, c_logical=8)
        self.gap_out = accel.make_axis_tensor(h=1, w=1, c_padded=8, c_logical=8)

        classifier = meta["classifier"]
        self.classifier_weight = np.asarray(weights[classifier["weight_key"]], dtype=np.float32)
        self.classifier_bias = np.asarray(weights[classifier["bias_key"]], dtype=np.float32)
        gap_layer = next(layer for layer in meta["layers"] if layer["type"] == "global_average_pool")
        self.gap_scale = float(gap_layer["quant"]["scale_out"])
        self.gap_zero_point = int(gap_layer["quant"]["zero_point_out"])

    def infer(self, image_int8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(image_int8, dtype=np.int8)
        if x.shape != (28, 28, 8):
            raise ValueError(f"Expected image shape (28, 28, 8), got {x.shape}.")

        profile: dict[str, float] = {}
        for block_index, (spec, cmds, (scratch, padded)) in enumerate(
            zip(self.specs, self.block_cmds, self.block_scratch), start=1
        ):
            x = run_ir_block(
                accel=self.accel,
                spec=spec,
                cmds=cmds,
                pingpong=self.pingpong,
                scratch_exp_dw=scratch,
                scratch_dwpad=padded,
                x_in_nhwc_i8=x,
                profile=profile,
                profile_prefix=f"block{block_index}_",
            )

        t0 = time.perf_counter()
        self.gap_in.write_from_nhwc_i8(x)
        self.accel.run_layer_tensor(cmd=self.gap_cmd, in_tensor=self.gap_in, out_tensor=self.gap_out)
        gap_i8 = self.gap_out.read_to_nhwc_i8().reshape(8)
        profile["gap_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        gap_float = (gap_i8.astype(np.float32) - self.gap_zero_point) * self.gap_scale
        logits = self.classifier_weight @ gap_float + self.classifier_bias
        profile["classifier_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_profile = profile
        return logits.astype(np.float32), gap_i8


def summarize_times(times_ms: list[float]) -> str:
    return (
        f"mean={statistics.mean(times_ms):.3f} ms, median={statistics.median(times_ms):.3f} ms, "
        f"min={min(times_ms):.3f} ms, max={max(times_ms):.3f} ms"
    )


def write_results(rows: list[list[object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample_index", "label", "fpga_top1", "reference_top1", "latency_ms", "fpga_correct", "reference_match", *PROFILE_COLUMNS]
        )
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    meta_path, weights_path, samples_path = require_files(args.model_dir)
    meta = load_json(meta_path)

    with np.load(weights_path) as weights, np.load(samples_path) as samples:
        images = samples["images_int8"]
        labels = samples["labels"]
        reference_top1 = samples["expected_top1"]
        class_names = meta["class_names"]

        start = max(0, int(args.start_index))
        stop = min(len(images), start + max(1, int(args.samples)))
        if start >= stop:
            raise ValueError(f"Invalid sample range: start={start}, stop={stop}, available={len(images)}")

        print("=== Fashion-MNIST MobileNetV2 FPGA inference ===")
        print(f"bitfile={BITFILE}")
        print(f"model_dir={args.model_dir}")
        print(f"samples={stop - start}, range=[{start}, {stop})")
        print(f"dma_limit_bytes={DMA_MAX_BYTES}, tile_max_channels={TILE_MAX_CHANNELS}")

        accel = CnnAccelerator(
            bitfile=str(BITFILE),
            core_name=CORE,
            dma_main_name=DMA_MAIN,
            dma_residual_name=DMA_RES,
        )
        runner = FpgaFashionRunner(accel, meta, weights)

        for _ in range(max(0, int(args.warmup))):
            runner.infer(images[start])

        rows: list[list[object]] = []
        times_ms: list[float] = []
        correct = 0
        reference_matches = 0
        for run_number, sample_index in enumerate(range(start, stop), start=1):
            t0 = time.perf_counter()
            logits, gap_i8 = runner.infer(images[sample_index])
            latency_ms = (time.perf_counter() - t0) * 1000.0
            pred = int(np.argmax(logits))
            label = int(labels[sample_index])
            ref = int(reference_top1[sample_index])
            correct += int(pred == label)
            reference_matches += int(pred == ref)
            times_ms.append(latency_ms)
            rows.append(
                [
                    sample_index,
                    label,
                    pred,
                    ref,
                    f"{latency_ms:.6f}",
                    int(pred == label),
                    int(pred == ref),
                    *[f"{runner.last_profile[name]:.6f}" for name in PROFILE_COLUMNS],
                ]
            )

            if run_number == 1 or run_number % max(1, args.print_every) == 0 or sample_index + 1 == stop:
                print(
                    f"[{run_number:04d}/{stop - start:04d}] index={sample_index} "
                    f"label={class_names[label]!r} fpga={class_names[pred]!r} ref={class_names[ref]!r} "
                    f"latency={latency_ms:.3f} ms gap={gap_i8.tolist()}"
                )

    count = stop - start
    write_results(rows, args.output)
    print("\n=== Result summary ===")
    print(f"FPGA label accuracy: {correct}/{count} = {correct / count:.4%}")
    print(f"FPGA vs INT8 reference top1 agreement: {reference_matches}/{count} = {reference_matches / count:.4%}")
    print(f"End-to-end FPGA+CPU classifier latency: {summarize_times(times_ms)}")
    print(f"Throughput: {1000.0 / statistics.mean(times_ms):.3f} images/s")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
