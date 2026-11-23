from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "model_export"
RESULT_CSV = BASE_DIR / "mobilenetv2_cpu_results.csv"
PROFILE_COLUMNS = [
    "block1_expand_ms",
    "block1_depthwise_ms",
    "block1_project_residual_ms",
    "block2_expand_ms",
    "block2_depthwise_ms",
    "block2_project_residual_ms",
    "gap_ms",
    "classifier_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FPGA-compatible INT8 model on the ARM CPU only.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--output", type=Path, default=RESULT_CSV)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def requantize(acc: np.ndarray, quant: dict) -> np.ndarray:
    multiplier = int(quant["out_multiplier"])
    shift = int(quant["out_shift"])
    product = acc.astype(np.int64) * multiplier
    if shift > 0:
        product = (product + (1 << (shift - 1))) >> shift
    elif shift < 0:
        product = product << (-shift)
    product += int(quant["out_zero_point"])
    return product


def quant_desc(layer: dict) -> dict:
    q = layer["quant"]
    real_multiplier = float(q["scale_in"]) * float(q["scale_w"]) / float(q["scale_out"])
    significand, exponent = np.frexp(real_multiplier)
    multiplier = int(round(float(significand) * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exponent += 1
    return {
        "out_multiplier": multiplier,
        "out_shift": 31 - int(exponent),
        "out_zero_point": int(q["zero_point_out"]),
        "act_min": int(q["act_min"]),
        "act_max": int(q["act_max"]),
    }


class CpuInt8FashionRunner:
    """CPU baseline with the same integer arithmetic contract as the HLS core."""

    def __init__(self, meta: dict, weights: np.lib.npyio.NpzFile) -> None:
        self.meta = meta
        self.layers = [layer for layer in meta["layers"] if layer["type"] in {"conv1x1", "depthwise3x3"}]
        self.weights = {key: np.asarray(weights[key]) for key in weights.files}
        self.qdesc = {layer["name"]: quant_desc(layer) for layer in self.layers}

        gap_layer = next(layer for layer in meta["layers"] if layer["type"] == "global_average_pool")
        self.gap_scale = float(gap_layer["quant"]["scale_out"])
        self.gap_zero_point = int(gap_layer["quant"]["zero_point_out"])
        self.gap_quant = {
            "out_multiplier": 1402438301,
            "out_shift": 40,
            "out_zero_point": self.gap_zero_point,
            "act_min": int(gap_layer["quant"]["act_min"]),
            "act_max": int(gap_layer["quant"]["act_max"]),
        }

        classifier = meta["classifier"]
        self.classifier_weight = self.weights[classifier["weight_key"]].astype(np.float32)
        self.classifier_bias = self.weights[classifier["bias_key"]].astype(np.float32)

    def conv(self, x: np.ndarray, layer: dict, residual: np.ndarray | None = None) -> np.ndarray:
        weight = self.weights[layer["weight_key"]].astype(np.int32)
        bias = self.weights[layer["bias_key"]].astype(np.int32)
        if layer["type"] == "conv1x1":
            acc = np.tensordot(x.astype(np.int32), weight.T, axes=([2], [0]))
            acc += bias
        else:
            padded = np.pad(x.astype(np.int32), ((1, 1), (1, 1), (0, 0)))
            acc = np.broadcast_to(bias, x.shape).copy()
            for kh in range(3):
                for kw in range(3):
                    acc += padded[kh : kh + 28, kw : kw + 28, :] * weight[:, kh, kw]

        q = self.qdesc[layer["name"]]
        out = requantize(acc, q)
        if residual is not None:
            out = np.clip(out, q["act_min"], q["act_max"])
            out += residual.astype(np.int64) - q["out_zero_point"]
        return np.clip(out, q["act_min"], q["act_max"]).astype(np.int8)

    def infer(self, image_int8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.ascontiguousarray(image_int8, dtype=np.int8)
        if x.shape != (28, 28, 8):
            raise ValueError(f"Expected image shape (28, 28, 8), got {x.shape}.")

        profile: dict[str, float] = {}
        for block_index, layer_start in enumerate((0, 3), start=1):
            residual = x
            t0 = time.perf_counter()
            x = self.conv(x, self.layers[layer_start])
            profile[f"block{block_index}_expand_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            x = self.conv(x, self.layers[layer_start + 1])
            profile[f"block{block_index}_depthwise_ms"] = (time.perf_counter() - t0) * 1000.0
            t0 = time.perf_counter()
            x = self.conv(x, self.layers[layer_start + 2], residual)
            profile[f"block{block_index}_project_residual_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        sums = x.astype(np.int32).sum(axis=(0, 1))
        gap_i8 = np.clip(
            requantize(sums, self.gap_quant),
            self.gap_quant["act_min"],
            self.gap_quant["act_max"],
        ).astype(np.int8)
        profile["gap_ms"] = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        gap_float = (gap_i8.astype(np.float32) - self.gap_zero_point) * self.gap_scale
        logits = self.classifier_weight @ gap_float + self.classifier_bias
        profile["classifier_ms"] = (time.perf_counter() - t0) * 1000.0
        self.last_profile = profile
        return logits.astype(np.float32), gap_i8


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = parse_args()
    meta_path = args.model_dir / "model_meta.json"
    weights_path = args.model_dir / "model_weights.npz"
    samples_path = args.model_dir / "reference_samples.npz"
    for path in (meta_path, weights_path, samples_path):
        if not path.exists():
            raise FileNotFoundError(path)

    meta = load_json(meta_path)
    with np.load(weights_path) as weights, np.load(samples_path) as samples:
        runner = CpuInt8FashionRunner(meta, weights)
        images = samples["images_int8"]
        labels = samples["labels"]
        reference_top1 = samples["expected_top1"]
        class_names = meta["class_names"]

        start = max(0, args.start_index)
        stop = min(len(images), start + max(1, args.samples))
        for _ in range(max(0, args.warmup)):
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
                    f"label={class_names[label]!r} cpu={class_names[pred]!r} ref={class_names[ref]!r} "
                    f"latency={latency_ms:.3f} ms gap={gap_i8.tolist()}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sample_index", "label", "cpu_top1", "reference_top1", "latency_ms", "cpu_correct", "reference_match", *PROFILE_COLUMNS]
        )
        writer.writerows(rows)

    count = stop - start
    mean_ms = statistics.mean(times_ms)
    print("\n=== CPU-only result summary ===")
    print(f"CPU label accuracy: {correct}/{count} = {correct / count:.4%}")
    print(f"CPU vs INT8 reference top1 agreement: {reference_matches}/{count} = {reference_matches / count:.4%}")
    print(
        f"CPU-only latency: mean={mean_ms:.3f} ms, median={statistics.median(times_ms):.3f} ms, "
        f"p95={percentile(times_ms, 95):.3f} ms, p99={percentile(times_ms, 99):.3f} ms, "
        f"min={min(times_ms):.3f} ms, max={max(times_ms):.3f} ms"
    )
    print(f"Throughput: {1000.0 / mean_ms:.3f} images/s")
    print(f"CSV: {args.output}")


if __name__ == "__main__":
    main()
