from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched CPU and FPGA benchmark CSV files.")
    parser.add_argument("--fpga", type=Path, default=BASE_DIR / "mobilenetv2_fpga_results.csv")
    parser.add_argument("--cpu", type=Path, default=BASE_DIR / "mobilenetv2_cpu_results.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "benchmark_summary")
    return parser.parse_args()


def read_csv(path: Path, pred_column: str) -> dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {
        "sample_index": np.asarray([int(row["sample_index"]) for row in rows]),
        "label": np.asarray([int(row["label"]) for row in rows]),
        "prediction": np.asarray([int(row[pred_column]) for row in rows]),
        "reference": np.asarray([int(row["reference_top1"]) for row in rows]),
        "latency_ms": np.asarray([float(row["latency_ms"]) for row in rows]),
    }
    for column in rows[0]:
        if column.endswith("_ms") and column != "latency_ms":
            data[column] = np.asarray([float(row[column]) for row in rows])
    return data


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int = 10) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def metrics(data: dict[str, np.ndarray]) -> dict[str, float | int]:
    latency = data["latency_ms"]
    result = {
        "samples": int(len(latency)),
        "accuracy": float(np.mean(data["prediction"] == data["label"])),
        "reference_agreement": float(np.mean(data["prediction"] == data["reference"])),
        "latency_mean_ms": float(np.mean(latency)),
        "latency_std_ms": float(np.std(latency)),
        "latency_median_ms": float(np.median(latency)),
        "latency_p90_ms": float(np.percentile(latency, 90)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "latency_p99_ms": float(np.percentile(latency, 99)),
        "latency_min_ms": float(np.min(latency)),
        "latency_max_ms": float(np.max(latency)),
        "throughput_images_s": float(1000.0 / np.mean(latency)),
        "total_time_s": float(np.sum(latency) / 1000.0),
        "per_class_accuracy": [
            float(np.mean(data["prediction"][data["label"] == cls] == cls)) for cls in range(10)
        ],
    }
    profile = {
        name: float(np.mean(values))
        for name, values in data.items()
        if name.endswith("_ms") and name != "latency_ms"
    }
    if profile:
        result["profile_mean_ms"] = profile
    return result


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label/prediction", *range(matrix.shape[1])])
        for index, row in enumerate(matrix):
            writer.writerow([index, *row.tolist()])


def main() -> None:
    args = parse_args()
    fpga = read_csv(args.fpga, "fpga_top1")
    cpu = read_csv(args.cpu, "cpu_top1")
    if not np.array_equal(fpga["sample_index"], cpu["sample_index"]):
        raise ValueError("CPU and FPGA CSV files do not contain the same sample indices in the same order.")
    if not np.array_equal(fpga["label"], cpu["label"]):
        raise ValueError("CPU and FPGA CSV labels differ.")

    fpga_metrics = metrics(fpga)
    cpu_metrics = metrics(cpu)
    summary = {
        "fpga": fpga_metrics,
        "cpu": cpu_metrics,
        "comparison": {
            "speedup_cpu_over_fpga": cpu_metrics["latency_mean_ms"] / fpga_metrics["latency_mean_ms"],
            "throughput_gain": fpga_metrics["throughput_images_s"] / cpu_metrics["throughput_images_s"],
            "cpu_fpga_top1_agreement": float(np.mean(cpu["prediction"] == fpga["prediction"])),
            "accuracy_difference_fpga_minus_cpu": fpga_metrics["accuracy"] - cpu_metrics["accuracy"],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_matrix(args.output_dir / "fpga_confusion_matrix.csv", confusion_matrix(fpga["label"], fpga["prediction"]))
    write_matrix(args.output_dir / "cpu_confusion_matrix.csv", confusion_matrix(cpu["label"], cpu["prediction"]))

    print(json.dumps(summary, indent=2))
    print(f"Summary files: {args.output_dir}")


if __name__ == "__main__":
    main()
