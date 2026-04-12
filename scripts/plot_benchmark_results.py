#!/usr/bin/env python3
"""
Plot graphs from benchmarks/results.csv produced by run_benchmark_matrix.sh.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def read_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status", "").startswith("ok") and row.get("wall_time_s") not in ("", "NA"):
                row["gpus"] = int(row["gpus"])
                row["wall_time_s"] = float(row["wall_time_s"])
                rows.append(row)
    return rows


def main():
    in_csv = "benchmarks/results.csv"
    out_dir = "benchmarks/figures"
    os.makedirs(out_dir, exist_ok=True)

    rows = read_rows(in_csv)
    if not rows:
        raise SystemExit("No valid benchmark rows found in benchmarks/results.csv")

    by_dataset_gpu = defaultdict(list)
    for r in rows:
        by_dataset_gpu[(r["dataset"], r["gpus"])].append(r["wall_time_s"])

    summary = []
    for (dataset, gpus), vals in by_dataset_gpu.items():
        mean = sum(vals) / len(vals)
        summary.append((dataset, gpus, mean, len(vals)))
    summary.sort(key=lambda x: (x[0], x[1]))

    # 1) Bar chart: mean wall-time by dataset/gpu
    labels = [f"{d}\n{g} GPU" + ("" if g == 1 else "s") for d, g, _, _ in summary]
    means = [m for _, _, m, _ in summary]
    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, means)
    plt.ylabel("Mean Wall Time (s)")
    plt.title("LPSim Benchmark Results")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for b, m in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{m:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/wall_time_bar.png", dpi=140)
    plt.close()

    # 2) Line chart: scaling per dataset
    by_dataset = defaultdict(list)
    for d, g, m, n in summary:
        by_dataset[d].append((g, m, n))
    plt.figure(figsize=(8, 4.8))
    for d, items in sorted(by_dataset.items()):
        items.sort()
        x = [g for g, _, _ in items]
        y = [m for _, m, _ in items]
        plt.plot(x, y, marker="o", linewidth=2, label=d)
    plt.xlabel("GPUs")
    plt.ylabel("Mean Wall Time (s)")
    plt.title("LPSim Scaling Curve (from local benchmarks)")
    plt.xticks(sorted({g for _, g, _, _ in summary}))
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/scaling_line.png", dpi=140)
    plt.close()

    print(f"Saved: {out_dir}/wall_time_bar.png")
    print(f"Saved: {out_dir}/scaling_line.png")


if __name__ == "__main__":
    main()

