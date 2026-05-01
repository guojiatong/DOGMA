#!/usr/bin/env python3
"""
Export comparison figures for latent diffusion plausibility summaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = (
    "fid_pose",
    "diversity_at_k",
    "jitter_band_violation_rate",
    "jerk_band_violation_rate",
    "heading_delta_band_violation_rate",
    "rom_band_violation_rate",
)

LOWER_BETTER_METRICS = (
    "fid_pose",
    "jitter_band_violation_rate",
    "jerk_band_violation_rate",
    "heading_delta_band_violation_rate",
    "rom_band_violation_rate",
)

TITLE_MAP = {
    "fid_pose": "FID Pose",
    "diversity_at_k": "Diversity@K",
    "jitter_band_violation_rate": "Jitter Band Violation Rate",
    "jerk_band_violation_rate": "Jerk Band Violation Rate",
    "heading_delta_band_violation_rate": "Heading-Delta Band Violation Rate",
    "rom_band_violation_rate": "ROM Band Violation Rate",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_plausibility_summaries(summary_root: Path) -> list[Path]:
    summary_paths = sorted(summary_root.glob("*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No plausibility summaries found under {summary_root}")
    return summary_paths


def _rank_data(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in rows]
    composite_parts: dict[str, list[float]] = {row["run_label"]: [] for row in ranked}
    for metric_key in LOWER_BETTER_METRICS:
        sorted_rows = sorted(ranked, key=lambda row: float(row[metric_key]))
        for rank_index, row in enumerate(sorted_rows, start=1):
            composite_parts[row["run_label"]].append(float(rank_index))
    for row in ranked:
        row["plausibility_rank_score"] = float(np.mean(composite_parts[row["run_label"]]))
    return sorted(ranked, key=lambda row: (float(row["plausibility_rank_score"]), str(row["run_label"])))


def load_summary_rows(summary_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in discover_plausibility_summaries(summary_root):
        payload = read_json(summary_path)
        metrics = payload["global_metrics"]
        rows.append(
            {
                "run_name": payload["run_name"],
                "run_label": payload["run_label"],
                "summary_path": str(summary_path),
                "fid_pose": float(metrics["fid_pose"]),
                "diversity_at_k": float(metrics["diversity_at_k"]),
                "jitter_band_violation_rate": float(metrics["jitter_band_violation_rate"]),
                "jerk_band_violation_rate": float(metrics["jerk_band_violation_rate"]),
                "heading_delta_band_violation_rate": float(metrics["heading_delta_band_violation_rate"]),
                "rom_band_violation_rate": float(metrics["rom_band_violation_rate"]),
            }
        )
    return _rank_data(rows)


def _plot_metric_bars(ax: plt.Axes, rows: list[dict[str, Any]], metric_key: str) -> None:
    labels = [row["run_label"] for row in rows]
    values = [float(row[metric_key]) for row in rows]
    x = np.arange(len(labels))
    color = "#4C78A8" if metric_key != "diversity_at_k" else "#54A24B"
    bars = ax.bar(x, values, color=color)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_title(TITLE_MAP[metric_key])
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "run_label",
        "run_name",
        "fid_pose",
        "diversity_at_k",
        "jitter_band_violation_rate",
        "jerk_band_violation_rate",
        "heading_delta_band_violation_rate",
        "rom_band_violation_rate",
        "plausibility_rank_score",
        "summary_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_plausibility_comparison_figures(
    *,
    summary_root: Path,
    output_dir: Path | None = None,
    top_k: int = 6,
) -> dict[str, Path]:
    rows = load_summary_rows(summary_root)
    output_dir = summary_root if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "plausibility_metrics.csv", rows)

    all_rows = rows
    top_rows = rows[: min(int(top_k), len(rows))]
    outputs = {
        "all": output_dir / "latent_diffusion_plausibility_all.png",
        "topk": output_dir / "latent_diffusion_plausibility_top6.png",
    }
    for key, subset in (("all", all_rows), ("topk", top_rows)):
        fig, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
        fig.suptitle(
            "Latent Diffusion Plausibility Comparison"
            + (" (Top-6 by plausibility rank)" if key == "topk" else " (All full-train runs)"),
            fontsize=16,
        )
        for ax, metric_key in zip(axes.flat, METRIC_KEYS, strict=True):
            _plot_metric_bars(ax, subset, metric_key)
        fig.savefig(outputs[key], dpi=180)
        plt.close(fig)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latent diffusion plausibility comparison figures")
    parser.add_argument(
        "--summary-root",
        type=Path,
        required=True,
        help="Directory containing plausibility run subdirectories with summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNG and CSV artifacts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top runs to keep in the top-k figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = export_plausibility_comparison_figures(
        summary_root=args.summary_root,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(outputs["all"])
    print(outputs["topk"])


if __name__ == "__main__":
    main()
