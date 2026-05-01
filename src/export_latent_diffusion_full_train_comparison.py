#!/usr/bin/env python3
"""
Export comparison plots for all latent diffusion full-train runs.
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


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No metrics rows found in {path}")
    return rows


def human_label(run_name: str) -> str:
    if "mutation_005" in run_name:
        return "Autoresearch mutation_005"
    if "100e" in run_name:
        return "Full Train 100e"
    if "24e" in run_name:
        return "Full Train 24e"
    return run_name


def discover_full_train_runs(latent_root: Path) -> list[Path]:
    runs = sorted(path for path in latent_root.glob("*full_train*") if (path / "metrics.jsonl").exists())
    if not runs:
        raise FileNotFoundError(f"No full-train latent diffusion runs found under {latent_root}")
    return runs


def discover_eval_summaries(eval_root: Path) -> dict[str, Path]:
    summaries: dict[str, Path] = {}
    for path in sorted(eval_root.glob("*full_train*/summary.json")):
        payload = read_json(path)
        checkpoint_path = payload.get("checkpoint_path")
        if not checkpoint_path:
            continue
        run_name = Path(checkpoint_path).parent.name
        summaries[run_name] = path
    return summaries


def build_payload(run_dir: Path, *, eval_summary_path: Path | None = None, label: str | None = None) -> dict[str, Any]:
    rows = read_metrics_jsonl(run_dir / "metrics.jsonl")
    eval_summary = read_json(eval_summary_path) if eval_summary_path is not None else None
    return {
        "run_dir": run_dir,
        "run_name": run_dir.name,
        "label": label or human_label(run_dir.name),
        "rows": rows,
        "eval_summary_path": eval_summary_path,
        "eval_summary": eval_summary,
    }


def plot_epoch_series(ax: plt.Axes, run_payloads: list[dict[str, Any]], metric_key: str, *, title: str, y_label: str) -> None:
    for payload in run_payloads:
        rows = payload["rows"]
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row[metric_key]) for row in rows]
        ax.plot(epochs, values, linewidth=2.0, label=payload["label"])
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)


def plot_bar_metric(ax: plt.Axes, labels: list[str], values: list[float], *, title: str, y_label: str) -> None:
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"][: len(labels)])
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)


def build_run_payloads(
    latent_root: Path,
    eval_root: Path,
    *,
    extra_run_dirs: list[Path] | None = None,
    extra_eval_summaries: list[Path] | None = None,
    extra_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    eval_summaries = discover_eval_summaries(eval_root)
    payloads: list[dict[str, Any]] = []
    for run_dir in discover_full_train_runs(latent_root):
        payloads.append(build_payload(run_dir, eval_summary_path=eval_summaries.get(run_dir.name)))
    extra_run_dirs = extra_run_dirs or []
    extra_eval_summaries = extra_eval_summaries or []
    extra_labels = extra_labels or []
    if not (len(extra_run_dirs) == len(extra_eval_summaries) == len(extra_labels)):
        raise ValueError("extra_run_dirs, extra_eval_summaries, and extra_labels must have the same length")
    for run_dir, eval_summary_path, label in zip(extra_run_dirs, extra_eval_summaries, extra_labels):
        payloads.append(build_payload(run_dir, eval_summary_path=eval_summary_path, label=label))
    return payloads


def export_comparison_figure(
    latent_root: Path,
    eval_root: Path,
    *,
    output_path: Path | None = None,
    extra_run_dirs: list[Path] | None = None,
    extra_eval_summaries: list[Path] | None = None,
    extra_labels: list[str] | None = None,
) -> Path:
    payloads = build_run_payloads(
        latent_root,
        eval_root,
        extra_run_dirs=extra_run_dirs,
        extra_eval_summaries=extra_eval_summaries,
        extra_labels=extra_labels,
    )
    if output_path is None:
        output_path = latent_root / "full_train_metric_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 15), constrained_layout=True)
    fig.suptitle("Latent Diffusion Run Comparison", fontsize=16)

    plot_epoch_series(
        axes[0, 0],
        payloads,
        "val_noise_mse",
        title="Validation Noise MSE",
        y_label="Noise MSE",
    )
    plot_epoch_series(
        axes[0, 1],
        payloads,
        "val_x0_future_root_heading_error_deg",
        title="Validation Future Heading Error",
        y_label="Degrees",
    )
    plot_epoch_series(
        axes[1, 0],
        payloads,
        "val_x0_future_recon_mpjpe",
        title="Validation Future MPJPE",
        y_label="MPJPE",
    )
    plot_epoch_series(
        axes[1, 1],
        payloads,
        "val_x0_future_jerk_error",
        title="Validation Future Jerk Error",
        y_label="Jerk Error",
    )

    labels = [payload["label"] for payload in payloads if payload["eval_summary"] is not None]
    sample0_mpjpe = [float(payload["eval_summary"]["global_metrics"]["sample0_recon_mpjpe"]) for payload in payloads if payload["eval_summary"] is not None]
    min10_heading = [float(payload["eval_summary"]["global_metrics"]["min_root_heading_error_deg_at_k"]) for payload in payloads if payload["eval_summary"] is not None]
    plot_bar_metric(
        axes[2, 0],
        labels,
        sample0_mpjpe,
        title="Held-out Sample0 Recon MPJPE",
        y_label="MPJPE",
    )
    plot_bar_metric(
        axes[2, 1],
        labels,
        min10_heading,
        title="Held-out Min Heading Error @10",
        y_label="Degrees",
    )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latent diffusion run comparison figure")
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=Path("/Users/jiatongguo/Desktop/DOGMA/outputs/imu_only_v1/latent_diffusion"),
        help="Root directory containing latent diffusion training runs",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("/Users/jiatongguo/Desktop/DOGMA/outputs/imu_only_v1/latent_diffusion_eval"),
        help="Root directory containing latent diffusion held-out evaluation runs",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    parser.add_argument(
        "--extra-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional non-full-train run directory to include",
    )
    parser.add_argument(
        "--extra-eval-summary",
        type=Path,
        action="append",
        default=[],
        help="Held-out summary.json corresponding to --extra-run-dir",
    )
    parser.add_argument(
        "--extra-label",
        type=str,
        action="append",
        default=[],
        help="Display label corresponding to --extra-run-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_comparison_figure(
        args.latent_root,
        args.eval_root,
        output_path=args.output_path,
        extra_run_dirs=args.extra_run_dir,
        extra_eval_summaries=args.extra_eval_summary,
        extra_labels=args.extra_label,
    )
    print(output_path)


if __name__ == "__main__":
    main()
