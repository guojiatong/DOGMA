#!/usr/bin/env python3
"""
Export training/validation loss and metric curves for baseline runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
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


def infer_task(rows: list[dict[str, Any]]) -> str:
    keys = set(rows[0].keys())
    if "val_masked_loss" in keys and "val_masked_rot_geodesic_error" in keys:
        return "masked_reconstruction"
    if "val_future_masked_loss" in keys and "val_future_masked_rot_geodesic_error" in keys:
        return "future_prediction"
    raise ValueError(f"Could not infer task from metrics keys: {sorted(keys)}")


def plot_series(ax: plt.Axes, rows: list[dict[str, Any]], *, title: str, y_label: str, series: list[tuple[str, str, str]]) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    for key, label, style in series:
        if key not in rows[0]:
            continue
        values = [float(row[key]) for row in rows]
        ax.plot(epochs, values, style, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    if len(ax.lines) > 0:
        ax.legend(fontsize=8)


def export_run_curves(run_dir: Path, *, output_path: Path | None = None) -> Path:
    run_dir = Path(run_dir)
    rows = read_metrics_jsonl(run_dir / "metrics.jsonl")
    task = infer_task(rows)

    if output_path is None:
        output_path = run_dir / "plots" / "metrics_overview.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"{task.replace('_', ' ').title()} | {run_dir.name}", fontsize=14)

    if task == "masked_reconstruction":
        plot_series(
            axes[0, 0],
            rows,
            title="Masked Loss",
            y_label="Loss",
            series=[
                ("train_masked_loss", "train masked loss", "-"),
                ("val_masked_loss", "val masked loss", "-"),
            ],
        )
        plot_series(
            axes[0, 1],
            rows,
            title="Rotation Geodesic Error",
            y_label="Radians",
            series=[
                ("train_masked_rot_geodesic_error", "train rot geodesic", "-"),
                ("val_masked_rot_geodesic_error", "val rot geodesic", "-"),
            ],
        )
        plot_series(
            axes[1, 0],
            rows,
            title="Gyroscope RMSE",
            y_label="RMSE",
            series=[
                ("train_gyr_rmse", "train gyr RMSE", "-"),
                ("val_gyr_rmse", "val gyr RMSE", "-"),
            ],
        )
        plot_series(
            axes[1, 1],
            rows,
            title="FreeAcc RMSE",
            y_label="RMSE",
            series=[
                ("train_freeacc_rmse", "train freeacc RMSE", "-"),
                ("val_freeacc_rmse", "val freeacc RMSE", "-"),
            ],
        )
    else:
        plot_series(
            axes[0, 0],
            rows,
            title="Future Loss",
            y_label="Loss",
            series=[
                ("train_future_masked_loss", "train future loss", "-"),
                ("val_future_masked_loss", "val future loss", "-"),
                ("val_persistence_masked_loss", "persistence loss", "--"),
            ],
        )
        plot_series(
            axes[0, 1],
            rows,
            title="Rotation Geodesic Error",
            y_label="Radians",
            series=[
                ("train_future_masked_rot_geodesic_error", "train rot geodesic", "-"),
                ("val_future_masked_rot_geodesic_error", "val rot geodesic", "-"),
                ("val_persistence_masked_rot_geodesic_error", "persistence rot geodesic", "--"),
            ],
        )
        plot_series(
            axes[1, 0],
            rows,
            title="Gyroscope RMSE",
            y_label="RMSE",
            series=[
                ("train_future_gyr_rmse", "train gyr RMSE", "-"),
                ("val_future_gyr_rmse", "val gyr RMSE", "-"),
                ("val_persistence_gyr_rmse", "persistence gyr RMSE", "--"),
            ],
        )
        plot_series(
            axes[1, 1],
            rows,
            title="FreeAcc RMSE",
            y_label="RMSE",
            series=[
                ("train_future_freeacc_rmse", "train freeacc RMSE", "-"),
                ("val_future_freeacc_rmse", "val freeacc RMSE", "-"),
                ("val_persistence_freeacc_rmse", "persistence freeacc RMSE", "--"),
            ],
        )

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export baseline metric curves to PNG")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="One or more baseline run directories")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = [str(export_run_curves(run_dir)) for run_dir in args.run_dir]
    print(json.dumps({"outputs": output_paths}, indent=2))


if __name__ == "__main__":
    main()
