#!/usr/bin/env python3
"""
Export a single comparison figure covering all latent diffusion autoresearch experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PLAUSIBILITY_DIRECTIONS = {
    "fid_pose": "low",
    "diversity_at_k": "high",
    "jitter_band_violation_rate": "low",
    "jerk_band_violation_rate": "low",
    "heading_delta_band_violation_rate": "low",
    "rom_band_violation_rate": "low",
}

EXPERIMENT_GROUPS = {"deeper", "wider", "hybrid"}


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


def read_results_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_source_alias(autoresearch_dir: Path) -> str:
    name = autoresearch_dir.name
    if "fulltrain" in name and "2h" in name:
        return "2h-ft"
    if "8h" in name:
        return "8h"
    if "2h" in name:
        return "2h"
    return name


def classify_experiment_group(experiment_id: str) -> str | None:
    if experiment_id.startswith("mutation_") or "_blocks_" in experiment_id:
        return "hybrid"
    if experiment_id.startswith("blocks_"):
        return "deeper"
    if experiment_id.startswith("hidden_"):
        return "wider"
    return None


def build_experiment_payloads(
    autoresearch_dir: Path,
    *,
    keep_only: bool = False,
    top_k: int | None = None,
    include_source_suffix: bool = False,
    experiment_group: str | None = None,
) -> list[dict[str, Any]]:
    if experiment_group is not None and experiment_group not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unsupported experiment_group: {experiment_group}")
    results_rows = read_results_tsv(autoresearch_dir / "results.tsv")
    payloads: list[dict[str, Any]] = []
    source_alias = infer_source_alias(autoresearch_dir)
    for row in results_rows:
        if keep_only and row.get("keep") != "keep":
            continue
        experiment_id = row["experiment_id"]
        if experiment_group is not None and classify_experiment_group(experiment_id) != experiment_group:
            continue
        run_dir = autoresearch_dir / experiment_id
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue
        rows = read_metrics_jsonl(metrics_path)
        display_id = f"{experiment_id} [{source_alias}]" if include_source_suffix else experiment_id
        payloads.append(
            {
                "experiment_id": experiment_id,
                "display_id": display_id,
                "source_alias": source_alias,
                "status": row.get("status", ""),
                "keep": row.get("keep", ""),
                "score": float(row["score"]),
                "best_score_epoch": int(float(row["best_score_epoch"])),
                "best_val_noise_mse": float(row["best_val_noise_mse"]),
                "best_val_heading_deg": float(row["best_val_heading_deg"]),
                "duration_sec": float(row["duration_sec"]),
                "description": row.get("description", ""),
                "run_dir": run_dir,
                "rows": rows,
                "plausibility_metrics": (
                    read_json(run_dir / "plausibility" / "summary.json").get("global_metrics")
                    if (run_dir / "plausibility" / "summary.json").exists()
                    else None
                ),
            }
        )
    if not payloads:
        raise FileNotFoundError(f"No experiment payloads found under {autoresearch_dir}")
    payloads.sort(key=lambda item: item["score"])
    if top_k is not None:
        payloads = payloads[:top_k]
    return payloads


def attach_plausibility_rank_scores(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = [payload for payload in payloads if payload.get("plausibility_metrics") is not None]
    if not available:
        return payloads
    rank_parts: dict[str, list[float]] = {payload["display_id"]: [] for payload in available}
    for metric_key, direction in PLAUSIBILITY_DIRECTIONS.items():
        sorted_payloads = sorted(
            available,
            key=lambda payload: float(payload["plausibility_metrics"][metric_key]),
            reverse=(direction == "high"),
        )
        for rank_index, payload in enumerate(sorted_payloads, start=1):
            rank_parts[payload["display_id"]].append(float(rank_index))
    for payload in available:
        payload["plausibility_rank_score"] = float(np.mean(rank_parts[payload["display_id"]]))
    return payloads


def plot_bar_metric(
    ax: plt.Axes,
    payloads: list[dict[str, Any]],
    *,
    value_getter: Any,
    title: str,
    y_label: str,
    color_map: dict[str, Any],
) -> None:
    sorted_payloads = sorted(payloads, key=value_getter)
    x = np.arange(len(sorted_payloads))
    bar_colors = [color_map[payload["display_id"]] for payload in sorted_payloads]
    bars = ax.bar(x, [value_getter(payload) for payload in sorted_payloads], color=bar_colors, alpha=0.9)
    ax.set_xticks(x, [payload["display_id"] for payload in sorted_payloads], rotation=75, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, payload in zip(bars, sorted_payloads):
        if payload["experiment_id"] in {"baseline", "mutation_005"}:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value_getter(payload):.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def plot_metric(ax: plt.Axes, payloads: list[dict[str, Any]], metric_key: str, *, title: str, y_label: str, color_map: dict[str, Any]) -> None:
    handles = []
    labels = []
    for payload in payloads:
        exp_id = payload["experiment_id"]
        display_id = payload["display_id"]
        rows = payload["rows"]
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row[metric_key]) for row in rows]
        is_best = exp_id == "mutation_005"
        is_baseline = exp_id == "baseline"
        is_keep = payload["keep"] == "keep"
        linewidth = 2.5 if is_best else 2.0 if is_baseline else 1.5 if is_keep else 0.9
        alpha = 0.95 if is_best else 0.85 if is_baseline else 0.7 if is_keep else 0.28
        zorder = 5 if is_best else 4 if is_baseline else 3 if is_keep else 1
        line = ax.plot(
            epochs,
            values,
            linewidth=linewidth,
            alpha=alpha,
            color=color_map[display_id],
            label=display_id,
            zorder=zorder,
        )[0]
        handles.append(line)
        labels.append(display_id)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    return handles, labels


def build_metric_grid_output_path(
    autoresearch_dir: Path,
    *,
    keep_only: bool,
    top_k: int | None,
    experiment_group: str | None = None,
) -> Path:
    prefix = f"{experiment_group}_" if experiment_group is not None else ""
    if keep_only and top_k is not None:
        filename = f"{prefix}top{top_k}_keep_metric_grid.png"
    elif keep_only:
        filename = f"{prefix}keep_metric_grid.png"
    else:
        filename = f"{prefix}metric_grid.png"
    return autoresearch_dir / filename


def export_autoresearch_metric_grid_figure(
    autoresearch_dir: Path,
    *,
    output_path: Path | None = None,
    keep_only: bool = False,
    top_k: int | None = None,
    extra_autoresearch_dirs: list[Path] | None = None,
    experiment_group: str | None = None,
) -> Path:
    autoresearch_dir = Path(autoresearch_dir)
    extra_autoresearch_dirs = extra_autoresearch_dirs or []
    include_source_suffix = len(extra_autoresearch_dirs) > 0
    payloads = build_experiment_payloads(
        autoresearch_dir,
        keep_only=keep_only,
        top_k=None,
        include_source_suffix=include_source_suffix,
        experiment_group=experiment_group,
    )
    for extra_dir in extra_autoresearch_dirs:
        payloads.extend(
            build_experiment_payloads(
                Path(extra_dir),
                keep_only=keep_only,
                top_k=None,
                include_source_suffix=True,
                experiment_group=experiment_group,
            )
        )
    payloads.sort(key=lambda item: item["score"])
    if top_k is not None:
        payloads = payloads[:top_k]
    if not payloads:
        raise FileNotFoundError("No experiment payloads available after filtering")

    if output_path is None:
        output_path = build_metric_grid_output_path(
            autoresearch_dir,
            keep_only=keep_only,
            top_k=top_k,
            experiment_group=experiment_group,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_ids = [payload["display_id"] for payload in payloads]
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(display_ids), 20)))
    color_map = {display_id: colors[idx % len(colors)] for idx, display_id in enumerate(display_ids)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    if keep_only and top_k is not None:
        title_prefix = f"Latent Diffusion Autoresearch Top-{top_k} Keep Metrics"
    elif keep_only:
        title_prefix = "Latent Diffusion Autoresearch Keep Metrics"
    else:
        title_prefix = "Latent Diffusion Autoresearch Metrics"
    if experiment_group is not None:
        title_prefix = f"{title_prefix} | {experiment_group.title()}"
    fig.suptitle(f"{title_prefix} | {autoresearch_dir.name}", fontsize=15)

    handles, labels = plot_metric(
        axes[0, 0],
        payloads,
        "val_noise_mse",
        title="Validation Noise MSE",
        y_label="Noise MSE",
        color_map=color_map,
    )
    plot_metric(
        axes[0, 1],
        payloads,
        "val_x0_future_root_heading_error_deg",
        title="Validation Future Heading Error",
        y_label="Degrees",
        color_map=color_map,
    )
    plot_metric(
        axes[1, 0],
        payloads,
        "val_x0_future_recon_mpjpe",
        title="Validation Future MPJPE",
        y_label="MPJPE",
        color_map=color_map,
    )
    plot_metric(
        axes[1, 1],
        payloads,
        "val_x0_future_jerk_error",
        title="Validation Future Jerk Error",
        y_label="Jerk Error",
        color_map=color_map,
    )

    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_autoresearch_comparison_figure(
    autoresearch_dir: Path,
    *,
    output_path: Path | None = None,
    keep_only: bool = False,
    top_k: int | None = None,
    extra_autoresearch_dirs: list[Path] | None = None,
    experiment_group: str | None = None,
) -> Path:
    autoresearch_dir = Path(autoresearch_dir)
    extra_autoresearch_dirs = extra_autoresearch_dirs or []
    include_source_suffix = len(extra_autoresearch_dirs) > 0
    payloads = build_experiment_payloads(
        autoresearch_dir,
        keep_only=keep_only,
        top_k=None,
        include_source_suffix=include_source_suffix,
        experiment_group=experiment_group,
    )
    for extra_dir in extra_autoresearch_dirs:
        payloads.extend(
            build_experiment_payloads(
                Path(extra_dir),
                keep_only=keep_only,
                top_k=None,
                include_source_suffix=True,
                experiment_group=experiment_group,
            )
        )
    payloads.sort(key=lambda item: item["score"])
    if top_k is not None:
        payloads = payloads[:top_k]
    if not payloads:
        raise FileNotFoundError("No experiment payloads available after filtering")
    payloads = attach_plausibility_rank_scores(payloads)
    if output_path is None:
        if keep_only and top_k is not None:
            output_path = autoresearch_dir / f"top{top_k}_keep_metric_comparison.png"
        elif keep_only:
            output_path = autoresearch_dir / "keep_metric_comparison.png"
        else:
            output_path = autoresearch_dir / "all_experiments_metric_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_ids = [payload["display_id"] for payload in payloads]
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(display_ids), 20)))
    color_map = {display_id: colors[idx % len(colors)] for idx, display_id in enumerate(display_ids)}

    has_plausibility = all(payload.get("plausibility_metrics") is not None for payload in payloads)
    if has_plausibility:
        fig, axes = plt.subplots(4, 2, figsize=(20, 20), constrained_layout=True)
    else:
        fig, axes = plt.subplots(3, 2, figsize=(20, 16), constrained_layout=True)
    if keep_only and top_k is not None:
        title_prefix = f"Latent Diffusion Autoresearch Top-{top_k} Keep"
    elif keep_only:
        title_prefix = "Latent Diffusion Autoresearch Keep"
    else:
        title_prefix = "Latent Diffusion Autoresearch Comparison"
    if experiment_group is not None:
        title_prefix = f"{title_prefix} | {experiment_group.title()}"
    fig.suptitle(f"{title_prefix} | {autoresearch_dir.name}", fontsize=16)

    handles, labels = plot_metric(
        axes[0, 0],
        payloads,
        "val_noise_mse",
        title="Validation Noise MSE",
        y_label="Noise MSE",
        color_map=color_map,
    )
    plot_metric(
        axes[0, 1],
        payloads,
        "val_x0_future_root_heading_error_deg",
        title="Validation Future Heading Error",
        y_label="Degrees",
        color_map=color_map,
    )
    plot_metric(
        axes[1, 0],
        payloads,
        "val_x0_future_recon_mpjpe",
        title="Validation Future MPJPE",
        y_label="MPJPE",
        color_map=color_map,
    )
    plot_metric(
        axes[1, 1],
        payloads,
        "val_x0_future_jerk_error",
        title="Validation Future Jerk Error",
        y_label="Jerk Error",
        color_map=color_map,
    )

    scatter_ax = axes[2, 0]
    for payload in payloads:
        exp_id = payload["experiment_id"]
        display_id = payload["display_id"]
        scatter_ax.scatter(
            payload["best_val_noise_mse"],
            payload["best_val_heading_deg"],
            color=color_map[display_id],
            s=85 if exp_id == "mutation_005" else 70 if exp_id == "baseline" else 45,
            alpha=0.95 if payload["keep"] == "keep" else 0.4,
            edgecolors="black" if payload["keep"] == "keep" else "none",
            linewidths=0.6,
        )
    for payload in payloads[:8]:
        scatter_ax.annotate(
            payload["display_id"],
            (payload["best_val_noise_mse"], payload["best_val_heading_deg"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    scatter_ax.set_title("Best Validation Noise vs Heading")
    scatter_ax.set_xlabel("Best Val Noise MSE")
    scatter_ax.set_ylabel("Best Val Heading Error (deg)")
    scatter_ax.grid(True, alpha=0.25)

    bar_ax = axes[2, 1]
    plot_bar_metric(
        bar_ax,
        payloads,
        value_getter=lambda payload: float(payload["score"]),
        title="Autoresearch Score (lower is better)",
        y_label="Score",
        color_map=color_map,
    )

    if has_plausibility:
        fid_ax = axes[3, 0]
        plot_bar_metric(
            fid_ax,
            payloads,
            value_getter=lambda payload: float(payload["plausibility_metrics"]["fid_pose"]),
            title="Plausibility FID Pose",
            y_label="FID Pose",
            color_map=color_map,
        )
        rank_ax = axes[3, 1]
        plot_bar_metric(
            rank_ax,
            payloads,
            value_getter=lambda payload: float(payload.get("plausibility_rank_score", np.inf)),
            title="Plausibility Rank Score (lower is better)",
            y_label="Rank Score",
            color_map=color_map,
        )

    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export latent diffusion autoresearch comparison figure")
    parser.add_argument(
        "--autoresearch-dir",
        type=Path,
        default=Path("/Users/jiatongguo/Desktop/DOGMA/outputs/imu_only_v1/latent_diffusion/latent_diffusion_autoresearch_8h_20260414_012039"),
        help="Autoresearch output directory containing results.tsv and experiment subdirectories",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    parser.add_argument(
        "--keep-only",
        action="store_true",
        help="Only include experiments whose keep column is 'keep'",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="After filtering, keep only the top-k experiments by score",
    )
    parser.add_argument(
        "--extra-autoresearch-dir",
        type=Path,
        action="append",
        default=[],
        help="Additional autoresearch result directory to merge into the plot",
    )
    parser.add_argument(
        "--experiment-group",
        choices=sorted(EXPERIMENT_GROUPS),
        default=None,
        help="Only include one experiment family: deeper, wider, or hybrid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = export_autoresearch_comparison_figure(
        args.autoresearch_dir,
        output_path=args.output_path,
        keep_only=args.keep_only,
        top_k=args.top_k,
        extra_autoresearch_dirs=args.extra_autoresearch_dir,
        experiment_group=args.experiment_group,
    )
    print(output_path)


if __name__ == "__main__":
    main()
