#!/usr/bin/env python3
"""
Export held-out latent diffusion sample comparisons.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export_temporal_vae_failure_pack import (
    build_context_target_payload,
    compare_payloads,
    render_payload_comparison_video,
    resolve_workspace_path,
    save_payload_npz,
    slice_payload,
    stitch_future_payload_with_context,
)
from train_imu_masked_recon import write_json
from train_latent_diffusion import (
    CONDITION_DIM,
    build_sampling_generator,
    decode_latent_to_pose,
    load_latent_diffusion_checkpoint,
    normalize_condition_tensor,
    sample_latent_diffusion,
)
from train_temporal_vae import POSE_POSITION_DIM, compute_pose_recon_metrics, flatten_pose_window, resolve_position_smoothing_kernel
from video_export_paths import build_export_video_path
from visualize_pose import PseudoPosePayload, load_pseudo_pose


COMPARISON_COLUMNS = (
    "rank",
    "selection_mode",
    "sample_metric",
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
    "context_frames",
    "prediction_frames",
    "sample_count",
    "selected_sample_index",
    "sample0_recon_mpjpe",
    "sample0_root_heading_error_deg",
    "sample0_jerk_error",
    "min_recon_mpjpe_at_k",
    "min_root_heading_error_deg_at_k",
    "min_jerk_error_at_k",
    "selected_sample_recon_mpjpe",
    "selected_sample_root_heading_error_deg",
    "selected_sample_jerk_error",
    "target_vs_sample_mean_mpjpe_mm",
    "target_vs_sample_max_mpjpe_mm",
    "target_vs_sample_mean_heading_error_deg",
    "target_vs_sample_max_heading_error_deg",
    "window_dir",
    "video_path",
)


@dataclass(frozen=True)
class LatentWindowMetric:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    start_frame_20hz: int
    valid_frames: int
    packet_start_20hz: int
    packet_end_20hz: int
    sample_count: int
    sample0_recon_mpjpe: float
    sample0_root_heading_error_deg: float
    sample0_jerk_error: float
    sample0_latent_mse: float
    min_recon_mpjpe_at_k: float
    min_root_heading_error_deg_at_k: float
    min_jerk_error_at_k: float
    min_latent_mse_at_k: float
    best_sample_index_by_mpjpe: int
    best_sample_index_by_heading: int
    best_sample_index_by_latent_mse: int


def read_latent_window_metrics(
    *,
    window_metrics_csv: Path,
    workspace_root: Path,
) -> list[LatentWindowMetric]:
    rows: list[LatentWindowMetric] = []
    with window_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                LatentWindowMetric(
                    participant=row["participant"],
                    segment_id=row["segment_id"],
                    pose_path=resolve_workspace_path(Path(row["pose_path"]), workspace_root),
                    feature_path=resolve_workspace_path(Path(row["feature_path"]), workspace_root),
                    start_frame_20hz=int(row["start_frame_20hz"]),
                    valid_frames=int(row["valid_frames"]),
                    packet_start_20hz=int(row["packet_start_20hz"]),
                    packet_end_20hz=int(row["packet_end_20hz"]),
                    sample_count=int(row["sample_count"]),
                    sample0_recon_mpjpe=float(row["sample0_recon_mpjpe"]),
                    sample0_root_heading_error_deg=float(row["sample0_root_heading_error_deg"]),
                    sample0_jerk_error=float(row["sample0_jerk_error"]),
                    sample0_latent_mse=float(row["sample0_latent_mse"]),
                    min_recon_mpjpe_at_k=float(row["min_recon_mpjpe_at_k"]),
                    min_root_heading_error_deg_at_k=float(row["min_root_heading_error_deg_at_k"]),
                    min_jerk_error_at_k=float(row["min_jerk_error_at_k"]),
                    min_latent_mse_at_k=float(row["min_latent_mse_at_k"]),
                    best_sample_index_by_mpjpe=int(row["best_sample_index_by_mpjpe"]),
                    best_sample_index_by_heading=int(row["best_sample_index_by_heading"]),
                    best_sample_index_by_latent_mse=int(row["best_sample_index_by_latent_mse"]),
                )
            )
    if not rows:
        raise ValueError(f"No window metrics found in {window_metrics_csv}")
    return rows


def select_window_metric_rows(
    rows: list[LatentWindowMetric],
    *,
    selection_mode: str,
    top_k: int,
    max_per_participant: int,
    seed: int,
) -> list[LatentWindowMetric]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if selection_mode == "worst_min_mpjpe":
        sorted_rows = sorted(rows, key=lambda row: row.min_recon_mpjpe_at_k, reverse=True)
    elif selection_mode == "worst_min_heading":
        sorted_rows = sorted(rows, key=lambda row: row.min_root_heading_error_deg_at_k, reverse=True)
    elif selection_mode == "worst_sample0_mpjpe":
        sorted_rows = sorted(rows, key=lambda row: row.sample0_recon_mpjpe, reverse=True)
    elif selection_mode == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
        sorted_rows = [rows[int(index)] for index in order.tolist()]
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    selected: list[LatentWindowMetric] = []
    counts_by_participant: dict[str, int] = {}
    for row in sorted_rows:
        if max_per_participant > 0 and counts_by_participant.get(row.participant, 0) >= max_per_participant:
            continue
        selected.append(row)
        counts_by_participant[row.participant] = counts_by_participant.get(row.participant, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def resolve_selected_sample_index(row: LatentWindowMetric, sample_metric: str) -> int:
    if sample_metric == "mpjpe":
        return int(row.best_sample_index_by_mpjpe)
    if sample_metric == "heading":
        return int(row.best_sample_index_by_heading)
    if sample_metric == "latent_mse":
        return int(row.best_sample_index_by_latent_mse)
    if sample_metric == "sample0":
        return 0
    raise ValueError(f"Unsupported sample_metric: {sample_metric}")


def load_condition_window(
    *,
    feature_path: Path,
    start_frame_20hz: int,
    past_frames: int,
) -> np.ndarray:
    with np.load(feature_path, allow_pickle=False) as payload:
        if "feature" not in payload.files:
            raise ValueError(f"Feature file missing 'feature': {feature_path}")
        feature = payload["feature"]
    past_start = int(start_frame_20hz - past_frames)
    past_end = int(start_frame_20hz)
    if past_start < 0 or past_end > feature.shape[0]:
        raise ValueError(
            f"Feature window [{past_start}, {past_end}) is outside feature axis length {feature.shape[0]} for {feature_path}"
        )
    return feature[past_start:past_end].reshape(past_frames, CONDITION_DIM).astype(np.float32)


def pose_array_to_payload(
    *,
    base_payload: PseudoPosePayload,
    pose_array: np.ndarray,
) -> PseudoPosePayload:
    pose_array = np.asarray(pose_array, dtype=np.float64)
    if pose_array.ndim != 2 or pose_array.shape[1] != POSE_POSITION_DIM + 6:
        raise ValueError(f"Expected pose_array [T,36], got {pose_array.shape}")
    return PseudoPosePayload(
        packet_counter=base_payload.packet_counter.copy(),
        relative_positions=pose_array[:, :POSE_POSITION_DIM].reshape(-1, 10, 3).copy(),
        root_heading_6d=pose_array[:, POSE_POSITION_DIM:].copy(),
        is_interpolated=base_payload.is_interpolated.copy(),
        joint_names=list(base_payload.joint_names),
    )


@torch.no_grad()
def export_latent_diffusion_sample_videos(
    *,
    checkpoint_path: Path,
    window_metrics_csv: Path,
    output_dir: Path,
    top_k: int = 6,
    sample_count: int = 0,
    sampling_seed: int = 0,
    selection_mode: str = "worst_min_mpjpe",
    sample_metric: str = "mpjpe",
    context_frames: int = 120,
    prediction_frames: int = 40,
    max_per_participant: int = 0,
    render_space: str = "heading",
    body_model: str = "smal",
    show_skeleton_overlay: bool = False,
    smal_model: Path | None = None,
    smal_mapping: Path | None = None,
    smal_data: Path | None = None,
    smal_family_index: int = 1,
    fps: int = 20,
    point_size: float = 42.0,
    device: str = "auto",
    no_video: bool = False,
    position_smoothing_kernel: str = "tri5",
    vae_checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if smal_model is None:
        smal_model = workspace_root / "SMAL" / "wolf_alph3.pkl"

    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        config,
        condition_mean,
        condition_std,
        latent_mean,
        latent_std,
        vae_model,
        vae_config,
        pose_mean,
        pose_std,
        diffusion_buffers,
        checkpoint,
    ) = load_latent_diffusion_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
        vae_checkpoint_override=vae_checkpoint_override,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)

    rows = read_latent_window_metrics(
        window_metrics_csv=window_metrics_csv,
        workspace_root=workspace_root,
    )
    rows = [
        row
        for row in rows
        if row.start_frame_20hz >= context_frames and row.valid_frames >= prediction_frames
    ]
    if not rows:
        raise ValueError(
            f"No windows satisfy context_frames={context_frames} and prediction_frames={prediction_frames}"
        )
    selected_rows = select_window_metric_rows(
        rows,
        selection_mode=selection_mode,
        top_k=top_k,
        max_per_participant=max_per_participant,
        seed=sampling_seed,
    )

    summary_rows: list[dict[str, Any]] = []
    if int(config.past_frames) != int(context_frames) or int(config.future_window_frames) != int(prediction_frames):
        raise ValueError(
            "Requested context/prediction frames do not match checkpoint config "
            f"({context_frames}/{prediction_frames} vs {config.past_frames}/{config.future_window_frames})"
        )
    latent_shape = (1, int(vae_config.window_frames // 8), int(vae_config.latent_dim))
    for rank, row in enumerate(selected_rows, start=1):
        selected_sample_index = resolve_selected_sample_index(row, sample_metric)
        if sample_count > 0 and int(row.sample_count) != int(sample_count):
            raise ValueError(
                f"window_metrics sample_count={row.sample_count} does not match requested sample_count={sample_count}"
            )
        target_payload_full = load_pseudo_pose(row.pose_path)
        target_window = slice_payload(
            target_payload_full,
            start_frame=row.start_frame_20hz,
            num_frames=row.valid_frames,
            rebase_heading=True,
        )
        target_context_future = build_context_target_payload(
            full_payload=target_payload_full,
            future_start_frame=row.start_frame_20hz,
            context_frames=context_frames,
            future_frames=prediction_frames,
        )
        condition_np = load_condition_window(
            feature_path=row.feature_path,
            start_frame_20hz=row.start_frame_20hz,
            past_frames=config.past_frames,
        )
        condition = torch.from_numpy(condition_np[None]).to(device=device_resolved, dtype=torch.float32)
        condition = normalize_condition_tensor(condition, condition_mean, condition_std)
        generator = build_sampling_generator(
            device=device_resolved,
            base_seed=sampling_seed,
            meta={
                "participant": row.participant,
                "segment_id": row.segment_id,
                "start_frame_20hz": row.start_frame_20hz,
                "packet_start_20hz": row.packet_start_20hz,
            },
            sample_index=selected_sample_index,
        )
        sampled_latent = sample_latent_diffusion(
            model=model,
            condition=condition,
            latent_shape=latent_shape,
            diffusion_buffers=diffusion_buffers,
            generator=generator,
        )
        sampled_pose = decode_latent_to_pose(
            vae_model=vae_model,
            latent=sampled_latent,
            normalize_pose=vae_config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            position_smoothing_kernel=smoothing_kernel,
        )
        sampled_pose_np = sampled_pose[0].detach().cpu().numpy().astype(np.float64)
        sample_payload = pose_array_to_payload(
            base_payload=target_window,
            pose_array=sampled_pose_np[config.past_frames:config.past_frames + prediction_frames],
        )
        sample_context_future = stitch_future_payload_with_context(
            context_target_payload=target_context_future,
            future_payload=sample_payload,
            context_frames=context_frames,
            future_frames=prediction_frames,
        )

        target_pose_tensor = torch.from_numpy(
            flatten_pose_window(
                target_context_future.relative_positions.astype(np.float32),
                target_context_future.root_heading_6d.astype(np.float32),
            )[None]
        )
        sample_pose_tensor = torch.from_numpy(
            flatten_pose_window(
                sample_context_future.relative_positions.astype(np.float32),
                sample_context_future.root_heading_6d.astype(np.float32),
            )[None]
        )
        pose_metrics = compute_pose_recon_metrics(
            prediction=sample_pose_tensor,
            target=target_pose_tensor,
        )
        comparison = compare_payloads(
            left_payload=target_context_future,
            right_payload=sample_context_future,
            render_space=render_space,
        )

        window_dir = output_dir / (
            f"{rank:02d}__{row.participant}__segment_{row.segment_id}__start_{row.start_frame_20hz}"
        )
        window_dir.mkdir(parents=True, exist_ok=True)
        save_payload_npz(window_dir / "target_window_rebased.npz", target_window)
        save_payload_npz(window_dir / "sample_prediction.npz", sample_payload)
        save_payload_npz(window_dir / "target_context_future_12s.npz", target_context_future)
        save_payload_npz(window_dir / "sample_context_future_12s.npz", sample_context_future)

        video_path = ""
        if not no_video:
            video_output_path = build_export_video_path(
                output_dir=output_dir,
                source_tag="latent-diffusion",
                participant=row.participant,
                segment_id=row.segment_id,
                start_frame_20hz=row.start_frame_20hz,
                rank=rank,
                extra_tags=(selection_mode, sample_metric),
            )
            render_payload_comparison_video(
                left_payload=target_context_future,
                right_payload=sample_context_future,
                output_path=video_output_path,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Pseudo-pose target",
                right_title="Latent diffusion",
                progress_label=f"{row.participant} seg{row.segment_id} start{row.start_frame_20hz}",
                fps=fps,
                point_size=point_size,
                transition_frame=context_frames,
                pre_transition_label=f"PAST {context_frames / 20.0:.1f}S CONTEXT ON BOTH SIDES",
                post_transition_label=f"FUTURE {prediction_frames / 20.0:.1f}S: LEFT TARGET | RIGHT PREDICTION",
                transition_banner="PREDICTION STARTS",
            )
            video_path = str(video_output_path)

        summary_rows.append(
            {
                "rank": int(rank),
                "selection_mode": selection_mode,
                "sample_metric": sample_metric,
                "participant": row.participant,
                "segment_id": row.segment_id,
                "pose_path": str(row.pose_path),
                "feature_path": str(row.feature_path),
                "start_frame_20hz": int(row.start_frame_20hz),
                "valid_frames": int(row.valid_frames),
                "packet_start_20hz": int(row.packet_start_20hz),
                "packet_end_20hz": int(row.packet_end_20hz),
                "context_frames": int(context_frames),
                "prediction_frames": int(prediction_frames),
                "sample_count": int(row.sample_count),
                "selected_sample_index": int(selected_sample_index),
                "sample0_recon_mpjpe": float(row.sample0_recon_mpjpe),
                "sample0_root_heading_error_deg": float(row.sample0_root_heading_error_deg),
                "sample0_jerk_error": float(row.sample0_jerk_error),
                "min_recon_mpjpe_at_k": float(row.min_recon_mpjpe_at_k),
                "min_root_heading_error_deg_at_k": float(row.min_root_heading_error_deg_at_k),
                "min_jerk_error_at_k": float(row.min_jerk_error_at_k),
                "selected_sample_recon_mpjpe": float(pose_metrics["recon_mpjpe"]),
                "selected_sample_root_heading_error_deg": float(pose_metrics["root_heading_error_deg"]),
                "selected_sample_jerk_error": float(pose_metrics["jerk_error"]),
                "target_vs_sample_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                "target_vs_sample_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                "target_vs_sample_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                "target_vs_sample_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
                "window_dir": str(window_dir),
                "video_path": video_path,
            }
        )

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_metrics_csv": str(window_metrics_csv),
        "selection_mode": selection_mode,
        "sample_metric": sample_metric,
        "top_k": len(summary_rows),
        "rows": summary_rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export held-out latent diffusion sample videos")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="worst_min_mpjpe",
        choices=("worst_min_mpjpe", "worst_min_heading", "worst_sample0_mpjpe", "random"),
    )
    parser.add_argument(
        "--sample-metric",
        type=str,
        default="mpjpe",
        choices=("mpjpe", "heading", "latent_mse", "sample0"),
    )
    parser.add_argument("--context-frames", type=int, default=120)
    parser.add_argument("--prediction-frames", type=int, default=40)
    parser.add_argument("--max-per-participant", type=int, default=0)
    parser.add_argument("--render-space", choices=["root", "heading"], default="heading")
    parser.add_argument("--body-model", choices=["shell", "smal"], default="smal")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=root / "SMAL" / "wolf_alph3.pkl")
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_latent_diffusion_sample_videos(
        checkpoint_path=args.checkpoint,
        window_metrics_csv=args.window_metrics_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        selection_mode=args.selection_mode,
        sample_metric=args.sample_metric,
        context_frames=args.context_frames,
        prediction_frames=args.prediction_frames,
        max_per_participant=args.max_per_participant,
        render_space=args.render_space,
        body_model=args.body_model,
        show_skeleton_overlay=args.show_skeleton_overlay,
        smal_model=args.smal_model,
        smal_mapping=args.smal_mapping,
        smal_data=args.smal_data,
        smal_family_index=args.smal_family_index,
        fps=args.fps,
        point_size=args.point_size,
        device=args.device,
        no_video=args.no_video,
        position_smoothing_kernel=args.position_smoothing_kernel,
        vae_checkpoint_override=args.vae_checkpoint,
    )
    print(summary)


if __name__ == "__main__":
    main()
