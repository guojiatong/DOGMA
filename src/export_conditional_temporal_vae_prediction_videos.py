#!/usr/bin/env python3
"""
Export conditional temporal VAE future-prediction videos with past context prepended.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from evaluate_conditional_temporal_vae import load_conditional_temporal_vae_checkpoint
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
from train_temporal_vae import POSE_POSITION_DIM, compute_pose_recon_metrics, flatten_pose_window, resolve_position_smoothing_kernel
from train_conditional_temporal_vae import predict_conditional_future_pose
from video_export_paths import build_export_video_path
from visualize_pose import load_pseudo_pose


SUMMARY_COLUMNS = (
    "rank",
    "selection_mode",
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
    "eval_recon_mpjpe",
    "eval_root_heading_error_deg",
    "eval_jerk_error",
    "selected_recon_mpjpe",
    "selected_root_heading_error_deg",
    "selected_jerk_error",
    "target_vs_prediction_mean_mpjpe_mm",
    "target_vs_prediction_max_mpjpe_mm",
    "target_vs_prediction_mean_heading_error_deg",
    "target_vs_prediction_max_heading_error_deg",
    "window_dir",
    "video_path",
)


@dataclass(frozen=True)
class ConditionalTemporalVaeWindow:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    start_frame_20hz: int
    valid_frames: int
    packet_start_20hz: int
    packet_end_20hz: int
    recon_mpjpe: float
    root_heading_error_deg: float
    position_rmse: float
    heading_rmse: float
    jerk_error: float


def read_window_metrics(*, window_metrics_csv: Path, workspace_root: Path) -> list[ConditionalTemporalVaeWindow]:
    rows: list[ConditionalTemporalVaeWindow] = []
    with window_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                ConditionalTemporalVaeWindow(
                    participant=row["participant"],
                    segment_id=row["segment_id"],
                    pose_path=resolve_workspace_path(Path(row["pose_path"]), workspace_root),
                    feature_path=resolve_workspace_path(Path(row["feature_path"]), workspace_root),
                    start_frame_20hz=int(row["start_frame_20hz"]),
                    valid_frames=int(row["valid_frames"]),
                    packet_start_20hz=int(row["packet_start_20hz"]),
                    packet_end_20hz=int(row["packet_end_20hz"]),
                    recon_mpjpe=float(row["recon_mpjpe"]),
                    root_heading_error_deg=float(row["root_heading_error_deg"]),
                    position_rmse=float(row["position_rmse"]),
                    heading_rmse=float(row["heading_rmse"]),
                    jerk_error=float(row["jerk_error"]),
                )
            )
    if not rows:
        raise ValueError(f"No conditional temporal VAE window metrics found in {window_metrics_csv}")
    return rows


def select_windows(
    rows: list[ConditionalTemporalVaeWindow],
    *,
    selection_mode: str,
    top_k: int,
    max_per_participant: int,
    seed: int,
) -> list[ConditionalTemporalVaeWindow]:
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if selection_mode == "worst_recon":
        sorted_rows = sorted(rows, key=lambda row: row.recon_mpjpe, reverse=True)
    elif selection_mode == "worst_heading":
        sorted_rows = sorted(rows, key=lambda row: row.root_heading_error_deg, reverse=True)
    elif selection_mode == "random":
        import numpy as np

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(rows))
        sorted_rows = [rows[int(index)] for index in order.tolist()]
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    selected: list[ConditionalTemporalVaeWindow] = []
    counts_by_participant: dict[str, int] = {}
    for row in sorted_rows:
        if max_per_participant > 0 and counts_by_participant.get(row.participant, 0) >= max_per_participant:
            continue
        selected.append(row)
        counts_by_participant[row.participant] = counts_by_participant.get(row.participant, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def load_condition_window(
    *,
    feature_path: Path,
    start_frame_20hz: int,
    past_frames: int,
) -> torch.Tensor:
    import numpy as np

    with np.load(feature_path, allow_pickle=False) as payload:
        feature = payload["feature"]
    past_start = int(start_frame_20hz - past_frames)
    past_end = int(start_frame_20hz)
    if past_start < 0 or past_end > feature.shape[0]:
        raise ValueError(
            f"Feature window [{past_start}, {past_end}) is outside feature axis length {feature.shape[0]} for {feature_path}"
        )
    condition = feature[past_start:past_end].reshape(past_frames, -1).astype("float32")
    return torch.from_numpy(condition)


@torch.no_grad()
def export_conditional_temporal_vae_prediction_videos(
    *,
    checkpoint_path: Path,
    window_metrics_csv: Path,
    output_dir: Path,
    top_k: int = 5,
    selection_mode: str = "random",
    selection_seed: int = 23,
    context_frames: int = 160,
    prediction_frames: int = 80,
    max_per_participant: int = 3,
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
) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if smal_model is None:
        smal_model = workspace_root / "SMAL" / "wolf_alph3.pkl"

    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, pose_mean, pose_std, condition_mean, condition_std, checkpoint = load_conditional_temporal_vae_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)
    rows = read_window_metrics(window_metrics_csv=window_metrics_csv, workspace_root=workspace_root)
    rows = [row for row in rows if row.start_frame_20hz >= context_frames and row.valid_frames >= prediction_frames]
    if not rows:
        raise ValueError(f"No conditional temporal VAE windows satisfy context_frames={context_frames} and prediction_frames={prediction_frames}")
    selected_rows = select_windows(
        rows,
        selection_mode=selection_mode,
        top_k=top_k,
        max_per_participant=max_per_participant,
        seed=selection_seed,
    )

    summary_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_rows, start=1):
        full_payload = load_pseudo_pose(row.pose_path)
        future_target_window = slice_payload(
            full_payload,
            start_frame=row.start_frame_20hz,
            num_frames=prediction_frames,
            rebase_heading=True,
            reference_root_heading_6d=(
                full_payload.root_heading_6d[row.start_frame_20hz - 1 : row.start_frame_20hz]
                if config.future_heading_anchor == "context_end"
                else None
            ),
        )
        target_context_future = build_context_target_payload(
            full_payload=full_payload,
            future_start_frame=row.start_frame_20hz,
            context_frames=context_frames,
            future_frames=prediction_frames,
        )
        condition_raw = load_condition_window(
            feature_path=row.feature_path,
            start_frame_20hz=row.start_frame_20hz,
            past_frames=config.past_frames,
        ).to(device_resolved)
        condition = (condition_raw.view(1, config.past_frames, -1) - condition_mean) / condition_std
        future_prediction_tensor = predict_conditional_future_pose(
            model=model,
            condition=condition,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            position_smoothing_kernel=smoothing_kernel,
        )[0]

        import numpy as np
        from visualize_pose import PseudoPosePayload

        future_prediction = PseudoPosePayload(
            packet_counter=future_target_window.packet_counter.copy(),
            relative_positions=future_prediction_tensor[:, :POSE_POSITION_DIM].detach().cpu().numpy().reshape(-1, 10, 3).astype(np.float64),
            root_heading_6d=future_prediction_tensor[:, POSE_POSITION_DIM:].detach().cpu().numpy().astype(np.float64),
            is_interpolated=future_target_window.is_interpolated.copy(),
            joint_names=list(future_target_window.joint_names),
        )
        prediction_context_future = stitch_future_payload_with_context(
            context_target_payload=target_context_future,
            future_payload=future_prediction,
            context_frames=context_frames,
            future_frames=prediction_frames,
            future_heading_anchor=config.future_heading_anchor,
        )

        target_pose_tensor = torch.from_numpy(
            flatten_pose_window(
                target_context_future.relative_positions.astype("float32"),
                target_context_future.root_heading_6d.astype("float32"),
            )[None]
        )
        prediction_pose_tensor = torch.from_numpy(
            flatten_pose_window(
                prediction_context_future.relative_positions.astype("float32"),
                prediction_context_future.root_heading_6d.astype("float32"),
            )[None]
        )
        pose_metrics = compute_pose_recon_metrics(prediction=prediction_pose_tensor, target=target_pose_tensor)
        comparison = compare_payloads(
            left_payload=target_context_future,
            right_payload=prediction_context_future,
            render_space=render_space,
        )

        window_dir = output_dir / f"{rank:02d}__{row.participant}__segment_{row.segment_id}__start_{row.start_frame_20hz}"
        window_dir.mkdir(parents=True, exist_ok=True)
        save_payload_npz(window_dir / "future_target_window_rebased.npz", future_target_window)
        save_payload_npz(window_dir / "future_prediction.npz", future_prediction)
        save_payload_npz(window_dir / "target_context_future_12s.npz", target_context_future)
        save_payload_npz(window_dir / "prediction_context_future_12s.npz", prediction_context_future)

        video_path = ""
        if not no_video:
            video_output_path = build_export_video_path(
                output_dir=output_dir,
                source_tag="conditional-temporal-vae-prediction",
                participant=row.participant,
                segment_id=row.segment_id,
                start_frame_20hz=row.start_frame_20hz,
                rank=rank,
                extra_tags=(selection_mode,),
            )
            render_payload_comparison_video(
                left_payload=target_context_future,
                right_payload=prediction_context_future,
                output_path=video_output_path,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Pseudo-pose target",
                right_title="Conditional temporal VAE",
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
                "eval_recon_mpjpe": float(row.recon_mpjpe),
                "eval_root_heading_error_deg": float(row.root_heading_error_deg),
                "eval_jerk_error": float(row.jerk_error),
                "selected_recon_mpjpe": float(pose_metrics["recon_mpjpe"]),
                "selected_root_heading_error_deg": float(pose_metrics["root_heading_error_deg"]),
                "selected_jerk_error": float(pose_metrics["jerk_error"]),
                "target_vs_prediction_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                "target_vs_prediction_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                "target_vs_prediction_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                "target_vs_prediction_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
                "window_dir": str(window_dir),
                "video_path": video_path,
            }
        )

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_metrics_csv": str(window_metrics_csv),
        "future_heading_anchor": config.future_heading_anchor,
        "selection_mode": selection_mode,
        "top_k": int(top_k),
        "rows": summary_rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export conditional temporal VAE prediction videos")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--selection-mode", type=str, default="random", choices=("random", "worst_recon", "worst_heading"))
    parser.add_argument("--selection-seed", type=int, default=23)
    parser.add_argument("--context-frames", type=int, default=160)
    parser.add_argument("--prediction-frames", type=int, default=80)
    parser.add_argument("--max-per-participant", type=int, default=3)
    parser.add_argument("--render-space", type=str, default="heading", choices=("root", "heading"))
    parser.add_argument("--body-model", type=str, default="smal", choices=("shell", "smal"))
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=None)
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_conditional_temporal_vae_prediction_videos(
        checkpoint_path=args.checkpoint,
        window_metrics_csv=args.window_metrics_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        selection_mode=args.selection_mode,
        selection_seed=args.selection_seed,
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
    )
    print(summary)


if __name__ == "__main__":
    main()
