#!/usr/bin/env python3
"""
Export reconstruction failure packs for temporal VAE windows.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_temporal_vae import load_temporal_vae_checkpoint
from train_imu_masked_recon import write_json
from train_temporal_vae import (
    POSE_POSITION_DIM,
    apply_position_temporal_filter,
    denormalize_pose_tensor,
    flatten_pose_window,
    normalize_pose_tensor,
    rebase_root_heading_6d,
    resolve_position_smoothing_kernel,
)
from video_export_paths import build_export_video_path
from visualize_pose import (
    PseudoPosePayload,
    build_render_arrays,
    build_selected_smal_mesh,
    compute_frame_comparison_metrics,
    compute_motion_metrics,
    compute_temporal_diagnostics,
    infer_pose_metadata,
    load_pseudo_pose,
    load_raw_imu_reference_pose,
    render_overlay_pose_comparison_animation,
    render_raw_vs_pseudo_animation,
)


FAILURE_PACK_COLUMNS = (
    "participant",
    "segment_id",
    "start_frame_20hz",
    "end_frame_20hz",
    "packet_start_20hz",
    "packet_end_20hz",
    "eval_recon_mpjpe",
    "eval_root_heading_error_deg",
    "gt_window_max_root_heading_delta_deg",
    "raw_vs_pseudo_mean_mpjpe_mm",
    "raw_vs_pseudo_max_mpjpe_mm",
    "raw_vs_pseudo_mean_heading_error_deg",
    "raw_vs_pseudo_max_heading_error_deg",
    "gt_vs_recon_mean_mpjpe_mm",
    "gt_vs_recon_max_mpjpe_mm",
    "gt_vs_recon_mean_heading_error_deg",
    "gt_vs_recon_max_heading_error_deg",
    "diagnosis",
    "window_dir",
    "raw_vs_pseudo_video",
    "gt_vs_recon_video",
)


@dataclass(frozen=True)
class FailureWindow:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    start_frame_20hz: int
    valid_frames: int
    packet_start_20hz: int
    packet_end_20hz: int
    eval_recon_mpjpe: float
    eval_root_heading_error_deg: float
    eval_position_rmse: float
    eval_heading_rmse: float
    eval_jerk_error: float


def resolve_workspace_path(path: Path, workspace_root: Path) -> Path:
    if path.exists():
        return path
    text = str(path)
    prefixes = (
        "/home/jiatong/DOGMA/",
        "/Users/jiatongguo/Desktop/DOGMA/",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            candidate = workspace_root / text.removeprefix(prefix)
            if candidate.exists():
                return candidate
    return path


def read_failure_windows(
    *,
    window_metrics_csv: Path,
    participant: str,
    top_k: int,
    workspace_root: Path,
) -> list[FailureWindow]:
    rows: list[FailureWindow] = []
    with window_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["participant"] != participant:
                continue
            rows.append(
                FailureWindow(
                    participant=row["participant"],
                    segment_id=row["segment_id"],
                    pose_path=resolve_workspace_path(Path(row["pose_path"]), workspace_root),
                    feature_path=resolve_workspace_path(Path(row["feature_path"]), workspace_root),
                    start_frame_20hz=int(row["start_frame_20hz"]),
                    valid_frames=int(row["valid_frames"]),
                    packet_start_20hz=int(row["packet_start_20hz"]),
                    packet_end_20hz=int(row["packet_end_20hz"]),
                    eval_recon_mpjpe=float(row["recon_mpjpe"]),
                    eval_root_heading_error_deg=float(row["root_heading_error_deg"]),
                    eval_position_rmse=float(row["position_rmse"]),
                    eval_heading_rmse=float(row["heading_rmse"]),
                    eval_jerk_error=float(row["jerk_error"]),
                )
            )
    rows.sort(key=lambda row: row.eval_root_heading_error_deg, reverse=True)
    return rows[:top_k]


def slice_payload(
    payload: PseudoPosePayload,
    *,
    start_frame: int,
    num_frames: int,
    rebase_heading: bool,
    reference_root_heading_6d: np.ndarray | None = None,
) -> PseudoPosePayload:
    end_frame = start_frame + num_frames
    root_heading_6d = payload.root_heading_6d[start_frame:end_frame]
    if rebase_heading:
        root_heading_6d = rebase_root_heading_6d(
            root_heading_6d,
            reference_root_heading_6d=reference_root_heading_6d,
        )
    return PseudoPosePayload(
        packet_counter=payload.packet_counter[start_frame:end_frame].copy(),
        relative_positions=payload.relative_positions[start_frame:end_frame].copy(),
        root_heading_6d=np.asarray(root_heading_6d, dtype=np.float64).copy(),
        is_interpolated=payload.is_interpolated[start_frame:end_frame].copy(),
        joint_names=list(payload.joint_names),
    )


def save_payload_npz(path: Path, payload: PseudoPosePayload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_joint_name_len = max((len(str(name)) for name in payload.joint_names), default=1)
    np.savez_compressed(
        path,
        packet_counter=payload.packet_counter.astype(np.int64),
        relative_positions=payload.relative_positions.astype(np.float32),
        root_heading_6d=payload.root_heading_6d.astype(np.float32),
        joint_names=np.asarray(payload.joint_names, dtype=f"<U{max_joint_name_len}"),
        is_interpolated=payload.is_interpolated.astype(bool),
    )


def root_heading_6d_to_angles(root_heading_6d: np.ndarray) -> np.ndarray:
    root_heading_6d = np.asarray(root_heading_6d, dtype=np.float64)
    if root_heading_6d.ndim != 2 or root_heading_6d.shape[1] != 6:
        raise ValueError(f"root_heading_6d must have shape [T,6], got {root_heading_6d.shape}")
    if root_heading_6d.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    forward_xy = root_heading_6d[:, :2]
    norms = np.linalg.norm(forward_xy, axis=1, keepdims=True)
    safe_forward_xy = np.divide(
        forward_xy,
        np.maximum(norms, 1e-8),
        out=np.zeros_like(forward_xy),
        where=norms > 1e-8,
    )
    safe_forward_xy[0] = safe_forward_xy[0] if np.linalg.norm(safe_forward_xy[0]) > 1e-8 else np.array([1.0, 0.0])
    for frame_index in range(1, safe_forward_xy.shape[0]):
        if np.linalg.norm(safe_forward_xy[frame_index]) <= 1e-8:
            safe_forward_xy[frame_index] = safe_forward_xy[frame_index - 1]
    return np.unwrap(np.arctan2(safe_forward_xy[:, 1], safe_forward_xy[:, 0]))


def angles_to_root_heading_6d(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float64)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    zeros = np.zeros_like(cosine)
    return np.stack([cosine, sine, zeros, -sine, cosine, zeros], axis=1).astype(np.float32)


def build_context_target_payload(
    *,
    full_payload: PseudoPosePayload,
    future_start_frame: int,
    context_frames: int,
    future_frames: int,
) -> PseudoPosePayload:
    if context_frames < 0 or future_frames <= 0:
        raise ValueError(f"context_frames must be >= 0 and future_frames > 0, got {context_frames}, {future_frames}")
    if future_start_frame < context_frames:
        raise ValueError(
            f"future_start_frame={future_start_frame} is smaller than context_frames={context_frames}"
        )
    combined = slice_payload(
        full_payload,
        start_frame=int(future_start_frame - context_frames),
        num_frames=int(context_frames + future_frames),
        rebase_heading=False,
    )
    return PseudoPosePayload(
        packet_counter=combined.packet_counter.copy(),
        relative_positions=combined.relative_positions.copy(),
        root_heading_6d=rebase_root_heading_6d(combined.root_heading_6d),
        is_interpolated=combined.is_interpolated.copy(),
        joint_names=list(combined.joint_names),
    )


def stitch_future_payload_with_context(
    *,
    context_target_payload: PseudoPosePayload,
    future_payload: PseudoPosePayload,
    context_frames: int,
    future_frames: int,
    future_heading_anchor: str = "future_start",
) -> PseudoPosePayload:
    total_frames = int(context_frames + future_frames)
    if context_target_payload.packet_counter.shape[0] != total_frames:
        raise ValueError(
            f"context_target_payload length must be {total_frames}, got {context_target_payload.packet_counter.shape[0]}"
        )
    if future_payload.packet_counter.shape[0] < future_frames:
        raise ValueError(
            f"future_payload length must be >= {future_frames}, got {future_payload.packet_counter.shape[0]}"
        )
    if future_heading_anchor not in {"future_start", "context_end"}:
        raise ValueError(
            f"future_heading_anchor must be 'future_start' or 'context_end', got {future_heading_anchor}"
        )
    anchor_angles = root_heading_6d_to_angles(context_target_payload.root_heading_6d)
    future_delta_angles = root_heading_6d_to_angles(future_payload.root_heading_6d[:future_frames])
    if context_frames <= 0:
        anchor_angle = 0.0
    elif future_heading_anchor == "context_end":
        anchor_angle = float(anchor_angles[context_frames - 1])
    else:
        anchor_angle = float(anchor_angles[context_frames])
    anchored_future_heading = angles_to_root_heading_6d(anchor_angle + future_delta_angles)
    return PseudoPosePayload(
        packet_counter=context_target_payload.packet_counter.copy(),
        relative_positions=np.concatenate(
            [
                context_target_payload.relative_positions[:context_frames].copy(),
                future_payload.relative_positions[:future_frames].copy(),
            ],
            axis=0,
        ),
        root_heading_6d=np.concatenate(
            [
                context_target_payload.root_heading_6d[:context_frames].copy(),
                anchored_future_heading,
            ],
            axis=0,
        ),
        is_interpolated=np.concatenate(
            [
                context_target_payload.is_interpolated[:context_frames].copy(),
                future_payload.is_interpolated[:future_frames].copy(),
            ],
            axis=0,
        ),
        joint_names=list(context_target_payload.joint_names),
    )


@torch.no_grad()
def reconstruct_payload_window(
    *,
    model: torch.nn.Module,
    normalize_pose: bool,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    target_payload: PseudoPosePayload,
    device: torch.device,
    position_smoothing_kernel: tuple[float, ...] | None = None,
) -> PseudoPosePayload:
    pose_np = flatten_pose_window(
        target_payload.relative_positions.astype(np.float32),
        target_payload.root_heading_6d.astype(np.float32),
    )
    pose = torch.from_numpy(pose_np[None]).to(device=device, dtype=torch.float32)
    target = normalize_pose_tensor(pose, pose_mean, pose_std) if normalize_pose else pose
    reconstruction, _, _ = model(target)
    reconstruction = denormalize_pose_tensor(reconstruction, pose_mean, pose_std) if normalize_pose else reconstruction
    reconstruction = apply_position_temporal_filter(
        reconstruction,
        kernel_weights=position_smoothing_kernel,
    )
    reconstruction_np = reconstruction[0].detach().cpu().numpy().astype(np.float64)
    return PseudoPosePayload(
        packet_counter=target_payload.packet_counter.copy(),
        relative_positions=reconstruction_np[:, :POSE_POSITION_DIM].reshape(-1, 10, 3),
        root_heading_6d=reconstruction_np[:, POSE_POSITION_DIM:POSE_POSITION_DIM + 6],
        is_interpolated=target_payload.is_interpolated.copy(),
        joint_names=list(target_payload.joint_names),
    )


def compare_payloads(
    *,
    left_payload: PseudoPosePayload,
    right_payload: PseudoPosePayload,
    render_space: str,
) -> dict[str, Any]:
    left_positions, _, _, _, _, _ = build_render_arrays(left_payload, render_space=render_space)
    right_positions, _, _, _, _, _ = build_render_arrays(right_payload, render_space=render_space)
    return compute_frame_comparison_metrics(
        raw_positions=left_positions,
        pseudo_positions=right_positions,
        raw_root_heading_6d=left_payload.root_heading_6d,
        pseudo_root_heading_6d=right_payload.root_heading_6d,
    )


def render_payload_comparison_video(
    *,
    left_payload: PseudoPosePayload,
    right_payload: PseudoPosePayload,
    output_path: Path,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    left_title: str,
    right_title: str,
    progress_label: str,
    fps: int,
    point_size: float,
    transition_frame: int | None = None,
    pre_transition_label: str = "",
    post_transition_label: str = "",
    transition_banner: str = "",
    comparison_layout: str = "overlay",
) -> dict[str, Any]:
    comparison = compare_payloads(
        left_payload=left_payload,
        right_payload=right_payload,
        render_space=render_space,
    )
    left_positions, left_root_translation, left_rotation_matrices, left_body_points, left_back_line, left_belly_line = (
        build_render_arrays(left_payload, render_space=render_space)
    )
    right_positions, right_root_translation, right_rotation_matrices, right_body_points, right_back_line, right_belly_line = (
        build_render_arrays(right_payload, render_space=render_space)
    )
    left_vertices, smal_faces = build_selected_smal_mesh(
        body_model=body_model,
        smal_model=smal_model,
        smal_mapping=smal_mapping,
        smal_data=smal_data,
        smal_family_index=smal_family_index,
        positions=left_positions,
        body_points_by_name=left_body_points,
    )
    right_vertices, right_faces = build_selected_smal_mesh(
        body_model=body_model,
        smal_model=smal_model,
        smal_mapping=smal_mapping,
        smal_data=smal_data,
        smal_family_index=smal_family_index,
        positions=right_positions,
        body_points_by_name=right_body_points,
    )
    if smal_faces is None:
        smal_faces = right_faces
    render_kwargs = dict(
        output_path=output_path,
        frame_indices=np.arange(left_payload.packet_counter.shape[0], dtype=np.int64),
        packet_counter=left_payload.packet_counter,
        raw_positions=left_positions,
        pseudo_positions=right_positions,
        raw_body_points_by_name=left_body_points,
        pseudo_body_points_by_name=right_body_points,
        raw_back_line=left_back_line,
        pseudo_back_line=right_back_line,
        raw_belly_line=left_belly_line,
        pseudo_belly_line=right_belly_line,
        raw_root_translation=left_root_translation,
        pseudo_root_translation=right_root_translation,
        raw_rotation_matrices=left_rotation_matrices,
        pseudo_rotation_matrices=right_rotation_matrices,
        frame_mpjpe_m=np.asarray(comparison["frame_mpjpe_m"], dtype=np.float64),
        frame_heading_error_deg=np.asarray(comparison["frame_heading_error_deg"], dtype=np.float64),
        fps=fps,
        point_size=point_size,
        body_model=body_model,
        show_skeleton_overlay=show_skeleton_overlay,
        raw_smal_vertices=left_vertices,
        pseudo_smal_vertices=right_vertices,
        smal_faces=smal_faces,
        left_title=left_title,
        right_title=right_title,
        progress_label=progress_label,
        transition_frame=transition_frame,
        pre_transition_label=pre_transition_label,
        post_transition_label=post_transition_label,
        transition_banner=transition_banner,
    )
    if comparison_layout == "overlay":
        render_overlay_pose_comparison_animation(**render_kwargs)
    elif comparison_layout == "side_by_side":
        render_raw_vs_pseudo_animation(**render_kwargs)
    else:
        raise ValueError(f"Unsupported comparison_layout={comparison_layout!r}")
    return comparison


def diagnose_failure(
    *,
    raw_vs_pseudo_mean_heading_error_deg: float,
    raw_vs_pseudo_max_heading_error_deg: float,
    gt_vs_recon_mean_heading_error_deg: float,
    gt_vs_recon_max_heading_error_deg: float,
    gt_window_max_root_heading_delta_deg: float,
) -> str:
    upstream_noise = (
        raw_vs_pseudo_mean_heading_error_deg >= 25.0
        or raw_vs_pseudo_max_heading_error_deg >= 90.0
        or gt_window_max_root_heading_delta_deg >= 30.0
    )
    model_failure = (
        gt_vs_recon_mean_heading_error_deg >= raw_vs_pseudo_mean_heading_error_deg + 10.0
        or gt_vs_recon_max_heading_error_deg >= raw_vs_pseudo_max_heading_error_deg + 20.0
    )
    if upstream_noise and model_failure:
        return "mixed_upstream_and_model"
    if upstream_noise:
        return "upstream_pseudo_pose_noise_suspected"
    if model_failure:
        return "model_failure_suspected"
    return "unclear"


def export_failure_pack(
    *,
    checkpoint_path: Path,
    window_metrics_csv: Path,
    participant: str,
    filled_root: Path,
    output_dir: Path,
    top_k: int,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
    fps: int,
    point_size: float,
    device: str,
    no_video: bool,
    allow_missing_raw_reference: bool = False,
    position_smoothing_kernel: str = "tri5",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, pose_mean, pose_std, checkpoint = load_temporal_vae_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)
    workspace_root = Path(__file__).resolve().parents[1]
    windows = read_failure_windows(
        window_metrics_csv=window_metrics_csv,
        participant=participant,
        top_k=top_k,
        workspace_root=workspace_root,
    )
    summary_rows: list[dict[str, Any]] = []
    for rank, window in enumerate(windows, start=1):
        full_payload = load_pseudo_pose(window.pose_path)
        gt_segment_window = slice_payload(
            full_payload,
            start_frame=window.start_frame_20hz,
            num_frames=window.valid_frames,
            rebase_heading=False,
        )
        gt_rebased_window = slice_payload(
            full_payload,
            start_frame=window.start_frame_20hz,
            num_frames=window.valid_frames,
            rebase_heading=True,
        )
        recon_window = reconstruct_payload_window(
            model=model,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            target_payload=gt_rebased_window,
            device=device_resolved,
            position_smoothing_kernel=smoothing_kernel,
        )
        raw_reference_source = "filled_imu"
        try:
            raw_window = load_raw_imu_reference_pose(
                pose_path=window.pose_path,
                filled_root=filled_root,
                target_packet_counter=gt_segment_window.packet_counter,
                neutral_pose_mode=neutral_pose_mode,
                gyr_motion_scale=gyr_motion_scale,
            )
        except FileNotFoundError:
            if not allow_missing_raw_reference:
                raise
            raw_window = gt_segment_window
            raw_reference_source = "fallback_pseudo_pose"

        window_dir = output_dir / f"{rank:02d}__segment_{window.segment_id}__start_{window.start_frame_20hz}"
        save_payload_npz(window_dir / "pseudo_gt_segment_heading.npz", gt_segment_window)
        save_payload_npz(window_dir / "pseudo_gt_window_rebased.npz", gt_rebased_window)
        save_payload_npz(window_dir / "vae_reconstruction.npz", recon_window)
        save_payload_npz(window_dir / "raw_reference.npz", raw_window)

        raw_vs_pseudo = compare_payloads(
            left_payload=raw_window,
            right_payload=gt_segment_window,
            render_space=render_space,
        )
        gt_vs_recon = compare_payloads(
            left_payload=gt_rebased_window,
            right_payload=recon_window,
            render_space=render_space,
        )
        gt_temporal = compute_temporal_diagnostics(
            relative_positions=gt_segment_window.relative_positions,
            root_heading_6d=gt_segment_window.root_heading_6d,
        )
        diagnosis = diagnose_failure(
            raw_vs_pseudo_mean_heading_error_deg=float(raw_vs_pseudo["mean_heading_error_deg"]),
            raw_vs_pseudo_max_heading_error_deg=float(raw_vs_pseudo["max_heading_error_deg"]),
            gt_vs_recon_mean_heading_error_deg=float(gt_vs_recon["mean_heading_error_deg"]),
            gt_vs_recon_max_heading_error_deg=float(gt_vs_recon["max_heading_error_deg"]),
            gt_window_max_root_heading_delta_deg=float(gt_temporal["max_root_heading_delta_deg"]),
        )
        window_meta = asdict(window)
        window_meta["pose_path"] = str(window_meta["pose_path"])
        window_meta["feature_path"] = str(window_meta["feature_path"])

        raw_vs_pseudo_video = build_export_video_path(
            output_dir=output_dir,
            source_tag="temporal-vae-failure-raw-vs-pseudo",
            participant=window.participant,
            segment_id=window.segment_id,
            start_frame_20hz=window.start_frame_20hz,
            rank=rank,
        )
        gt_vs_recon_video = build_export_video_path(
            output_dir=output_dir,
            source_tag="temporal-vae-failure-gt-vs-recon",
            participant=window.participant,
            segment_id=window.segment_id,
            start_frame_20hz=window.start_frame_20hz,
            rank=rank,
        )
        if not no_video:
            render_payload_comparison_video(
                left_payload=raw_window,
                right_payload=gt_segment_window,
                output_path=raw_vs_pseudo_video,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Raw IMU reference",
                right_title="Pseudo-pose target",
                progress_label=f"Failure pack raw-vs-pseudo #{rank}",
                fps=fps,
                point_size=point_size,
            )
            render_payload_comparison_video(
                left_payload=gt_rebased_window,
                right_payload=recon_window,
                output_path=gt_vs_recon_video,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Pseudo-pose target (window-rebased)",
                right_title="Temporal VAE reconstruction",
                progress_label=f"Failure pack gt-vs-recon #{rank}",
                fps=fps,
                point_size=point_size,
            )

        meta = {
            **window_meta,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "render_space": render_space,
            "position_smoothing_kernel": position_smoothing_kernel,
            "raw_reference_source": raw_reference_source,
            "raw_vs_pseudo": {
                "mean_mpjpe_mm": float(raw_vs_pseudo["mean_mpjpe_m"] * 1000.0),
                "max_mpjpe_mm": float(raw_vs_pseudo["max_mpjpe_m"] * 1000.0),
                "mean_heading_error_deg": float(raw_vs_pseudo["mean_heading_error_deg"]),
                "max_heading_error_deg": float(raw_vs_pseudo["max_heading_error_deg"]),
            },
            "gt_vs_recon": {
                "mean_mpjpe_mm": float(gt_vs_recon["mean_mpjpe_m"] * 1000.0),
                "max_mpjpe_mm": float(gt_vs_recon["max_mpjpe_m"] * 1000.0),
                "mean_heading_error_deg": float(gt_vs_recon["mean_heading_error_deg"]),
                "max_heading_error_deg": float(gt_vs_recon["max_heading_error_deg"]),
            },
            "gt_window_motion": compute_motion_metrics(gt_segment_window),
            "gt_window_temporal": gt_temporal,
            "diagnosis": diagnosis,
        }
        write_json(window_dir / "window_summary.json", meta)
        summary_rows.append(
            {
                "participant": window.participant,
                "segment_id": window.segment_id,
                "start_frame_20hz": window.start_frame_20hz,
                "end_frame_20hz": window.start_frame_20hz + window.valid_frames - 1,
                "packet_start_20hz": window.packet_start_20hz,
                "packet_end_20hz": window.packet_end_20hz,
                "eval_recon_mpjpe": window.eval_recon_mpjpe,
                "eval_root_heading_error_deg": window.eval_root_heading_error_deg,
                "gt_window_max_root_heading_delta_deg": float(gt_temporal["max_root_heading_delta_deg"]),
                "raw_vs_pseudo_mean_mpjpe_mm": float(raw_vs_pseudo["mean_mpjpe_m"] * 1000.0),
                "raw_vs_pseudo_max_mpjpe_mm": float(raw_vs_pseudo["max_mpjpe_m"] * 1000.0),
                "raw_vs_pseudo_mean_heading_error_deg": float(raw_vs_pseudo["mean_heading_error_deg"]),
                "raw_vs_pseudo_max_heading_error_deg": float(raw_vs_pseudo["max_heading_error_deg"]),
                "gt_vs_recon_mean_mpjpe_mm": float(gt_vs_recon["mean_mpjpe_m"] * 1000.0),
                "gt_vs_recon_max_mpjpe_mm": float(gt_vs_recon["max_mpjpe_m"] * 1000.0),
                "gt_vs_recon_mean_heading_error_deg": float(gt_vs_recon["mean_heading_error_deg"]),
                "gt_vs_recon_max_heading_error_deg": float(gt_vs_recon["max_heading_error_deg"]),
                "diagnosis": diagnosis,
                "window_dir": str(window_dir),
                "raw_vs_pseudo_video": "" if no_video else str(raw_vs_pseudo_video),
                "gt_vs_recon_video": "" if no_video else str(gt_vs_recon_video),
            }
        )

    with (output_dir / "failure_pack_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FAILURE_PACK_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_metrics_csv": str(window_metrics_csv),
        "participant": participant,
        "top_k": len(summary_rows),
        "rows": summary_rows,
    }
    write_json(output_dir / "failure_pack_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export temporal VAE failure packs for selected worst windows")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-metrics-csv", type=Path, required=True)
    parser.add_argument("--participant", type=str, default="Bill-Coco")
    parser.add_argument("--filled-root", type=Path, default=root / "Data" / "IMU_New2_Filled")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--render-space", choices=["root", "heading"], default="heading")
    parser.add_argument("--body-model", choices=["shell", "smal"], default="smal")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=root / "SMAL" / "wolf_alph3.pkl")
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--neutral-pose-mode", choices=["first-frame", "sequence-median"], default="sequence-median")
    parser.add_argument("--gyr-motion-scale", type=float, default=0.8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--allow-missing-raw-reference", action="store_true")
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_failure_pack(
        checkpoint_path=args.checkpoint,
        window_metrics_csv=args.window_metrics_csv,
        participant=args.participant,
        filled_root=args.filled_root,
        output_dir=args.output_dir,
        top_k=args.top_k,
        render_space=args.render_space,
        body_model=args.body_model,
        show_skeleton_overlay=args.show_skeleton_overlay,
        smal_model=args.smal_model,
        smal_mapping=args.smal_mapping,
        smal_data=args.smal_data,
        smal_family_index=args.smal_family_index,
        neutral_pose_mode=args.neutral_pose_mode,
        gyr_motion_scale=args.gyr_motion_scale,
        fps=args.fps,
        point_size=args.point_size,
        device=args.device,
        no_video=args.no_video,
        allow_missing_raw_reference=args.allow_missing_raw_reference,
        position_smoothing_kernel=args.position_smoothing_kernel,
    )
    print(summary)


if __name__ == "__main__":
    main()
