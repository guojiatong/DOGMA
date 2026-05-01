#!/usr/bin/env python3
"""
Export held-out temporal VAE checkpoint comparison videos on shared windows.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from evaluate_temporal_vae import load_temporal_vae_checkpoint
from export_temporal_vae_failure_pack import (
    compare_payloads,
    reconstruct_payload_window,
    resolve_workspace_path,
    save_payload_npz,
    slice_payload,
)
from train_imu_masked_recon import write_json
from train_temporal_vae import resolve_position_smoothing_kernel
from video_export_paths import build_export_video_path
from visualize_dog_skeleton import BONES, JOINT_ORDER, ProgressPrinter, build_body_meshes, build_local_torso_faces, set_equal_axes, transform_faces
from visualize_pose import (
    PseudoPosePayload,
    build_render_arrays,
    build_selected_smal_mesh,
    load_pseudo_pose,
)


COMPARISON_COLUMNS = (
    "rank",
    "selection_mode",
    "participant",
    "segment_id",
    "start_frame_20hz",
    "end_frame_20hz",
    "packet_start_20hz",
    "packet_end_20hz",
    "reference_eval_recon_mpjpe",
    "reference_eval_root_heading_error_deg",
    "reference_eval_jerk_error",
    "candidate_eval_recon_mpjpe",
    "candidate_eval_root_heading_error_deg",
    "candidate_eval_jerk_error",
    "candidate_minus_reference_mpjpe",
    "candidate_minus_reference_heading_deg",
    "reference_minus_candidate_jerk",
    "reference_vs_gt_mean_mpjpe_mm",
    "reference_vs_gt_mean_heading_error_deg",
    "candidate_vs_gt_mean_mpjpe_mm",
    "candidate_vs_gt_mean_heading_error_deg",
    "window_dir",
    "video_path",
)


@dataclass(frozen=True)
class SharedWindow:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    start_frame_20hz: int
    valid_frames: int
    packet_start_20hz: int
    packet_end_20hz: int
    reference_eval_recon_mpjpe: float
    reference_eval_root_heading_error_deg: float
    reference_eval_position_rmse: float
    reference_eval_heading_rmse: float
    reference_eval_jerk_error: float
    candidate_eval_recon_mpjpe: float
    candidate_eval_root_heading_error_deg: float
    candidate_eval_position_rmse: float
    candidate_eval_heading_rmse: float
    candidate_eval_jerk_error: float

    @property
    def jerk_improvement(self) -> float:
        return float(self.reference_eval_jerk_error - self.candidate_eval_jerk_error)

    @property
    def mpjpe_delta(self) -> float:
        return float(self.candidate_eval_recon_mpjpe - self.reference_eval_recon_mpjpe)

    @property
    def heading_delta_deg(self) -> float:
        return float(self.candidate_eval_root_heading_error_deg - self.reference_eval_root_heading_error_deg)


def _window_key(row: dict[str, str]) -> tuple[str, str, int, int, int, int]:
    return (
        row["participant"],
        row["segment_id"],
        int(row["start_frame_20hz"]),
        int(row["valid_frames"]),
        int(row["packet_start_20hz"]),
        int(row["packet_end_20hz"]),
    )


def read_shared_windows(
    *,
    reference_window_metrics_csv: Path,
    candidate_window_metrics_csv: Path,
    workspace_root: Path,
) -> list[SharedWindow]:
    with reference_window_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    with candidate_window_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))

    candidate_by_key = {_window_key(row): row for row in candidate_rows}
    shared_windows: list[SharedWindow] = []
    for reference_row in reference_rows:
        key = _window_key(reference_row)
        candidate_row = candidate_by_key.get(key)
        if candidate_row is None:
            continue
        shared_windows.append(
            SharedWindow(
                participant=reference_row["participant"],
                segment_id=reference_row["segment_id"],
                pose_path=resolve_workspace_path(Path(reference_row["pose_path"]), workspace_root),
                feature_path=resolve_workspace_path(Path(reference_row["feature_path"]), workspace_root),
                start_frame_20hz=int(reference_row["start_frame_20hz"]),
                valid_frames=int(reference_row["valid_frames"]),
                packet_start_20hz=int(reference_row["packet_start_20hz"]),
                packet_end_20hz=int(reference_row["packet_end_20hz"]),
                reference_eval_recon_mpjpe=float(reference_row["recon_mpjpe"]),
                reference_eval_root_heading_error_deg=float(reference_row["root_heading_error_deg"]),
                reference_eval_position_rmse=float(reference_row["position_rmse"]),
                reference_eval_heading_rmse=float(reference_row["heading_rmse"]),
                reference_eval_jerk_error=float(reference_row["jerk_error"]),
                candidate_eval_recon_mpjpe=float(candidate_row["recon_mpjpe"]),
                candidate_eval_root_heading_error_deg=float(candidate_row["root_heading_error_deg"]),
                candidate_eval_position_rmse=float(candidate_row["position_rmse"]),
                candidate_eval_heading_rmse=float(candidate_row["heading_rmse"]),
                candidate_eval_jerk_error=float(candidate_row["jerk_error"]),
            )
        )
    if not shared_windows:
        raise ValueError("No shared windows found between reference and candidate metrics CSVs")
    return shared_windows


def select_windows(
    windows: list[SharedWindow],
    *,
    selection_mode: str,
    top_k: int,
    max_per_participant: int,
    require_positive_jerk_gain: bool = False,
    require_positive_mpjpe_gain: bool = False,
    max_heading_delta_deg: float | None = None,
) -> list[SharedWindow]:
    filtered_windows = list(windows)
    if require_positive_jerk_gain:
        filtered_windows = [row for row in filtered_windows if row.jerk_improvement > 0.0]
    if require_positive_mpjpe_gain:
        filtered_windows = [row for row in filtered_windows if row.mpjpe_delta < 0.0]
    if max_heading_delta_deg is not None:
        filtered_windows = [row for row in filtered_windows if row.heading_delta_deg <= float(max_heading_delta_deg)]

    if selection_mode == "jerk_improvement":
        sorted_windows = sorted(
            filtered_windows,
            key=lambda row: (row.jerk_improvement, -row.heading_delta_deg, -row.mpjpe_delta),
            reverse=True,
        )
    elif selection_mode == "balanced_improvement":
        sorted_windows = sorted(
            filtered_windows,
            key=lambda row: (row.jerk_improvement, -row.mpjpe_delta, -row.heading_delta_deg),
            reverse=True,
        )
    elif selection_mode == "heading_regression":
        sorted_windows = sorted(
            filtered_windows,
            key=lambda row: row.heading_delta_deg,
            reverse=True,
        )
    elif selection_mode == "mpjpe_regression":
        sorted_windows = sorted(
            filtered_windows,
            key=lambda row: row.mpjpe_delta,
            reverse=True,
        )
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    selected: list[SharedWindow] = []
    counts_by_participant: dict[str, int] = {}
    for window in sorted_windows:
        if max_per_participant > 0 and counts_by_participant.get(window.participant, 0) >= max_per_participant:
            continue
        selected.append(window)
        counts_by_participant[window.participant] = counts_by_participant.get(window.participant, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def render_threeway_pose_comparison(
    *,
    output_path: Path,
    gt_payload: PseudoPosePayload,
    reference_payload: PseudoPosePayload,
    candidate_payload: PseudoPosePayload,
    reference_label: str,
    candidate_label: str,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    fps: int,
    point_size: float,
    progress_label: str,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gt_vs_reference = compare_payloads(left_payload=gt_payload, right_payload=reference_payload, render_space=render_space)
    gt_vs_candidate = compare_payloads(left_payload=gt_payload, right_payload=candidate_payload, render_space=render_space)

    payloads = [gt_payload, reference_payload, candidate_payload]
    titles = [
        "Pseudo-pose target (window-rebased)",
        reference_label,
        candidate_label,
    ]
    side_arrays = []
    all_positions: list[np.ndarray] = []
    all_vertices: list[np.ndarray] = []
    smal_faces = None
    for payload in payloads:
        positions, root_translation, rotation_matrices, body_points, back_line, belly_line = build_render_arrays(
            payload=payload,
            render_space=render_space,
        )
        side_arrays.append(
            (
                positions,
                root_translation,
                rotation_matrices,
                body_points,
                back_line,
                belly_line,
            )
        )
        all_positions.append(positions)
        vertices, faces = build_selected_smal_mesh(
            body_model=body_model,
            smal_model=smal_model,
            smal_mapping=smal_mapping,
            smal_data=smal_data,
            smal_family_index=smal_family_index,
            positions=positions,
            body_points_by_name=body_points,
        )
        if vertices is not None:
            all_vertices.append(vertices)
        if smal_faces is None and faces is not None:
            smal_faces = faces

    fig = plt.figure(figsize=(24, 8))
    axes = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132, projection="3d"), fig.add_subplot(133, projection="3d")]
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    local_torso_faces = build_local_torso_faces() if body_model == "shell" else []
    overlay_enabled = body_model == "shell" or show_skeleton_overlay
    overlay_alpha = 0.95 if body_model == "shell" else 0.18

    axis_limits_source = np.concatenate(all_vertices if body_model == "smal" and all_vertices else all_positions, axis=0)
    side_state: list[dict[str, Any]] = []
    for axis, title, arrays in zip(axes, titles, side_arrays):
        positions, _, _, _, _, _ = arrays
        state: dict[str, Any] = {"axis": axis}
        if overlay_enabled:
            state["line_artists"] = [
                axis.plot([], [], [], linewidth=1.2, color="#1f2937", alpha=overlay_alpha)[0]
                for _ in BONES
            ]
            state["dorsal_artist"] = axis.plot([], [], [], linewidth=3.2, color="#4b3621", alpha=overlay_alpha)[0]
            state["ventral_artist"] = axis.plot([], [], [], linewidth=2.4, color="#7c5c3b", alpha=overlay_alpha)[0]
            state["scatter"] = axis.scatter(
                [],
                [],
                [],
                s=point_size * (0.45 if body_model == "shell" else 0.28),
                c="#c2410c",
                depthshade=True,
                alpha=overlay_alpha,
            )
        if body_model == "shell":
            state["torso_collection"] = Poly3DCollection([], facecolors="#8b6b4a", edgecolors="none", alpha=0.36)
            state["limb_collection"] = Poly3DCollection([], facecolors="#a67c52", edgecolors="none", alpha=0.28)
            state["head_collection"] = Poly3DCollection([], facecolors="#6f4e37", edgecolors="none", alpha=0.42)
            state["ear_collection"] = Poly3DCollection([], facecolors="#5b3a29", edgecolors="none", alpha=0.60)
            for key in ("torso_collection", "limb_collection", "head_collection", "ear_collection"):
                axis.add_collection3d(state[key])
        else:
            if smal_faces is None:
                raise ValueError("SMAL comparison rendering requires faces")
            state["smal_collection"] = Poly3DCollection([], facecolors="#9c7a56", edgecolors="none", alpha=0.62)
            axis.add_collection3d(state["smal_collection"])
        set_equal_axes(axis, axis_limits_source)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_title(title)
        axis.view_init(elev=18, azim=-60)
        axis.grid(True, alpha=0.35)
        side_state.append(state)

    metric_text = fig.text(
        0.5,
        0.975,
        "",
        ha="center",
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 5},
    )

    def update(frame_id: int):
        artists = [metric_text]
        for state, arrays in zip(side_state, side_arrays):
            positions, root_translation, rotation_matrices, body_points_by_name, back_line, belly_line = arrays
            current = positions[frame_id]
            current_body_points = {
                name: point_series[frame_id]
                for name, point_series in body_points_by_name.items()
            }
            if body_model == "shell":
                state["torso_collection"].set_verts(
                    transform_faces(
                        local_faces=local_torso_faces,
                        rotation_matrix=rotation_matrices[frame_id],
                        translation=root_translation[frame_id],
                    )
                )
                limb_faces, head_faces, ear_faces = build_body_meshes(
                    current=current,
                    current_body_points=current_body_points,
                    joint_index=joint_index,
                )
                state["limb_collection"].set_verts(limb_faces)
                state["head_collection"].set_verts(head_faces)
                state["ear_collection"].set_verts(ear_faces)
                artists.extend(
                    [
                        state["torso_collection"],
                        state["limb_collection"],
                        state["head_collection"],
                        state["ear_collection"],
                    ]
                )
            else:
                state["smal_collection"].set_verts(all_vertices[side_state.index(state)][frame_id][smal_faces])
                artists.append(state["smal_collection"])

            if overlay_enabled:
                state["dorsal_artist"].set_data(back_line[frame_id, :, 0], back_line[frame_id, :, 1])
                state["dorsal_artist"].set_3d_properties(back_line[frame_id, :, 2])
                state["ventral_artist"].set_data(belly_line[frame_id, :, 0], belly_line[frame_id, :, 1])
                state["ventral_artist"].set_3d_properties(belly_line[frame_id, :, 2])
                for artist, (parent, child) in zip(state["line_artists"], BONES):
                    parent_xyz = current[joint_index[parent]]
                    child_xyz = current[joint_index[child]]
                    artist.set_data([parent_xyz[0], child_xyz[0]], [parent_xyz[1], child_xyz[1]])
                    artist.set_3d_properties([parent_xyz[2], child_xyz[2]])
                state["scatter"]._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
                artists.extend(
                    state["line_artists"] + [state["dorsal_artist"], state["ventral_artist"], state["scatter"]]
                )

        metric_text.set_text(
            "frame={0} packet={1}\n{2}: MPJPE={3:.3f} mm | heading={4:.3f} deg    {5}: MPJPE={6:.3f} mm | heading={7:.3f} deg".format(
                int(gt_payload.packet_counter[frame_id] - gt_payload.packet_counter[0]),
                int(gt_payload.packet_counter[frame_id]),
                reference_label,
                float(gt_vs_reference["frame_mpjpe_m"][frame_id] * 1000.0),
                float(gt_vs_reference["frame_heading_error_deg"][frame_id]),
                candidate_label,
                float(gt_vs_candidate["frame_mpjpe_m"][frame_id] * 1000.0),
                float(gt_vs_candidate["frame_heading_error_deg"][frame_id]),
            )
        )
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=gt_payload.packet_counter.shape[0],
        interval=1000 / max(fps, 1),
        blit=False,
    )
    progress = ProgressPrinter(progress_label, int(gt_payload.packet_counter.shape[0]))
    progress_callback = lambda current_frame, total_frames: progress.update(current_frame + 1)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), progress_callback=progress_callback)
    elif suffix == ".mp4":
        animation.save(output_path, writer=FFMpegWriter(fps=fps), progress_callback=progress_callback)
    else:
        raise ValueError("Comparison video output must end with .mp4 or .gif")
    plt.close(fig)
    return {
        "reference_vs_gt": gt_vs_reference,
        "candidate_vs_gt": gt_vs_candidate,
    }


def export_checkpoint_comparison_batch(
    *,
    reference_checkpoint: Path,
    reference_window_metrics_csv: Path,
    candidate_checkpoint: Path,
    candidate_window_metrics_csv: Path,
    output_dir: Path,
    top_k: int,
    selection_mode: str,
    max_per_participant: int,
    require_positive_jerk_gain: bool,
    require_positive_mpjpe_gain: bool,
    max_heading_delta_deg: float | None,
    reference_label: str,
    candidate_label: str,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    fps: int,
    point_size: float,
    device: str,
    no_video: bool,
    reference_position_smoothing_kernel: str = "tri5",
    candidate_position_smoothing_kernel: str = "tri5",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(__file__).resolve().parents[1]
    shared_windows = read_shared_windows(
        reference_window_metrics_csv=reference_window_metrics_csv,
        candidate_window_metrics_csv=candidate_window_metrics_csv,
        workspace_root=workspace_root,
    )
    selected_windows = select_windows(
        shared_windows,
        selection_mode=selection_mode,
        top_k=top_k,
        max_per_participant=max_per_participant,
        require_positive_jerk_gain=require_positive_jerk_gain,
        require_positive_mpjpe_gain=require_positive_mpjpe_gain,
        max_heading_delta_deg=max_heading_delta_deg,
    )
    if not selected_windows:
        raise ValueError("No windows selected for checkpoint comparison export")

    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_model, reference_config, reference_mean, reference_std, reference_checkpoint_payload = load_temporal_vae_checkpoint(
        checkpoint_path=reference_checkpoint,
        device=device_resolved,
    )
    candidate_model, candidate_config, candidate_mean, candidate_std, candidate_checkpoint_payload = load_temporal_vae_checkpoint(
        checkpoint_path=candidate_checkpoint,
        device=device_resolved,
    )
    reference_smoothing_kernel = resolve_position_smoothing_kernel(reference_position_smoothing_kernel)
    candidate_smoothing_kernel = resolve_position_smoothing_kernel(candidate_position_smoothing_kernel)

    summary_rows: list[dict[str, Any]] = []
    for rank, window in enumerate(selected_windows, start=1):
        full_payload = load_pseudo_pose(window.pose_path)
        gt_window = slice_payload(
            full_payload,
            start_frame=window.start_frame_20hz,
            num_frames=window.valid_frames,
            rebase_heading=True,
        )
        reference_recon = reconstruct_payload_window(
            model=reference_model,
            normalize_pose=reference_config.normalize_pose,
            pose_mean=reference_mean,
            pose_std=reference_std,
            target_payload=gt_window,
            device=device_resolved,
            position_smoothing_kernel=reference_smoothing_kernel,
        )
        candidate_recon = reconstruct_payload_window(
            model=candidate_model,
            normalize_pose=candidate_config.normalize_pose,
            pose_mean=candidate_mean,
            pose_std=candidate_std,
            target_payload=gt_window,
            device=device_resolved,
            position_smoothing_kernel=candidate_smoothing_kernel,
        )

        window_dir = output_dir / f"{rank:02d}__{window.participant}__segment_{window.segment_id}__start_{window.start_frame_20hz}"
        save_payload_npz(window_dir / "pseudo_gt_window_rebased.npz", gt_window)
        save_payload_npz(window_dir / "reference_reconstruction.npz", reference_recon)
        save_payload_npz(window_dir / "candidate_reconstruction.npz", candidate_recon)

        reference_vs_gt = compare_payloads(left_payload=gt_window, right_payload=reference_recon, render_space=render_space)
        candidate_vs_gt = compare_payloads(left_payload=gt_window, right_payload=candidate_recon, render_space=render_space)
        video_path = build_export_video_path(
            output_dir=output_dir,
            source_tag="temporal-vae-compare",
            participant=window.participant,
            segment_id=window.segment_id,
            start_frame_20hz=window.start_frame_20hz,
            rank=rank,
            extra_tags=(selection_mode, reference_label, candidate_label),
        )
        if not no_video:
            render_threeway_pose_comparison(
                output_path=video_path,
                gt_payload=gt_window,
                reference_payload=reference_recon,
                candidate_payload=candidate_recon,
                reference_label=reference_label,
                candidate_label=candidate_label,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                fps=fps,
                point_size=point_size,
                progress_label=f"Rendering temporal VAE comparison #{rank}",
            )

        window_meta = asdict(window)
        window_meta["pose_path"] = str(window_meta["pose_path"])
        window_meta["feature_path"] = str(window_meta["feature_path"])
        meta = {
            **window_meta,
            "selection_mode": selection_mode,
            "reference_label": reference_label,
            "candidate_label": candidate_label,
            "reference_checkpoint": str(reference_checkpoint),
            "reference_checkpoint_epoch": int(reference_checkpoint_payload.get("epoch", -1)),
            "candidate_checkpoint": str(candidate_checkpoint),
            "candidate_checkpoint_epoch": int(candidate_checkpoint_payload.get("epoch", -1)),
            "reference_position_smoothing_kernel": reference_position_smoothing_kernel,
            "candidate_position_smoothing_kernel": candidate_position_smoothing_kernel,
            "reference_vs_gt": {
                "mean_mpjpe_mm": float(reference_vs_gt["mean_mpjpe_m"] * 1000.0),
                "max_mpjpe_mm": float(reference_vs_gt["max_mpjpe_m"] * 1000.0),
                "mean_heading_error_deg": float(reference_vs_gt["mean_heading_error_deg"]),
                "max_heading_error_deg": float(reference_vs_gt["max_heading_error_deg"]),
            },
            "candidate_vs_gt": {
                "mean_mpjpe_mm": float(candidate_vs_gt["mean_mpjpe_m"] * 1000.0),
                "max_mpjpe_mm": float(candidate_vs_gt["max_mpjpe_m"] * 1000.0),
                "mean_heading_error_deg": float(candidate_vs_gt["mean_heading_error_deg"]),
                "max_heading_error_deg": float(candidate_vs_gt["max_heading_error_deg"]),
            },
        }
        write_json(window_dir / "window_summary.json", meta)

        summary_rows.append(
            {
                "rank": rank,
                "selection_mode": selection_mode,
                "participant": window.participant,
                "segment_id": window.segment_id,
                "start_frame_20hz": window.start_frame_20hz,
                "end_frame_20hz": window.start_frame_20hz + window.valid_frames - 1,
                "packet_start_20hz": window.packet_start_20hz,
                "packet_end_20hz": window.packet_end_20hz,
                "reference_eval_recon_mpjpe": window.reference_eval_recon_mpjpe,
                "reference_eval_root_heading_error_deg": window.reference_eval_root_heading_error_deg,
                "reference_eval_jerk_error": window.reference_eval_jerk_error,
                "candidate_eval_recon_mpjpe": window.candidate_eval_recon_mpjpe,
                "candidate_eval_root_heading_error_deg": window.candidate_eval_root_heading_error_deg,
                "candidate_eval_jerk_error": window.candidate_eval_jerk_error,
                "candidate_minus_reference_mpjpe": window.mpjpe_delta,
                "candidate_minus_reference_heading_deg": window.heading_delta_deg,
                "reference_minus_candidate_jerk": window.jerk_improvement,
                "reference_vs_gt_mean_mpjpe_mm": float(reference_vs_gt["mean_mpjpe_m"] * 1000.0),
                "reference_vs_gt_mean_heading_error_deg": float(reference_vs_gt["mean_heading_error_deg"]),
                "candidate_vs_gt_mean_mpjpe_mm": float(candidate_vs_gt["mean_mpjpe_m"] * 1000.0),
                "candidate_vs_gt_mean_heading_error_deg": float(candidate_vs_gt["mean_heading_error_deg"]),
                "window_dir": str(window_dir),
                "video_path": "" if no_video else str(video_path),
            }
        )

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "reference_checkpoint": str(reference_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "reference_window_metrics_csv": str(reference_window_metrics_csv),
        "candidate_window_metrics_csv": str(candidate_window_metrics_csv),
        "selection_mode": selection_mode,
        "top_k": len(summary_rows),
        "rows": summary_rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export temporal VAE checkpoint comparison videos on shared held-out windows")
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-window-metrics-csv", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-window-metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--selection-mode",
        choices=("jerk_improvement", "balanced_improvement", "heading_regression", "mpjpe_regression"),
        default="jerk_improvement",
    )
    parser.add_argument("--max-per-participant", type=int, default=3)
    parser.add_argument("--require-positive-jerk-gain", action="store_true")
    parser.add_argument("--require-positive-mpjpe-gain", action="store_true")
    parser.add_argument("--max-heading-delta-deg", type=float, default=None)
    parser.add_argument("--reference-label", type=str, default="Reference checkpoint")
    parser.add_argument("--candidate-label", type=str, default="Candidate checkpoint")
    parser.add_argument("--render-space", choices=("root", "heading"), default="heading")
    parser.add_argument("--body-model", choices=("shell", "smal"), default="smal")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=root / "SMAL" / "wolf_alph3.pkl")
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--reference-position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--candidate-position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_checkpoint_comparison_batch(
        reference_checkpoint=args.reference_checkpoint,
        reference_window_metrics_csv=args.reference_window_metrics_csv,
        candidate_checkpoint=args.candidate_checkpoint,
        candidate_window_metrics_csv=args.candidate_window_metrics_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        selection_mode=args.selection_mode,
        max_per_participant=args.max_per_participant,
        require_positive_jerk_gain=args.require_positive_jerk_gain,
        require_positive_mpjpe_gain=args.require_positive_mpjpe_gain,
        max_heading_delta_deg=args.max_heading_delta_deg,
        reference_label=args.reference_label,
        candidate_label=args.candidate_label,
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
        reference_position_smoothing_kernel=args.reference_position_smoothing_kernel,
        candidate_position_smoothing_kernel=args.candidate_position_smoothing_kernel,
    )
    print(summary)


if __name__ == "__main__":
    main()
