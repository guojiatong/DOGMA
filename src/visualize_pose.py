#!/usr/bin/env python3
"""
Visualize exported 20Hz pseudo-pose files.

This script reads Data/IMU_Only_20Hz_v1/pseudo_pose_20hz/*.npz files and renders
the same simplified shell/skeleton view used by visualize_dog_skeleton.py.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from video_export_paths import build_export_video_path
from visualize_dog_skeleton import (
    BONES,
    DEFAULT_GYR_MOTION_SCALE,
    IMU_RATE_HZ,
    JOINT_ORDER,
    LOCAL_BACK_LINE,
    LOCAL_BELLY_LINE,
    LOCAL_BODY_POINTS,
    ProgressPrinter,
    bind_template_mesh_model,
    build_body_meshes,
    build_heading_rotation_matrices,
    build_local_torso_faces,
    build_official_smal_mesh_sequence,
    build_relative_positions,
    build_template_mesh_sequence,
    compute_pose_audit,
    downsample_motion_data,
    find_most_active_window,
    is_smal_preset_payload,
    load_filled_motion_data,
    load_official_smal_model,
    load_pickle_with_fake_chumpy,
    load_smal_mapping,
    load_smal_preset,
    load_template_mesh_model,
    render_animation,
    resolve_official_smal_model_path,
    resolve_smal_data_path,
    rotation_matrices_to_rot6d,
    select_frames,
    set_equal_axes,
    stack_joint_is_interpolated,
    transform_faces,
    transform_dynamic_points,
    transform_static_points,
)


VISUAL_AUDIT_COLUMNS = (
    "participant",
    "segment_id",
    "pose_path",
    "video_path",
    "render_space",
    "num_frames_total",
    "frame_start",
    "frame_end",
    "packet_start",
    "packet_end",
    "finite_ratio",
    "interp_ratio",
    "max_joint_interp_ratio",
    "worst_interp_joint",
    "full_preview_status",
    "selected_finite_ratio",
    "selected_interp_ratio",
    "selected_max_joint_interp_ratio",
    "selected_worst_interp_joint",
    "selected_preview_status",
    "left_right_sign_consistency",
    "distal_below_parent_consistency",
    "preview_status",
    "mean_joint_motion",
    "mean_foot_motion",
    "mean_root_heading_delta",
    "p95_joint_step",
    "max_joint_step",
    "p95_joint_jerk",
    "max_joint_jerk",
    "p95_root_heading_delta_deg",
    "max_root_heading_delta_deg",
    "selected_window_motion",
    "selected_p95_joint_step",
    "selected_max_joint_step",
    "selected_p95_joint_jerk",
    "selected_max_joint_jerk",
    "selected_p95_root_heading_delta_deg",
    "selected_max_root_heading_delta_deg",
    "compare_mean_mpjpe_mm",
    "compare_max_mpjpe_mm",
    "compare_mean_heading_error_deg",
    "compare_max_heading_error_deg",
)


@dataclass(frozen=True)
class PseudoPosePayload:
    packet_counter: np.ndarray
    relative_positions: np.ndarray
    root_heading_6d: np.ndarray
    is_interpolated: np.ndarray
    joint_names: list[str]


def ffmpeg_encoder_available(encoder_name: str) -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return encoder_name in result.stdout


def build_hardware_ffmpeg_writer(fps: int) -> FFMpegWriter:
    preferred_codec = os.environ.get("DOGMA_FFMPEG_CODEC", "").strip()
    if preferred_codec and ffmpeg_encoder_available(preferred_codec):
        return FFMpegWriter(
            fps=fps,
            codec=preferred_codec,
            extra_args=["-pix_fmt", "yuv420p"],
        )
    if ffmpeg_encoder_available("h264_nvenc"):
        return FFMpegWriter(
            fps=fps,
            codec="h264_nvenc",
            extra_args=["-preset", "p4", "-pix_fmt", "yuv420p"],
        )
    if ffmpeg_encoder_available("h264_videotoolbox"):
        return FFMpegWriter(
            fps=fps,
            codec="h264_videotoolbox",
            extra_args=["-pix_fmt", "yuv420p"],
        )
    return FFMpegWriter(fps=fps)


def normalize_rows(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    output = np.broadcast_to(fallback.astype(np.float64), values.shape).copy()
    output[valid] = values[valid] / norms[valid]
    return output


def rot6d_to_rotation_matrices(root_heading_6d: np.ndarray) -> np.ndarray:
    root_heading_6d = np.asarray(root_heading_6d, dtype=np.float64)
    if root_heading_6d.ndim != 2 or root_heading_6d.shape[1] != 6:
        raise ValueError(f"root_heading_6d must have shape [T,6], got {root_heading_6d.shape}")

    x_axis = normalize_rows(root_heading_6d[:, :3], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_raw = root_heading_6d[:, 3:6]
    z_axis = normalize_rows(np.cross(x_axis, y_raw), np.array([0.0, 0.0, 1.0], dtype=np.float64))
    y_axis = normalize_rows(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    return np.stack([x_axis, y_axis, z_axis], axis=2)


def infer_pose_metadata(pose_path: Path) -> tuple[str, str]:
    participant = pose_path.parent.name
    stem = pose_path.stem
    if not stem.startswith("segment_"):
        raise ValueError(f"Expected pseudo-pose filename segment_<id>.npz, got {pose_path.name}")
    return participant, stem.removeprefix("segment_")


def load_pseudo_pose(path: Path) -> PseudoPosePayload:
    with np.load(path, allow_pickle=False) as payload:
        required = {"packet_counter", "relative_positions", "root_heading_6d", "joint_names", "is_interpolated"}
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Pseudo-pose file missing required keys {missing}: {path}")

        packet_counter = payload["packet_counter"].astype(np.int64)
        relative_positions = payload["relative_positions"].astype(np.float64)
        root_heading_6d = payload["root_heading_6d"].astype(np.float64)
        is_interpolated = payload["is_interpolated"].astype(bool)
        joint_names = [str(name) for name in payload["joint_names"].tolist()]

    if joint_names != list(JOINT_ORDER):
        raise ValueError(f"Unexpected joint order in {path}: {joint_names}")
    if relative_positions.shape != (packet_counter.shape[0], len(JOINT_ORDER), 3):
        raise ValueError(f"relative_positions shape does not match packet axis in {path}")
    if root_heading_6d.shape != (packet_counter.shape[0], 6):
        raise ValueError(f"root_heading_6d shape does not match packet axis in {path}")
    if is_interpolated.shape != (packet_counter.shape[0], len(JOINT_ORDER)):
        raise ValueError(f"is_interpolated shape does not match packet axis in {path}")

    return PseudoPosePayload(
        packet_counter=packet_counter,
        relative_positions=relative_positions,
        root_heading_6d=root_heading_6d,
        is_interpolated=is_interpolated,
        joint_names=joint_names,
    )


def select_packet_axis(
    *,
    source_packet_counter: np.ndarray,
    target_packet_counter: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    source_index_by_packet = {
        int(packet): index
        for index, packet in enumerate(np.asarray(source_packet_counter, dtype=np.int64).tolist())
    }
    selected_indices = []
    for packet in np.asarray(target_packet_counter, dtype=np.int64).tolist():
        packet_int = int(packet)
        if packet_int not in source_index_by_packet:
            raise ValueError(f"Raw IMU packet axis does not contain pseudo-pose packet {packet_int}")
        selected_indices.append(source_index_by_packet[packet_int])
    indices = np.asarray(selected_indices, dtype=np.int64)
    return {name: value[indices] for name, value in arrays.items()}


def load_raw_imu_reference_pose(
    *,
    pose_path: Path,
    filled_root: Path,
    target_packet_counter: np.ndarray,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
) -> PseudoPosePayload:
    participant, segment_id = infer_pose_metadata(pose_path)
    segment_dir = filled_root / participant / segment_id
    packet_counter, sensor_data_by_joint = load_filled_motion_data(segment_dir)
    packet_counter, sensor_data_by_joint, _ = downsample_motion_data(
        packet_counter=packet_counter,
        sensor_data_by_joint=sensor_data_by_joint,
        source_rate_hz=float(IMU_RATE_HZ),
        target_rate_hz=20.0,
    )
    relative_positions = build_relative_positions(
        sensor_data_by_joint=sensor_data_by_joint,
        neutral_pose_mode=neutral_pose_mode,
        gyr_motion_scale=gyr_motion_scale,
    )
    root_heading_6d = rotation_matrices_to_rot6d(
        build_heading_rotation_matrices(sensor_data_by_joint["stern"]["quat"])
    )
    is_interpolated = stack_joint_is_interpolated(sensor_data_by_joint)
    aligned = select_packet_axis(
        source_packet_counter=packet_counter,
        target_packet_counter=target_packet_counter,
        arrays={
            "relative_positions": relative_positions,
            "root_heading_6d": root_heading_6d,
            "is_interpolated": is_interpolated,
        },
    )
    return PseudoPosePayload(
        packet_counter=np.asarray(target_packet_counter, dtype=np.int64),
        relative_positions=aligned["relative_positions"].astype(np.float64),
        root_heading_6d=aligned["root_heading_6d"].astype(np.float64),
        is_interpolated=aligned["is_interpolated"].astype(bool),
        joint_names=list(JOINT_ORDER),
    )


def compute_motion_metrics(payload: PseudoPosePayload) -> dict[str, float | str]:
    audit = compute_pose_audit(
        relative_positions=payload.relative_positions,
        is_interpolated=payload.is_interpolated,
    )
    temporal_diagnostics = compute_temporal_diagnostics(
        relative_positions=payload.relative_positions,
        root_heading_6d=payload.root_heading_6d,
    )
    if payload.relative_positions.shape[0] <= 1:
        mean_joint_motion = 0.0
        mean_foot_motion = 0.0
        mean_root_heading_delta = 0.0
    else:
        frame_delta = np.linalg.norm(np.diff(payload.relative_positions, axis=0), axis=2)
        foot_indices = [JOINT_ORDER.index(name) for name in ("left_hand", "right_hand", "left_foot", "right_foot")]
        mean_joint_motion = float(frame_delta.mean())
        mean_foot_motion = float(frame_delta[:, foot_indices].mean())
        mean_root_heading_delta = float(np.linalg.norm(np.diff(payload.root_heading_6d, axis=0), axis=1).mean())

    return {
        **audit,
        "mean_joint_motion": mean_joint_motion,
        "mean_foot_motion": mean_foot_motion,
        "mean_root_heading_delta": mean_root_heading_delta,
        **temporal_diagnostics,
    }


def build_render_arrays(
    payload: PseudoPosePayload,
    render_space: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    root_translation = np.zeros((payload.relative_positions.shape[0], 3), dtype=np.float64)
    if render_space == "root":
        rotation_matrices = np.broadcast_to(
            np.eye(3, dtype=np.float64),
            (payload.relative_positions.shape[0], 3, 3),
        ).copy()
        positions = payload.relative_positions.copy()
    elif render_space == "heading":
        rotation_matrices = rot6d_to_rotation_matrices(payload.root_heading_6d)
        positions = transform_dynamic_points(
            rotation_matrices=rotation_matrices,
            translation=root_translation,
            local_points=payload.relative_positions,
        )
    else:
        raise ValueError(f"Unsupported render_space: {render_space}")

    body_point_names = list(LOCAL_BODY_POINTS.keys())
    transformed_body_points = transform_static_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=np.asarray([LOCAL_BODY_POINTS[name] for name in body_point_names], dtype=np.float64),
    )
    body_points_by_name = {
        name: transformed_body_points[:, index]
        for index, name in enumerate(body_point_names)
    }
    back_line = transform_static_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=LOCAL_BACK_LINE,
    )
    belly_line = transform_static_points(
        rotation_matrices=rotation_matrices,
        translation=root_translation,
        local_points=LOCAL_BELLY_LINE,
    )
    return positions, root_translation, rotation_matrices, body_points_by_name, back_line, belly_line


def compute_frame_comparison_metrics(
    *,
    raw_positions: np.ndarray,
    pseudo_positions: np.ndarray,
    raw_root_heading_6d: np.ndarray,
    pseudo_root_heading_6d: np.ndarray,
) -> dict[str, np.ndarray | float]:
    if raw_positions.shape != pseudo_positions.shape:
        raise ValueError(f"Position shapes differ: raw={raw_positions.shape}, pseudo={pseudo_positions.shape}")
    frame_mpjpe_m = np.linalg.norm(raw_positions - pseudo_positions, axis=2).mean(axis=1)

    raw_rotation = rot6d_to_rotation_matrices(raw_root_heading_6d)
    pseudo_rotation = rot6d_to_rotation_matrices(pseudo_root_heading_6d)
    raw_forward = raw_rotation[:, :, 0]
    pseudo_forward = pseudo_rotation[:, :, 0]
    dot = np.sum(raw_forward * pseudo_forward, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    frame_heading_error_deg = np.degrees(np.arccos(dot))

    return {
        "frame_mpjpe_m": frame_mpjpe_m,
        "frame_heading_error_deg": frame_heading_error_deg,
        "mean_mpjpe_m": float(frame_mpjpe_m.mean()),
        "max_mpjpe_m": float(frame_mpjpe_m.max()),
        "mean_heading_error_deg": float(frame_heading_error_deg.mean()),
        "max_heading_error_deg": float(frame_heading_error_deg.max()),
    }


def compute_heading_delta_degrees(root_heading_6d: np.ndarray) -> np.ndarray:
    if root_heading_6d.shape[0] <= 1:
        return np.zeros((0,), dtype=np.float64)
    rotation_matrices = rot6d_to_rotation_matrices(root_heading_6d)
    forward_vectors = rotation_matrices[:, :, 0]
    dot = np.sum(forward_vectors[1:] * forward_vectors[:-1], axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def compute_temporal_diagnostics(
    *,
    relative_positions: np.ndarray,
    root_heading_6d: np.ndarray,
) -> dict[str, float]:
    if relative_positions.shape[0] <= 1:
        frame_delta = np.zeros((0, relative_positions.shape[1]), dtype=np.float64)
    else:
        frame_delta = np.linalg.norm(np.diff(relative_positions, axis=0), axis=2)

    if relative_positions.shape[0] <= 2:
        frame_jerk = np.zeros((0, relative_positions.shape[1]), dtype=np.float64)
    else:
        frame_jerk = np.linalg.norm(np.diff(relative_positions, n=2, axis=0), axis=2)

    heading_delta_deg = compute_heading_delta_degrees(root_heading_6d)
    return {
        "p95_joint_step": float(np.quantile(frame_delta, 0.95)) if frame_delta.size else 0.0,
        "max_joint_step": float(frame_delta.max()) if frame_delta.size else 0.0,
        "p95_joint_jerk": float(np.quantile(frame_jerk, 0.95)) if frame_jerk.size else 0.0,
        "max_joint_jerk": float(frame_jerk.max()) if frame_jerk.size else 0.0,
        "p95_root_heading_delta_deg": float(np.quantile(heading_delta_deg, 0.95)) if heading_delta_deg.size else 0.0,
        "max_root_heading_delta_deg": float(heading_delta_deg.max()) if heading_delta_deg.size else 0.0,
    }


def build_selected_smal_mesh(
    *,
    body_model: str,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    positions: np.ndarray,
    body_points_by_name: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if body_model != "smal":
        return None, None
    if smal_model.suffix.lower() == ".pkl":
        resolved_model_path = resolve_official_smal_model_path(smal_model)
        preset_payload = load_pickle_with_fake_chumpy(smal_model)
        smal_preset = load_smal_preset(smal_model) if is_smal_preset_payload(preset_payload) else None
        official_smal_model = load_official_smal_model(
            model_path=resolved_model_path,
            data_path=resolve_smal_data_path(resolved_model_path, smal_data),
            family_index=smal_family_index,
            betas_override=None if smal_preset is None else smal_preset.beta,
        )
        selected_smal_vertices, _ = build_official_smal_mesh_sequence(
            smal_model=official_smal_model,
            positions=positions,
            body_points_by_name=body_points_by_name,
            head_rotation_matrices=None,
        )
        return selected_smal_vertices, official_smal_model.faces

    template_mesh = load_template_mesh_model(smal_model)
    mapping = load_smal_mapping(smal_mapping)
    bound_template_mesh = bind_template_mesh_model(
        template_mesh=template_mesh,
        mapping=mapping,
    )
    return (
        build_template_mesh_sequence(
            template_mesh=bound_template_mesh,
            positions=positions,
            body_points_by_name=body_points_by_name,
        ),
        bound_template_mesh.faces,
    )


def render_raw_vs_pseudo_animation(
    *,
    output_path: Path,
    frame_indices: np.ndarray,
    packet_counter: np.ndarray,
    raw_positions: np.ndarray,
    pseudo_positions: np.ndarray,
    raw_body_points_by_name: dict[str, np.ndarray],
    pseudo_body_points_by_name: dict[str, np.ndarray],
    raw_back_line: np.ndarray,
    pseudo_back_line: np.ndarray,
    raw_belly_line: np.ndarray,
    pseudo_belly_line: np.ndarray,
    raw_root_translation: np.ndarray,
    pseudo_root_translation: np.ndarray,
    raw_rotation_matrices: np.ndarray,
    pseudo_rotation_matrices: np.ndarray,
    frame_mpjpe_m: np.ndarray,
    frame_heading_error_deg: np.ndarray,
    fps: int,
    point_size: float,
    body_model: str,
    show_skeleton_overlay: bool,
    raw_smal_vertices: np.ndarray | None = None,
    pseudo_smal_vertices: np.ndarray | None = None,
    smal_faces: np.ndarray | None = None,
    left_title: str = "Raw IMU reconstructed pose",
    right_title: str = "Exported pseudo-pose",
    progress_label: str = "Rendering raw-vs-pseudo comparison",
    transition_frame: int | None = None,
    pre_transition_label: str = "",
    post_transition_label: str = "",
    transition_banner: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(top=0.86, bottom=0.12)
    axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
    titles = [left_title, right_title]
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    local_torso_faces = build_local_torso_faces() if body_model == "shell" else []
    overlay_enabled = body_model == "shell" or show_skeleton_overlay
    overlay_alpha = 0.95 if body_model == "shell" else 0.18

    side_state = []
    for axis, title, positions, smal_vertices in zip(
        axes,
        titles,
        [raw_positions, pseudo_positions],
        [raw_smal_vertices, pseudo_smal_vertices],
    ):
        state: dict[str, object] = {"axis": axis}
        if overlay_enabled:
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
            set_equal_axes(axis, np.concatenate([raw_positions, pseudo_positions], axis=0))
        else:
            if smal_vertices is None or smal_faces is None:
                raise ValueError("SMAL comparison rendering requires raw/pseudo vertices and faces")
            state["smal_collection"] = Poly3DCollection([], facecolors="#9c7a56", edgecolors="none", alpha=0.62)
            axis.add_collection3d(state["smal_collection"])
            set_equal_axes(axis, np.concatenate([raw_smal_vertices, pseudo_smal_vertices], axis=0))

        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_title("")
        axis.view_init(elev=18, azim=-60)
        axis.grid(True, alpha=0.35)
        side_state.append(state)

    panel_labels = [
        fig.text(0.25, 0.035, left_title, ha="center", va="bottom", fontsize=14, color="#111827", fontweight="bold"),
        fig.text(0.75, 0.035, right_title, ha="center", va="bottom", fontsize=14, color="#111827", fontweight="bold"),
    ]
    for label in panel_labels:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4})

    phase_text = fig.text(
        0.5,
        0.985,
        "",
        ha="center",
        va="top",
        fontsize=18,
        color="#b91c1c",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 6},
    )
    metric_text = fig.text(
        0.5,
        0.92,
        "",
        ha="center",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 5},
    )

    side_arrays = [
        (
            raw_positions,
            raw_body_points_by_name,
            raw_back_line,
            raw_belly_line,
            raw_root_translation,
            raw_rotation_matrices,
            raw_smal_vertices,
        ),
        (
            pseudo_positions,
            pseudo_body_points_by_name,
            pseudo_back_line,
            pseudo_belly_line,
            pseudo_root_translation,
            pseudo_rotation_matrices,
            pseudo_smal_vertices,
        ),
    ]

    def update(frame_id: int):
        artists = [metric_text, phase_text]
        for state, arrays in zip(side_state, side_arrays):
            (
                positions,
                body_points_by_name,
                back_line,
                belly_line,
                root_translation,
                rotation_matrices,
                smal_vertices,
            ) = arrays
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
                state["smal_collection"].set_verts(smal_vertices[frame_id][smal_faces])
                artists.append(state["smal_collection"])

            if overlay_enabled:
                state["scatter"]._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
                artists.append(state["scatter"])

        metric_lines = [
            "frame={0} packet={1} | MPJPE={2:.3f} mm | heading_error={3:.3f} deg".format(
                int(frame_indices[frame_id]),
                int(packet_counter[frame_id]),
                float(frame_mpjpe_m[frame_id] * 1000.0),
                float(frame_heading_error_deg[frame_id]),
            )
        ]
        phase_lines: list[str] = []
        if transition_frame is not None:
            if frame_id < int(transition_frame):
                if pre_transition_label:
                    phase_lines.append(pre_transition_label)
            else:
                if post_transition_label:
                    phase_lines.append(post_transition_label)
            if frame_id == int(transition_frame) and transition_banner:
                phase_lines.append(transition_banner)
        metric_text.set_text("\n".join(metric_lines))
        phase_text.set_text("\n".join(phase_lines))
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / max(fps, 1),
        blit=False,
    )
    progress = ProgressPrinter(progress_label, len(frame_indices))
    progress_callback = lambda current_frame, total_frames: progress.update(current_frame + 1)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), progress_callback=progress_callback)
    elif suffix == ".mp4":
        animation.save(output_path, writer=build_hardware_ffmpeg_writer(fps=fps), progress_callback=progress_callback)
    else:
        raise ValueError("Comparison video output must end with .mp4 or .gif")
    plt.close(fig)


def render_overlay_pose_comparison_animation(
    *,
    output_path: Path,
    frame_indices: np.ndarray,
    packet_counter: np.ndarray,
    raw_positions: np.ndarray,
    pseudo_positions: np.ndarray,
    raw_body_points_by_name: dict[str, np.ndarray],
    pseudo_body_points_by_name: dict[str, np.ndarray],
    raw_back_line: np.ndarray,
    pseudo_back_line: np.ndarray,
    raw_belly_line: np.ndarray,
    pseudo_belly_line: np.ndarray,
    raw_root_translation: np.ndarray,
    pseudo_root_translation: np.ndarray,
    raw_rotation_matrices: np.ndarray,
    pseudo_rotation_matrices: np.ndarray,
    frame_mpjpe_m: np.ndarray,
    frame_heading_error_deg: np.ndarray,
    fps: int,
    point_size: float,
    body_model: str,
    show_skeleton_overlay: bool,
    raw_smal_vertices: np.ndarray | None = None,
    pseudo_smal_vertices: np.ndarray | None = None,
    smal_faces: np.ndarray | None = None,
    left_title: str = "Ground truth",
    right_title: str = "Reconstruction",
    progress_label: str = "Rendering overlay comparison",
    transition_frame: int | None = None,
    pre_transition_label: str = "",
    post_transition_label: str = "",
    transition_banner: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    fig.subplots_adjust(top=0.86, bottom=0.12)
    axis = fig.add_subplot(111, projection="3d")
    joint_index = {name: idx for idx, name in enumerate(JOINT_ORDER)}
    local_torso_faces = build_local_torso_faces() if body_model == "shell" else []
    overlay_enabled = body_model == "shell" or show_skeleton_overlay

    left_palette = {
        "line": "#1d4ed8",
        "dorsal": "#1e40af",
        "ventral": "#3b82f6",
        "scatter": "#60a5fa",
        "torso": "#2563eb",
        "limb": "#60a5fa",
        "head": "#1d4ed8",
        "ear": "#1e3a8a",
        "mesh": "#2563eb",
    }
    right_palette = {
        "line": "#c2410c",
        "dorsal": "#9a3412",
        "ventral": "#ea580c",
        "scatter": "#fb923c",
        "torso": "#ea580c",
        "limb": "#fb923c",
        "head": "#c2410c",
        "ear": "#9a3412",
        "mesh": "#ea580c",
    }

    def build_side_state(
        *,
        palette: dict[str, str],
        positions: np.ndarray,
        smal_vertices: np.ndarray | None,
        alpha_scale: float,
    ) -> dict[str, object]:
        state: dict[str, object] = {}
        if overlay_enabled:
            state["scatter"] = axis.scatter(
                [],
                [],
                [],
                s=point_size * (0.40 if body_model == "shell" else 0.24),
                c=palette["scatter"],
                depthshade=True,
                alpha=0.55 * alpha_scale,
            )

        if body_model == "shell":
            state["torso_collection"] = Poly3DCollection(
                [], facecolors=palette["torso"], edgecolors="none", alpha=0.22 * alpha_scale
            )
            state["limb_collection"] = Poly3DCollection(
                [], facecolors=palette["limb"], edgecolors="none", alpha=0.18 * alpha_scale
            )
            state["head_collection"] = Poly3DCollection(
                [], facecolors=palette["head"], edgecolors="none", alpha=0.26 * alpha_scale
            )
            state["ear_collection"] = Poly3DCollection(
                [], facecolors=palette["ear"], edgecolors="none", alpha=0.34 * alpha_scale
            )
            for key in ("torso_collection", "limb_collection", "head_collection", "ear_collection"):
                axis.add_collection3d(state[key])
            set_equal_axes(axis, np.concatenate([raw_positions, pseudo_positions], axis=0))
        else:
            if smal_vertices is None or smal_faces is None:
                raise ValueError("SMAL overlay rendering requires raw/pseudo vertices and faces")
            state["smal_collection"] = Poly3DCollection(
                [],
                facecolors=palette["mesh"],
                edgecolors="none",
                alpha=0.28 * alpha_scale,
            )
            axis.add_collection3d(state["smal_collection"])
            set_equal_axes(axis, np.concatenate([raw_smal_vertices, pseudo_smal_vertices], axis=0))
        return state

    left_state = build_side_state(
        palette=left_palette,
        positions=raw_positions,
        smal_vertices=raw_smal_vertices,
        alpha_scale=1.0,
    )
    right_state = build_side_state(
        palette=right_palette,
        positions=pseudo_positions,
        smal_vertices=pseudo_smal_vertices,
        alpha_scale=1.0,
    )

    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.set_title("")
    axis.view_init(elev=18, azim=-60)
    axis.grid(True, alpha=0.35)

    left_label = fig.text(
        0.38,
        0.035,
        left_title,
        ha="center",
        va="bottom",
        fontsize=14,
        color=left_palette["mesh"],
        fontweight="bold",
    )
    right_label = fig.text(
        0.62,
        0.035,
        right_title,
        ha="center",
        va="bottom",
        fontsize=14,
        color=right_palette["mesh"],
        fontweight="bold",
    )
    for label in (left_label, right_label):
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 4})

    phase_text = fig.text(
        0.5,
        0.985,
        "",
        ha="center",
        va="top",
        fontsize=18,
        color="#b91c1c",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 6},
    )
    metric_text = fig.text(
        0.5,
        0.92,
        "",
        ha="center",
        va="top",
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 5},
    )

    side_specs = [
        (
            left_state,
            raw_positions,
            raw_body_points_by_name,
            raw_back_line,
            raw_belly_line,
            raw_root_translation,
            raw_rotation_matrices,
            raw_smal_vertices,
        ),
        (
            right_state,
            pseudo_positions,
            pseudo_body_points_by_name,
            pseudo_back_line,
            pseudo_belly_line,
            pseudo_root_translation,
            pseudo_rotation_matrices,
            pseudo_smal_vertices,
        ),
    ]

    def update(frame_id: int):
        artists = [metric_text, phase_text]
        for state, positions, body_points_by_name, back_line, belly_line, root_translation, rotation_matrices, smal_vertices in side_specs:
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
                state["smal_collection"].set_verts(smal_vertices[frame_id][smal_faces])
                artists.append(state["smal_collection"])

            if overlay_enabled:
                state["scatter"]._offsets3d = (current[:, 0], current[:, 1], current[:, 2])
                artists.append(state["scatter"])

        metric_text.set_text(
            "frame={0} packet={1} | MPJPE={2:.3f} mm | heading_error={3:.3f} deg".format(
                int(frame_indices[frame_id]),
                int(packet_counter[frame_id]),
                float(frame_mpjpe_m[frame_id] * 1000.0),
                float(frame_heading_error_deg[frame_id]),
            )
        )
        phase_lines: list[str] = []
        if transition_frame is not None:
            if frame_id < int(transition_frame):
                if pre_transition_label:
                    phase_lines.append(pre_transition_label)
            else:
                if post_transition_label:
                    phase_lines.append(post_transition_label)
            if frame_id == int(transition_frame) and transition_banner:
                phase_lines.append(transition_banner)
        phase_text.set_text("\n".join(phase_lines))
        return artists

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / max(fps, 1),
        blit=False,
    )
    progress = ProgressPrinter(progress_label, len(frame_indices))
    progress_callback = lambda current_frame, total_frames: progress.update(current_frame + 1)
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), progress_callback=progress_callback)
    elif suffix == ".mp4":
        animation.save(output_path, writer=build_hardware_ffmpeg_writer(fps=fps), progress_callback=progress_callback)
    else:
        raise ValueError("Comparison video output must end with .mp4 or .gif")
    plt.close(fig)


def visualize_pseudo_pose_file(
    *,
    pose_path: Path,
    video_path: Path | None,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    start_frame: int | None,
    num_frames: int | None,
    stride: int,
    window_selection: str,
    search_step: int,
    fps: int,
    point_size: float,
) -> dict[str, object]:
    payload = load_pseudo_pose(pose_path)
    metrics = compute_motion_metrics(payload)
    positions, root_translation, rotation_matrices, body_points_by_name, back_line, belly_line = build_render_arrays(
        payload=payload,
        render_space=render_space,
    )

    selected_window_motion = ""
    if start_frame is None:
        if num_frames is not None and window_selection == "most-active":
            start_frame, _, selected_window_motion = find_most_active_window(
                positions=positions,
                window_frames=num_frames,
                search_step=search_step,
            )
        else:
            start_frame = 0

    frame_indices, selected_packet_counter, selected_positions = select_frames(
        packet_counter=payload.packet_counter,
        positions=positions,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_root_translation = select_frames(
        packet_counter=payload.packet_counter,
        positions=root_translation[:, None, :],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_root_translation = selected_root_translation[:, 0, :]
    _, _, selected_rotation_matrices = select_frames(
        packet_counter=payload.packet_counter,
        positions=rotation_matrices,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_relative_positions = select_frames(
        packet_counter=payload.packet_counter,
        positions=payload.relative_positions,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_root_heading_6d = select_frames(
        packet_counter=payload.packet_counter,
        positions=payload.root_heading_6d[:, None, :],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_root_heading_6d = selected_root_heading_6d[:, 0, :]
    _, _, selected_is_interpolated = select_frames(
        packet_counter=payload.packet_counter,
        positions=payload.is_interpolated.astype(np.float64)[:, :, None],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_is_interpolated = selected_is_interpolated[:, :, 0] > 0.5
    selected_body_points_by_name = {
        name: point_series[frame_indices]
        for name, point_series in body_points_by_name.items()
    }
    selected_back_line = back_line[frame_indices]
    selected_belly_line = belly_line[frame_indices]
    participant, segment_id = infer_pose_metadata(pose_path)
    frame_action_labels = np.asarray(
        [
            (
                f"Pseudo-pose | {participant} segment {segment_id} | "
                f"status={metrics['preview_status']} | space={render_space}"
            )
        ]
        * len(frame_indices),
        dtype=object,
    )
    selected_metrics = compute_pose_audit(
        relative_positions=selected_relative_positions,
        is_interpolated=selected_is_interpolated,
    )
    selected_temporal_diagnostics = compute_temporal_diagnostics(
        relative_positions=selected_relative_positions,
        root_heading_6d=selected_root_heading_6d,
    )

    selected_smal_vertices = None
    smal_faces = None
    if video_path is not None and body_model == "smal":
        if smal_model.suffix.lower() == ".pkl":
            resolved_model_path = resolve_official_smal_model_path(smal_model)
            preset_payload = load_pickle_with_fake_chumpy(smal_model)
            smal_preset = load_smal_preset(smal_model) if is_smal_preset_payload(preset_payload) else None
            official_smal_model = load_official_smal_model(
                model_path=resolved_model_path,
                data_path=resolve_smal_data_path(resolved_model_path, smal_data),
                family_index=smal_family_index,
                betas_override=None if smal_preset is None else smal_preset.beta,
            )
            selected_smal_vertices, _ = build_official_smal_mesh_sequence(
                smal_model=official_smal_model,
                positions=selected_positions,
                body_points_by_name=selected_body_points_by_name,
                head_rotation_matrices=None,
            )
            smal_faces = official_smal_model.faces
        else:
            template_mesh = load_template_mesh_model(smal_model)
            mapping = load_smal_mapping(smal_mapping)
            bound_template_mesh = bind_template_mesh_model(
                template_mesh=template_mesh,
                mapping=mapping,
            )
            selected_smal_vertices = build_template_mesh_sequence(
                template_mesh=bound_template_mesh,
                positions=selected_positions,
                body_points_by_name=selected_body_points_by_name,
            )
            smal_faces = bound_template_mesh.faces

    if video_path is not None:
        render_animation(
            output_path=video_path,
            frame_indices=frame_indices,
            packet_counter=selected_packet_counter,
            positions=selected_positions,
            body_points_by_name=selected_body_points_by_name,
            back_line=selected_back_line,
            belly_line=selected_belly_line,
            root_translation=selected_root_translation,
            rotation_matrices=selected_rotation_matrices,
            fps=fps,
            point_size=point_size,
            body_model=body_model,
            show_skeleton_overlay=show_skeleton_overlay,
            frame_action_labels=frame_action_labels,
            smal_vertices=selected_smal_vertices,
            smal_faces=smal_faces,
        )

    return {
        "participant": participant,
        "segment_id": segment_id,
        "pose_path": str(pose_path),
        "video_path": "" if video_path is None else str(video_path),
        "render_space": render_space,
        "num_frames_total": int(payload.packet_counter.shape[0]),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "packet_start": int(selected_packet_counter[0]),
        "packet_end": int(selected_packet_counter[-1]),
        **metrics,
        "full_preview_status": str(metrics["preview_status"]),
        "selected_finite_ratio": float(selected_metrics["finite_ratio"]),
        "selected_interp_ratio": float(selected_metrics["interp_ratio"]),
        "selected_max_joint_interp_ratio": float(selected_metrics["max_joint_interp_ratio"]),
        "selected_worst_interp_joint": str(selected_metrics["worst_interp_joint"]),
        "selected_preview_status": str(selected_metrics["preview_status"]),
        "preview_status": str(selected_metrics["preview_status"]),
        "selected_window_motion": "" if selected_window_motion == "" else float(selected_window_motion),
        "selected_p95_joint_step": float(selected_temporal_diagnostics["p95_joint_step"]),
        "selected_max_joint_step": float(selected_temporal_diagnostics["max_joint_step"]),
        "selected_p95_joint_jerk": float(selected_temporal_diagnostics["p95_joint_jerk"]),
        "selected_max_joint_jerk": float(selected_temporal_diagnostics["max_joint_jerk"]),
        "selected_p95_root_heading_delta_deg": float(selected_temporal_diagnostics["p95_root_heading_delta_deg"]),
        "selected_max_root_heading_delta_deg": float(selected_temporal_diagnostics["max_root_heading_delta_deg"]),
        "compare_mean_mpjpe_mm": "",
        "compare_max_mpjpe_mm": "",
        "compare_mean_heading_error_deg": "",
        "compare_max_heading_error_deg": "",
    }


def visualize_raw_vs_pseudo_file(
    *,
    pose_path: Path,
    filled_root: Path,
    video_path: Path,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
    start_frame: int | None,
    num_frames: int | None,
    stride: int,
    window_selection: str,
    search_step: int,
    fps: int,
    point_size: float,
) -> dict[str, object]:
    pseudo_payload = load_pseudo_pose(pose_path)
    raw_payload = load_raw_imu_reference_pose(
        pose_path=pose_path,
        filled_root=filled_root,
        target_packet_counter=pseudo_payload.packet_counter,
        neutral_pose_mode=neutral_pose_mode,
        gyr_motion_scale=gyr_motion_scale,
    )
    pseudo_metrics = compute_motion_metrics(pseudo_payload)
    pseudo_positions, pseudo_root_translation, pseudo_rotation_matrices, pseudo_body_points, pseudo_back_line, pseudo_belly_line = (
        build_render_arrays(payload=pseudo_payload, render_space=render_space)
    )
    raw_positions, raw_root_translation, raw_rotation_matrices, raw_body_points, raw_back_line, raw_belly_line = (
        build_render_arrays(payload=raw_payload, render_space=render_space)
    )

    selected_window_motion = ""
    if start_frame is None:
        if num_frames is not None and window_selection == "most-active":
            start_frame, _, selected_window_motion = find_most_active_window(
                positions=pseudo_positions,
                window_frames=num_frames,
                search_step=search_step,
            )
        else:
            start_frame = 0

    frame_indices, selected_packet_counter, selected_pseudo_positions = select_frames(
        packet_counter=pseudo_payload.packet_counter,
        positions=pseudo_positions,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_raw_positions = select_frames(
        packet_counter=raw_payload.packet_counter,
        positions=raw_positions,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_pseudo_relative_positions = select_frames(
        packet_counter=pseudo_payload.packet_counter,
        positions=pseudo_payload.relative_positions,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_pseudo_is_interpolated = select_frames(
        packet_counter=pseudo_payload.packet_counter,
        positions=pseudo_payload.is_interpolated.astype(np.float64)[:, :, None],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_pseudo_is_interpolated = selected_pseudo_is_interpolated[:, :, 0] > 0.5
    _, _, selected_pseudo_root_translation = select_frames(
        packet_counter=pseudo_payload.packet_counter,
        positions=pseudo_root_translation[:, None, :],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_pseudo_root_translation = selected_pseudo_root_translation[:, 0, :]
    _, _, selected_raw_root_translation = select_frames(
        packet_counter=raw_payload.packet_counter,
        positions=raw_root_translation[:, None, :],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_raw_root_translation = selected_raw_root_translation[:, 0, :]
    _, _, selected_pseudo_rotation_matrices = select_frames(
        packet_counter=pseudo_payload.packet_counter,
        positions=pseudo_rotation_matrices,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_raw_rotation_matrices = select_frames(
        packet_counter=raw_payload.packet_counter,
        positions=raw_rotation_matrices,
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    _, _, selected_pseudo_heading = select_frames(
        packet_counter=pseudo_payload.packet_counter,
        positions=pseudo_payload.root_heading_6d[:, None, :],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_pseudo_heading = selected_pseudo_heading[:, 0, :]
    _, _, selected_raw_heading = select_frames(
        packet_counter=raw_payload.packet_counter,
        positions=raw_payload.root_heading_6d[:, None, :],
        start_frame=start_frame,
        num_frames=num_frames,
        stride=stride,
    )
    selected_raw_heading = selected_raw_heading[:, 0, :]

    selected_pseudo_body_points = {
        name: point_series[frame_indices]
        for name, point_series in pseudo_body_points.items()
    }
    selected_raw_body_points = {
        name: point_series[frame_indices]
        for name, point_series in raw_body_points.items()
    }
    selected_pseudo_back_line = pseudo_back_line[frame_indices]
    selected_raw_back_line = raw_back_line[frame_indices]
    selected_pseudo_belly_line = pseudo_belly_line[frame_indices]
    selected_raw_belly_line = raw_belly_line[frame_indices]

    selected_metrics = compute_pose_audit(
        relative_positions=selected_pseudo_relative_positions,
        is_interpolated=selected_pseudo_is_interpolated,
    )
    selected_temporal_diagnostics = compute_temporal_diagnostics(
        relative_positions=selected_pseudo_relative_positions,
        root_heading_6d=selected_pseudo_heading,
    )
    comparison = compute_frame_comparison_metrics(
        raw_positions=selected_raw_positions,
        pseudo_positions=selected_pseudo_positions,
        raw_root_heading_6d=selected_raw_heading,
        pseudo_root_heading_6d=selected_pseudo_heading,
    )

    raw_smal_vertices, smal_faces = build_selected_smal_mesh(
        body_model=body_model,
        smal_model=smal_model,
        smal_mapping=smal_mapping,
        smal_data=smal_data,
        smal_family_index=smal_family_index,
        positions=selected_raw_positions,
        body_points_by_name=selected_raw_body_points,
    )
    pseudo_smal_vertices, pseudo_smal_faces = build_selected_smal_mesh(
        body_model=body_model,
        smal_model=smal_model,
        smal_mapping=smal_mapping,
        smal_data=smal_data,
        smal_family_index=smal_family_index,
        positions=selected_pseudo_positions,
        body_points_by_name=selected_pseudo_body_points,
    )
    if smal_faces is None:
        smal_faces = pseudo_smal_faces

    render_raw_vs_pseudo_animation(
        output_path=video_path,
        frame_indices=frame_indices,
        packet_counter=selected_packet_counter,
        raw_positions=selected_raw_positions,
        pseudo_positions=selected_pseudo_positions,
        raw_body_points_by_name=selected_raw_body_points,
        pseudo_body_points_by_name=selected_pseudo_body_points,
        raw_back_line=selected_raw_back_line,
        pseudo_back_line=selected_pseudo_back_line,
        raw_belly_line=selected_raw_belly_line,
        pseudo_belly_line=selected_pseudo_belly_line,
        raw_root_translation=selected_raw_root_translation,
        pseudo_root_translation=selected_pseudo_root_translation,
        raw_rotation_matrices=selected_raw_rotation_matrices,
        pseudo_rotation_matrices=selected_pseudo_rotation_matrices,
        frame_mpjpe_m=comparison["frame_mpjpe_m"],
        frame_heading_error_deg=comparison["frame_heading_error_deg"],
        fps=fps,
        point_size=point_size,
        body_model=body_model,
        show_skeleton_overlay=show_skeleton_overlay,
        raw_smal_vertices=raw_smal_vertices,
        pseudo_smal_vertices=pseudo_smal_vertices,
        smal_faces=smal_faces,
    )

    participant, segment_id = infer_pose_metadata(pose_path)
    return {
        "participant": participant,
        "segment_id": segment_id,
        "pose_path": str(pose_path),
        "video_path": str(video_path),
        "render_space": render_space,
        "num_frames_total": int(pseudo_payload.packet_counter.shape[0]),
        "frame_start": int(frame_indices[0]),
        "frame_end": int(frame_indices[-1]),
        "packet_start": int(selected_packet_counter[0]),
        "packet_end": int(selected_packet_counter[-1]),
        **pseudo_metrics,
        "full_preview_status": str(pseudo_metrics["preview_status"]),
        "selected_finite_ratio": float(selected_metrics["finite_ratio"]),
        "selected_interp_ratio": float(selected_metrics["interp_ratio"]),
        "selected_max_joint_interp_ratio": float(selected_metrics["max_joint_interp_ratio"]),
        "selected_worst_interp_joint": str(selected_metrics["worst_interp_joint"]),
        "selected_preview_status": str(selected_metrics["preview_status"]),
        "preview_status": str(selected_metrics["preview_status"]),
        "selected_window_motion": "" if selected_window_motion == "" else float(selected_window_motion),
        "selected_p95_joint_step": float(selected_temporal_diagnostics["p95_joint_step"]),
        "selected_max_joint_step": float(selected_temporal_diagnostics["max_joint_step"]),
        "selected_p95_joint_jerk": float(selected_temporal_diagnostics["p95_joint_jerk"]),
        "selected_max_joint_jerk": float(selected_temporal_diagnostics["max_joint_jerk"]),
        "selected_p95_root_heading_delta_deg": float(selected_temporal_diagnostics["p95_root_heading_delta_deg"]),
        "selected_max_root_heading_delta_deg": float(selected_temporal_diagnostics["max_root_heading_delta_deg"]),
        "compare_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
        "compare_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
        "compare_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
        "compare_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
    }


def discover_pose_files(pose_root: Path) -> list[Path]:
    return sorted(pose_root.glob("*/segment_*.npz"), key=lambda path: (path.parent.name, path.stem))


def write_audit(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VISUAL_AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "Data" / "IMU_Only_20Hz_v1"
    parser = argparse.ArgumentParser(description="Render pseudo-pose npz files as dog skeleton videos")
    parser.add_argument("--pose-path", type=Path, default=None, help="Single pseudo-pose npz to render")
    parser.add_argument("--pose-root", type=Path, default=data_root / "pseudo_pose_20hz")
    parser.add_argument("--video-output", type=Path, default=None, help="Output video for single-file mode")
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=None,
        help="Render a side-by-side raw IMU vs pseudo-pose comparison video for --pose-path.",
    )
    parser.add_argument("--filled-root", type=Path, default=root / "Data" / "IMU_New2_Filled")
    parser.add_argument("--output-root", type=Path, default=data_root / "visual_audit" / "pseudo_pose_previews")
    parser.add_argument("--audit-output", type=Path, default=data_root / "visual_audit" / "pseudo_pose_visual_audit.csv")
    parser.add_argument("--render-space", choices=["root", "heading"], default="heading")
    parser.add_argument(
        "--body-model",
        choices=["shell", "smal"],
        default="smal",
        help="Body renderer. Default: smal, matching visualize_dog_skeleton.py.",
    )
    parser.add_argument(
        "--show-skeleton-overlay",
        action="store_true",
        help="Render skeleton overlay on top of SMAL mesh. For SMAL this is off by default.",
    )
    parser.add_argument(
        "--smal-model",
        type=Path,
        default=root / "SMAL" / "wolf_alph3.pkl",
        help="SMAL model or preset path. Default: SMAL/wolf_alph3.pkl",
    )
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--neutral-pose-mode", choices=["first-frame", "sequence-median"], default="sequence-median")
    parser.add_argument("--gyr-motion-scale", type=float, default=DEFAULT_GYR_MOTION_SCALE)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=240)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--window-selection", choices=["from-start", "most-active"], default="most-active")
    parser.add_argument("--search-step", type=int, default=20)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.comparison_output is not None:
        if args.pose_path is None:
            raise ValueError("--comparison-output requires --pose-path")
        row = visualize_raw_vs_pseudo_file(
            pose_path=args.pose_path,
            filled_root=args.filled_root,
            video_path=args.comparison_output,
            render_space=args.render_space,
            body_model=args.body_model,
            show_skeleton_overlay=args.show_skeleton_overlay,
            smal_model=args.smal_model,
            smal_mapping=args.smal_mapping,
            smal_data=args.smal_data,
            smal_family_index=args.smal_family_index,
            neutral_pose_mode=args.neutral_pose_mode,
            gyr_motion_scale=args.gyr_motion_scale,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            stride=args.stride,
            window_selection=args.window_selection,
            search_step=args.search_step,
            fps=args.fps,
            point_size=args.point_size,
        )
        write_audit(args.audit_output, [row])
        print(f"Saved raw-vs-pseudo comparison audit to: {args.audit_output}")
        return

    if args.pose_path is not None:
        pose_files = [args.pose_path]
    else:
        pose_files = discover_pose_files(args.pose_root)
        if args.max_videos is not None:
            pose_files = pose_files[: args.max_videos]

    rows: list[dict[str, object]] = []
    total = len(pose_files)
    for index, pose_path in enumerate(pose_files, start=1):
        participant, segment_id = infer_pose_metadata(pose_path)
        video_path = None
        if not args.no_video:
            if args.video_output is not None:
                if total != 1:
                    raise ValueError("--video-output is only valid with --pose-path or a single input")
                video_path = args.video_output
            else:
                video_path = build_export_video_path(
                    output_dir=args.output_root,
                    source_tag="pseudo-pose",
                    participant=participant,
                    segment_id=segment_id,
                    start_frame_20hz=args.start_frame,
                )

        print(f"[{index}/{total}] Rendering pseudo-pose {participant} segment {segment_id}")
        row = visualize_pseudo_pose_file(
            pose_path=pose_path,
            video_path=video_path,
            render_space=args.render_space,
            body_model=args.body_model,
            show_skeleton_overlay=args.show_skeleton_overlay,
            smal_model=args.smal_model,
            smal_mapping=args.smal_mapping,
            smal_data=args.smal_data,
            smal_family_index=args.smal_family_index,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            stride=args.stride,
            window_selection=args.window_selection,
            search_step=args.search_step,
            fps=args.fps,
            point_size=args.point_size,
        )
        rows.append(row)

    write_audit(args.audit_output, rows)
    print(f"Saved pseudo-pose visual audit to: {args.audit_output}")


if __name__ == "__main__":
    main()
