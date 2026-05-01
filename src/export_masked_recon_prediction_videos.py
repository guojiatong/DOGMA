#!/usr/bin/env python3
"""
Export masked reconstruction validation videos with shared input context.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from export_temporal_vae_failure_pack import (
    compare_payloads,
    render_payload_comparison_video,
    resolve_workspace_path,
    save_payload_npz,
    slice_payload,
)
from train_imu_masked_recon import write_json
from video_export_paths import build_export_video_path
from visualize_dog_skeleton import (
    DEFAULT_GYR_MOTION_SCALE,
    JOINT_ORDER,
    align_quaternion_hemisphere,
    build_heading_rotation_matrices,
    build_relative_positions,
    normalize_quaternions,
    rotation_matrices_to_rot6d,
)
from visualize_pose import PseudoPosePayload


SUMMARY_COLUMNS = (
    "rank",
    "epoch",
    "participant",
    "segment_id",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
    "context_frames",
    "masked_token_count_post",
    "masked_token_ratio_post",
    "comparison_mean_mpjpe_mm",
    "comparison_max_mpjpe_mm",
    "comparison_mean_heading_error_deg",
    "comparison_max_heading_error_deg",
    "window_dir",
    "video_path",
)


def read_best_epoch(run_dir: Path) -> int:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    best_epoch: int | None = None
    best_metric = float("inf")
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            metric = float(row["val_masked_rot_geodesic_error"])
            epoch = int(row["epoch"])
            if metric <= best_metric:
                best_metric = metric
                best_epoch = epoch

    if best_epoch is None:
        raise ValueError(f"No rows found in {metrics_path}")
    return best_epoch


def rot6d_to_rotation_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    first = rot6d[..., 0:3]
    second = rot6d[..., 3:6]
    basis_x = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second = second - np.sum(basis_x * second, axis=-1, keepdims=True) * basis_x
    basis_y = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    basis_z = np.cross(basis_x, basis_y)
    return np.stack([basis_x, basis_y, basis_z], axis=-1)


def rotation_matrix_to_quaternion_np(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end with (3, 3), got {rotation.shape}")

    output = np.zeros(rotation.shape[:-2] + (4,), dtype=np.float64)
    flat_rotation = rotation.reshape(-1, 3, 3)
    flat_output = output.reshape(-1, 4)

    for index, matrix in enumerate(flat_rotation):
        trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            flat_output[index, 0] = 0.25 * s
            flat_output[index, 1] = (matrix[2, 1] - matrix[1, 2]) / s
            flat_output[index, 2] = (matrix[0, 2] - matrix[2, 0]) / s
            flat_output[index, 3] = (matrix[1, 0] - matrix[0, 1]) / s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-8)) * 2.0
            flat_output[index, 0] = (matrix[2, 1] - matrix[1, 2]) / s
            flat_output[index, 1] = 0.25 * s
            flat_output[index, 2] = (matrix[0, 1] + matrix[1, 0]) / s
            flat_output[index, 3] = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-8)) * 2.0
            flat_output[index, 0] = (matrix[0, 2] - matrix[2, 0]) / s
            flat_output[index, 1] = (matrix[0, 1] + matrix[1, 0]) / s
            flat_output[index, 2] = 0.25 * s
            flat_output[index, 3] = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-8)) * 2.0
            flat_output[index, 0] = (matrix[1, 0] - matrix[0, 1]) / s
            flat_output[index, 1] = (matrix[0, 2] + matrix[2, 0]) / s
            flat_output[index, 2] = (matrix[1, 2] + matrix[2, 1]) / s
            flat_output[index, 3] = 0.25 * s

    return normalize_quaternions(output)


def align_quaternion_sequence(quat: np.ndarray, reference_quat: np.ndarray | None = None) -> np.ndarray:
    quat = normalize_quaternions(np.asarray(quat, dtype=np.float64))
    if quat.ndim == 2 and quat.shape[1] == 4:
        aligned = quat.copy()
        if aligned.shape[0] == 0:
            return aligned
        if reference_quat is not None:
            aligned[0:1] = align_quaternion_hemisphere(aligned[0:1], np.asarray(reference_quat, dtype=np.float64)[None])
        for frame_index in range(1, aligned.shape[0]):
            aligned[frame_index : frame_index + 1] = align_quaternion_hemisphere(
                aligned[frame_index : frame_index + 1],
                aligned[frame_index - 1 : frame_index],
            )
        return aligned

    if quat.ndim == 3 and quat.shape[2] == 4:
        aligned = quat.copy()
        sensor_count = aligned.shape[1]
        reference = None if reference_quat is None else np.asarray(reference_quat, dtype=np.float64)
        for sensor_index in range(sensor_count):
            sensor_reference = None
            if reference is not None:
                if reference.ndim == 1:
                    sensor_reference = reference
                elif reference.ndim == 2:
                    sensor_reference = reference[sensor_index]
                else:
                    raise ValueError(f"Unsupported reference_quat shape: {reference.shape}")
            aligned[:, sensor_index, :] = align_quaternion_sequence(
                aligned[:, sensor_index, :],
                reference_quat=sensor_reference,
            )
        return aligned

    raise ValueError(f"quat must have shape [T,4] or [T,S,4], got {quat.shape}")


def predicted_rot6d_to_quaternion(
    pred_rot6d: np.ndarray,
    reference_quat: np.ndarray | None = None,
) -> np.ndarray:
    rotation = rot6d_to_rotation_matrix_np(pred_rot6d)
    quat = rotation_matrix_to_quaternion_np(rotation)
    return align_quaternion_sequence(quat, reference_quat=reference_quat)


def build_payload_from_sensor_window(
    *,
    packet_counter: np.ndarray,
    quat: np.ndarray,
    gyr: np.ndarray,
    freeacc: np.ndarray,
    is_interpolated: np.ndarray,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
) -> PseudoPosePayload:
    sensor_data_by_joint: dict[str, dict[str, np.ndarray]] = {}
    for joint_index, joint_name in enumerate(JOINT_ORDER):
        sensor_data_by_joint[joint_name] = {
            "quat": quat[:, joint_index].astype(np.float64),
            "gyr": gyr[:, joint_index].astype(np.float64),
            "freeacc": freeacc[:, joint_index].astype(np.float64),
            "is_interpolated": is_interpolated[:, joint_index].astype(bool),
        }

    relative_positions = build_relative_positions(
        sensor_data_by_joint=sensor_data_by_joint,
        neutral_pose_mode=neutral_pose_mode,
        gyr_motion_scale=gyr_motion_scale,
    ).astype(np.float64)
    root_heading_6d = rotation_matrices_to_rot6d(
        build_heading_rotation_matrices(sensor_data_by_joint["stern"]["quat"])
    ).astype(np.float64)
    return PseudoPosePayload(
        packet_counter=np.asarray(packet_counter, dtype=np.int64),
        relative_positions=relative_positions,
        root_heading_6d=root_heading_6d,
        is_interpolated=is_interpolated.astype(bool),
        joint_names=list(JOINT_ORDER),
    )


def load_feature_window(
    *,
    feature_path: Path,
    start_frame: int,
    valid_frames: int,
) -> dict[str, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as payload:
        end_frame = start_frame + valid_frames
        return {
            "packet_counter": payload["packet_counter"][start_frame:end_frame].astype(np.int64),
            "quat": payload["quat"][start_frame:end_frame].astype(np.float64),
            "gyr": payload["gyr"][start_frame:end_frame].astype(np.float64),
            "freeacc": payload["freeacc"][start_frame:end_frame].astype(np.float64),
            "is_interpolated": payload["is_interpolated"][start_frame:end_frame].astype(bool),
        }


def derive_feature_path(
    *,
    feature_root: Path,
    participant: str,
    segment_id: str,
) -> Path:
    feature_path = feature_root / participant / f"segment_{segment_id}.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file for sample: {feature_path}")
    return feature_path


def build_reconstructed_window(
    *,
    base_window: dict[str, np.ndarray],
    prediction_feature: np.ndarray,
    target_feature: np.ndarray,
    mask: np.ndarray,
    context_frames: int,
    use_blended_prediction: bool,
) -> dict[str, np.ndarray]:
    valid_frames = base_window["packet_counter"].shape[0]
    context_frames = min(int(context_frames), valid_frames)
    predicted_post = prediction_feature.copy()
    if use_blended_prediction:
        predicted_post = np.where(mask[..., None], prediction_feature, target_feature)

    output = {
        "packet_counter": base_window["packet_counter"].copy(),
        "quat": base_window["quat"].copy(),
        "gyr": base_window["gyr"].copy(),
        "freeacc": base_window["freeacc"].copy(),
        "is_interpolated": base_window["is_interpolated"].copy(),
    }
    if context_frames < valid_frames:
        output["quat"][context_frames:] = predicted_rot6d_to_quaternion(
            predicted_post[context_frames:, :, :6],
            reference_quat=base_window["quat"][context_frames - 1] if context_frames > 0 else base_window["quat"][0],
        )
        output["gyr"][context_frames:] = predicted_post[context_frames:, :, 6:9]
        output["freeacc"][context_frames:] = predicted_post[context_frames:, :, 9:12]
    return output


def load_sample_package(sample_path: Path) -> tuple[int, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(sample_path, allow_pickle=True) as payload:
        epoch = int(payload["epoch"])
        meta = json.loads(str(payload["meta_json"].tolist()))
        return (
            epoch,
            payload["input"].astype(np.float32),
            meta,
            payload["target"].astype(np.float32),
            payload["prediction"].astype(np.float32),
            payload["mask"].astype(bool),
        )


def export_masked_recon_prediction_videos(
    *,
    run_dir: Path,
    output_dir: Path | None = None,
    sample_path: Path | None = None,
    feature_root: Path | None = None,
    context_frames: int = 120,
    render_space: str = "heading",
    body_model: str = "smal",
    show_skeleton_overlay: bool = False,
    smal_model: Path | None = None,
    smal_mapping: Path | None = None,
    smal_data: Path | None = None,
    smal_family_index: int = 1,
    fps: int = 20,
    point_size: float = 42.0,
    no_video: bool = False,
    neutral_pose_mode: str = "first-frame",
    gyr_motion_scale: float = DEFAULT_GYR_MOTION_SCALE,
    use_blended_prediction: bool = True,
) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[1]
    run_dir = resolve_workspace_path(Path(run_dir), workspace_root)
    if feature_root is None:
        feature_root = workspace_root / "Data" / "IMU_Only_20Hz_v1" / "features_20hz"
    if smal_model is None:
        smal_model = workspace_root / "SMAL" / "wolf_alph3.pkl"

    if sample_path is None:
        best_epoch = read_best_epoch(run_dir)
        sample_path = run_dir / "sample_predictions" / f"epoch_{best_epoch:04d}.npz"
    else:
        sample_path = resolve_workspace_path(Path(sample_path), workspace_root)
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample prediction file: {sample_path}")

    if output_dir is None:
        output_dir = run_dir / "sample_videos" / sample_path.stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch, _, meta_rows, target_batch, prediction_batch, mask_batch = load_sample_package(sample_path)
    summary_rows: list[dict[str, Any]] = []

    for rank, meta in enumerate(meta_rows, start=1):
        participant = str(meta["participant"])
        segment_id = str(meta["segment_id"])
        start_frame = int(meta["start_frame_20hz"])
        valid_frames = int(meta["valid_frames"])
        feature_path = derive_feature_path(
            feature_root=Path(feature_root),
            participant=participant,
            segment_id=segment_id,
        )
        base_window = load_feature_window(
            feature_path=feature_path,
            start_frame=start_frame,
            valid_frames=valid_frames,
        )
        prediction_feature = prediction_batch[rank - 1, :valid_frames]
        target_feature = target_batch[rank - 1, :valid_frames]
        mask = mask_batch[rank - 1, :valid_frames]
        effective_context_frames = min(int(context_frames), valid_frames)
        reconstructed_window = build_reconstructed_window(
            base_window=base_window,
            prediction_feature=prediction_feature,
            target_feature=target_feature,
            mask=mask,
            context_frames=effective_context_frames,
            use_blended_prediction=use_blended_prediction,
        )
        target_payload = build_payload_from_sensor_window(
            packet_counter=base_window["packet_counter"],
            quat=base_window["quat"],
            gyr=base_window["gyr"],
            freeacc=base_window["freeacc"],
            is_interpolated=base_window["is_interpolated"],
            neutral_pose_mode=neutral_pose_mode,
            gyr_motion_scale=gyr_motion_scale,
        )
        reconstruction_payload = build_payload_from_sensor_window(
            packet_counter=reconstructed_window["packet_counter"],
            quat=reconstructed_window["quat"],
            gyr=reconstructed_window["gyr"],
            freeacc=reconstructed_window["freeacc"],
            is_interpolated=reconstructed_window["is_interpolated"],
            neutral_pose_mode=neutral_pose_mode,
            gyr_motion_scale=gyr_motion_scale,
        )
        comparison = compare_payloads(
            left_payload=slice_payload(
                target_payload,
                start_frame=effective_context_frames,
                num_frames=max(valid_frames - effective_context_frames, 1),
                rebase_heading=False,
            ),
            right_payload=slice_payload(
                reconstruction_payload,
                start_frame=effective_context_frames,
                num_frames=max(valid_frames - effective_context_frames, 1),
                rebase_heading=False,
            ),
            render_space=render_space,
        )

        window_dir = output_dir / f"{rank:02d}__{participant}__segment_{segment_id}__start_{start_frame}"
        window_dir.mkdir(parents=True, exist_ok=True)
        save_payload_npz(window_dir / "ground_truth_window.npz", target_payload)
        save_payload_npz(window_dir / "reconstruction_window.npz", reconstruction_payload)

        video_path = ""
        if not no_video:
            video_output_path = build_export_video_path(
                output_dir=output_dir,
                source_tag="masked-recon",
                participant=participant,
                segment_id=segment_id,
                start_frame_20hz=start_frame,
                rank=rank,
                extra_tags=(f"epoch-{epoch:04d}",),
            )
            render_payload_comparison_video(
                left_payload=target_payload,
                right_payload=reconstruction_payload,
                output_path=video_output_path,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=Path(smal_model),
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Ground truth",
                right_title="Masked reconstruction",
                progress_label=f"{participant} seg{segment_id} start{start_frame}",
                fps=fps,
                point_size=point_size,
                transition_frame=effective_context_frames,
                pre_transition_label=f"INPUT CONTEXT {effective_context_frames / 20.0:.1f}S ON BOTH SIDES",
                post_transition_label="LEFT GROUND TRUTH | RIGHT RECONSTRUCTION",
                transition_banner="RECONSTRUCTION VIEW",
            )
            video_path = str(video_output_path)

        post_mask = mask[effective_context_frames:]
        summary_rows.append(
            {
                "rank": int(rank),
                "epoch": int(epoch),
                "participant": participant,
                "segment_id": segment_id,
                "feature_path": str(feature_path),
                "start_frame_20hz": int(start_frame),
                "valid_frames": int(valid_frames),
                "packet_start_20hz": int(meta["packet_start_20hz"]),
                "packet_end_20hz": int(meta["packet_end_20hz"]),
                "context_frames": int(effective_context_frames),
                "masked_token_count_post": int(post_mask.sum()),
                "masked_token_ratio_post": float(post_mask.mean()) if post_mask.size > 0 else 0.0,
                "comparison_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                "comparison_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                "comparison_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                "comparison_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
                "window_dir": str(window_dir),
                "video_path": video_path,
            }
        )

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "run_dir": str(run_dir),
        "sample_path": str(sample_path),
        "epoch": int(epoch),
        "top_k": len(summary_rows),
        "rows": summary_rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export masked reconstruction comparison videos")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument("--feature-root", type=Path, default=root / "Data" / "IMU_Only_20Hz_v1" / "features_20hz")
    parser.add_argument("--context-frames", type=int, default=120)
    parser.add_argument("--render-space", choices=("root", "heading"), default="heading")
    parser.add_argument("--body-model", choices=("shell", "smal"), default="smal")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=root / "SMAL" / "wolf_alph3.pkl")
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--neutral-pose-mode", choices=("first-frame", "sequence-median"), default="first-frame")
    parser.add_argument("--gyr-motion-scale", type=float, default=DEFAULT_GYR_MOTION_SCALE)
    parser.add_argument("--use-raw-prediction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_masked_recon_prediction_videos(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        sample_path=args.sample_path,
        feature_root=args.feature_root,
        context_frames=args.context_frames,
        render_space=args.render_space,
        body_model=args.body_model,
        show_skeleton_overlay=args.show_skeleton_overlay,
        smal_model=args.smal_model,
        smal_mapping=args.smal_mapping,
        smal_data=args.smal_data,
        smal_family_index=args.smal_family_index,
        fps=args.fps,
        point_size=args.point_size,
        no_video=args.no_video,
        neutral_pose_mode=args.neutral_pose_mode,
        gyr_motion_scale=args.gyr_motion_scale,
        use_blended_prediction=not args.use_raw_prediction,
    )
    print(summary)


if __name__ == "__main__":
    main()
