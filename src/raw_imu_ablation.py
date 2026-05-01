#!/usr/bin/env python3
"""
Shared helpers for the raw-IMU latent ablation study.

This ablation keeps the current IMU-only data source, but swaps the latent
target from pseudo-pose windows to raw 20Hz IMU windows. Visualization still
projects decoded IMU predictions back into pseudo-pose space for inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from imu_new2_common import JOINT_ORDER


RAW_IMU_SENSOR_DIM = 13
RAW_IMU_ROT6D_DIM = 6
RAW_IMU_GYR_DIM = 3
RAW_IMU_FREEACC_DIM = 3
RAW_IMU_INTERP_DIM = 1
RAW_IMU_SENSOR_COUNT = len(JOINT_ORDER)
RAW_IMU_INPUT_DIM = RAW_IMU_SENSOR_COUNT * RAW_IMU_SENSOR_DIM
DEFAULT_GYR_MOTION_SCALE = 0.8


def normalize_quaternions(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return quat / norm


def align_quaternion_hemisphere(
    quat: np.ndarray,
    reference_quat: np.ndarray | None = None,
) -> np.ndarray:
    aligned = quat.copy()
    if reference_quat is None:
        reference_quat = aligned[0:1]
    dot = np.sum(aligned * np.broadcast_to(reference_quat, aligned.shape), axis=1, keepdims=True)
    aligned = np.where(dot < 0.0, -aligned, aligned)
    return aligned


def flatten_feature_window(feature_window: np.ndarray) -> np.ndarray:
    feature_window = np.asarray(feature_window, dtype=np.float32)
    if feature_window.ndim != 3 or feature_window.shape[1:] != (RAW_IMU_SENSOR_COUNT, RAW_IMU_SENSOR_DIM):
        raise ValueError(
            "feature_window must have shape [T, sensor_count, 13], "
            f"got {feature_window.shape}"
        )
    return feature_window.reshape(feature_window.shape[0], RAW_IMU_INPUT_DIM).astype(np.float32)


def unflatten_feature_window(flat_feature_window: np.ndarray) -> np.ndarray:
    flat_feature_window = np.asarray(flat_feature_window, dtype=np.float32)
    if flat_feature_window.ndim != 2 or flat_feature_window.shape[1] != RAW_IMU_INPUT_DIM:
        raise ValueError(
            f"flat_feature_window must have shape [T,{RAW_IMU_INPUT_DIM}], got {flat_feature_window.shape}"
        )
    return flat_feature_window.reshape(flat_feature_window.shape[0], RAW_IMU_SENSOR_COUNT, RAW_IMU_SENSOR_DIM)


def normalize_feature_tensor(feature: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (feature - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_feature_tensor(feature: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return feature * std.view(1, 1, -1) + mean.view(1, 1, -1)


def load_feature_tensor_from_npz(feature_path: Path) -> dict[str, np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as payload:
        feature = payload["feature"].astype(np.float32)
        packet_counter = payload["packet_counter"].astype(np.int64)
        sensor_ids = payload["sensor_ids"] if "sensor_ids" in payload.files else np.asarray([], dtype=str)
        joint_names = payload["joint_names"] if "joint_names" in payload.files else np.asarray(JOINT_ORDER)
    return {
        "feature": feature,
        "packet_counter": packet_counter,
        "sensor_ids": sensor_ids,
        "joint_names": joint_names,
    }


def slice_flat_feature_window_from_path(
    *,
    feature_path: Path,
    start_frame: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = load_feature_tensor_from_npz(feature_path)
    end_frame = int(start_frame + num_frames)
    return (
        flatten_feature_window(payload["feature"][start_frame:end_frame]),
        payload["packet_counter"][start_frame:end_frame].astype(np.int64),
    )


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
        raise ValueError(f"rotation must end with (3,3), got {rotation.shape}")

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


def flat_feature_window_to_pseudo_pose_payload(
    *,
    flat_feature_window: np.ndarray,
    packet_counter: np.ndarray,
    neutral_pose_mode: str = "sequence-median",
    gyr_motion_scale: float = DEFAULT_GYR_MOTION_SCALE,
    reference_quat: np.ndarray | None = None,
):
    from export_masked_recon_prediction_videos import build_payload_from_sensor_window

    feature_window = unflatten_feature_window(flat_feature_window).astype(np.float64)
    quat = predicted_rot6d_to_quaternion(feature_window[:, :, :RAW_IMU_ROT6D_DIM], reference_quat=reference_quat)
    return build_payload_from_sensor_window(
        packet_counter=np.asarray(packet_counter, dtype=np.int64),
        quat=quat,
        gyr=feature_window[:, :, RAW_IMU_ROT6D_DIM : RAW_IMU_ROT6D_DIM + RAW_IMU_GYR_DIM],
        freeacc=feature_window[
            :,
            :,
            RAW_IMU_ROT6D_DIM + RAW_IMU_GYR_DIM : RAW_IMU_ROT6D_DIM + RAW_IMU_GYR_DIM + RAW_IMU_FREEACC_DIM,
        ],
        is_interpolated=feature_window[:, :, -1] > 0.5,
        neutral_pose_mode=neutral_pose_mode,
        gyr_motion_scale=gyr_motion_scale,
    )


def component_feature_slices() -> dict[str, slice]:
    return {
        "rot6d": slice(0, RAW_IMU_ROT6D_DIM),
        "gyr": slice(RAW_IMU_ROT6D_DIM, RAW_IMU_ROT6D_DIM + RAW_IMU_GYR_DIM),
        "freeacc": slice(
            RAW_IMU_ROT6D_DIM + RAW_IMU_GYR_DIM,
            RAW_IMU_ROT6D_DIM + RAW_IMU_GYR_DIM + RAW_IMU_FREEACC_DIM,
        ),
        "is_interpolated": slice(RAW_IMU_SENSOR_DIM - 1, RAW_IMU_SENSOR_DIM),
    }


def reshape_flat_feature_tensor(flat_feature: torch.Tensor) -> torch.Tensor:
    if flat_feature.ndim != 3 or flat_feature.shape[-1] != RAW_IMU_INPUT_DIM:
        raise ValueError(
            f"flat_feature must have shape [B,T,{RAW_IMU_INPUT_DIM}], got {tuple(flat_feature.shape)}"
        )
    return flat_feature.view(flat_feature.shape[0], flat_feature.shape[1], RAW_IMU_SENSOR_COUNT, RAW_IMU_SENSOR_DIM)


def compute_raw_imu_component_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pred = reshape_flat_feature_tensor(prediction)
    gold = reshape_flat_feature_tensor(target)
    slices = component_feature_slices()
    rot_mse = torch.mean((pred[..., slices["rot6d"]] - gold[..., slices["rot6d"]]) ** 2)
    gyr_mse = torch.mean((pred[..., slices["gyr"]] - gold[..., slices["gyr"]]) ** 2)
    freeacc_mse = torch.mean((pred[..., slices["freeacc"]] - gold[..., slices["freeacc"]]) ** 2)
    interp_mse = torch.mean((pred[..., slices["is_interpolated"]] - gold[..., slices["is_interpolated"]]) ** 2)
    feature_mse = torch.mean((prediction - target) ** 2)
    return {
        "feature_mse": feature_mse,
        "rot_mse": rot_mse,
        "gyr_mse": gyr_mse,
        "freeacc_mse": freeacc_mse,
        "interp_mse": interp_mse,
    }


def compute_raw_imu_recon_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    component_mse = compute_raw_imu_component_mse(prediction, target)
    return {
        "feature_mse": float(component_mse["feature_mse"].item()),
        "feature_rmse": float(torch.sqrt(component_mse["feature_mse"]).item()),
        "rot_mse": float(component_mse["rot_mse"].item()),
        "rot_rmse": float(torch.sqrt(component_mse["rot_mse"]).item()),
        "gyr_mse": float(component_mse["gyr_mse"].item()),
        "gyr_rmse": float(torch.sqrt(component_mse["gyr_mse"]).item()),
        "freeacc_mse": float(component_mse["freeacc_mse"].item()),
        "freeacc_rmse": float(torch.sqrt(component_mse["freeacc_mse"]).item()),
        "interp_mse": float(component_mse["interp_mse"].item()),
        "interp_rmse": float(torch.sqrt(component_mse["interp_mse"]).item()),
    }
