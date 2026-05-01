#!/usr/bin/env python3
"""
Temporal VAE over 20Hz pseudo-pose windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from imu_new2_common import JOINT_ORDER
from models.pose_generative import TemporalPoseVAE
from train_imu_masked_recon import (
    append_jsonl,
    build_scheduler,
    resolve_device,
    rot6d_to_rotation_matrix,
    set_global_seed,
    write_json,
)

POSE_POSITION_DIM = 10 * 3
ROOT_HEADING_DIM = 6
WINDOW_SUBSET_COLUMNS = (
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "end_frame_20hz",
    "valid_frames",
)
POSITION_SMOOTHING_KERNELS: dict[str, tuple[float, ...] | None] = {
    "none": None,
    "tri3": (1.0, 2.0, 1.0),
    "tri5": (1.0, 2.0, 3.0, 2.0, 1.0),
}


@dataclass(frozen=True)
class PoseWindowRecord:
    pose_path: Path
    feature_path: Path
    start_frame: int
    valid_frames: int


@dataclass(frozen=True)
class TemporalVaeTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_frames: int = 240
    train_stride_frames: int = 20
    val_stride_frames: int = 120
    hidden_dim: int = 128
    latent_dim: int = 64
    beta: float = 1e-3
    position_loss_weight: float = 30.0 / 36.0
    position_velocity_loss_weight: float = 0.0
    heading_loss_weight: float = 6.0 / 36.0
    heading_forward_loss_weight: float = 0.0
    distal_joint_scale: float = 1.0
    temporal_smoothness_loss_weight: float = 0.0
    heading_velocity_loss_weight: float = 0.0
    heading_acceleration_loss_weight: float = 0.0
    dropout: float = 0.1
    decoder_mode: str = "transpose_conv"
    logvar_min: float = -8.0
    logvar_max: float = 8.0
    max_grad_norm: float = 5.0
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 0
    scheduler_type: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.5
    scheduler_t_max: int = 0
    max_train_windows: int = 0
    max_val_windows: int = 0
    overfit_windows: int = 0
    shuffle_train_windows: bool = False
    shuffle_val_windows: bool = False
    window_subset_seed: int = 0
    sample_export_count: int = 0
    sample_seed: int = 0
    sample_position_smoothing_kernel: str = "tri5"
    normalize_pose: bool = True


def read_pose_window_records_from_csv(window_index_csv: Path) -> list[PoseWindowRecord]:
    records: list[PoseWindowRecord] = []
    with Path(window_index_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_frame = int(row["start_frame_20hz"])
            end_frame = int(row["end_frame_20hz"])
            records.append(
                PoseWindowRecord(
                    pose_path=Path(row["pose_path"]),
                    feature_path=Path(row["feature_path"]),
                    start_frame=start_frame,
                    valid_frames=end_frame - start_frame + 1,
                )
            )
    if not records:
        raise ValueError(f"No windows found in {window_index_csv}")
    return records


def select_pose_window_records(
    records: list[PoseWindowRecord],
    *,
    max_windows: int = 0,
    shuffle: bool = False,
    subset_seed: int = 0,
) -> list[PoseWindowRecord]:
    selected = list(records)
    if shuffle and len(selected) > 1:
        rng = np.random.default_rng(subset_seed)
        permutation = rng.permutation(len(selected)).tolist()
        selected = [selected[int(index)] for index in permutation]
    if max_windows > 0:
        selected = selected[:max_windows]
    return selected


def write_pose_window_subset_csv(path: Path, records: list[PoseWindowRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=WINDOW_SUBSET_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "pose_path": str(record.pose_path),
                    "feature_path": str(record.feature_path),
                    "start_frame_20hz": int(record.start_frame),
                    "end_frame_20hz": int(record.start_frame + record.valid_frames - 1),
                    "valid_frames": int(record.valid_frames),
                }
            )


def flatten_pose_window(relative_positions: np.ndarray, root_heading_6d: np.ndarray) -> np.ndarray:
    frame_count = int(relative_positions.shape[0])
    return np.concatenate(
        [
            relative_positions.reshape(frame_count, -1).astype(np.float32),
            root_heading_6d.astype(np.float32),
        ],
        axis=1,
    )


def root_heading_6d_to_angles(root_heading_6d: np.ndarray) -> np.ndarray:
    root_heading_6d = np.asarray(root_heading_6d, dtype=np.float64)
    if root_heading_6d.ndim != 2 or root_heading_6d.shape[1] != ROOT_HEADING_DIM:
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


def rebase_root_heading_6d(
    root_heading_6d: np.ndarray,
    *,
    reference_root_heading_6d: np.ndarray | None = None,
) -> np.ndarray:
    # VAE windows model root heading delta, not absolute segment-global yaw.
    # By default, rebase every window to its first frame; conditional prediction
    # can instead anchor future heading to the last observed context frame.
    heading_angles = root_heading_6d_to_angles(root_heading_6d)
    if heading_angles.shape[0] == 0:
        return np.zeros((0, ROOT_HEADING_DIM), dtype=np.float32)

    if reference_root_heading_6d is None:
        reference_angle = float(heading_angles[0])
    else:
        reference_angles = root_heading_6d_to_angles(np.asarray(reference_root_heading_6d, dtype=np.float64))
        if reference_angles.shape[0] == 0:
            raise ValueError("reference_root_heading_6d must contain at least one frame")
        reference_angle = float(reference_angles[-1])

    heading_delta = heading_angles - reference_angle
    return angles_to_root_heading_6d(heading_delta)


class PseudoPoseWindowDataset(Dataset):
    def __init__(self, *, window_records: list[PoseWindowRecord], window_frames: int = 240) -> None:
        self.window_records = list(window_records)
        self.window_frames = int(window_frames)
        self._payload_cache: dict[Path, dict[str, np.ndarray]] = {}
        if not self.window_records:
            raise ValueError("No temporal VAE windows available")

    @classmethod
    def from_window_index_csv(
        cls,
        *,
        window_index_csv: Path,
        max_windows: int = 0,
        shuffle: bool = False,
        subset_seed: int = 0,
    ) -> "PseudoPoseWindowDataset":
        records = read_pose_window_records_from_csv(window_index_csv)
        records = select_pose_window_records(
            records,
            max_windows=max_windows,
            shuffle=shuffle,
            subset_seed=subset_seed,
        )
        if not records:
            raise ValueError("No temporal VAE windows available")
        window_frames = records[0].valid_frames
        return cls(window_records=records, window_frames=window_frames)

    def _load_payload(self, pose_path: Path) -> dict[str, np.ndarray]:
        if pose_path not in self._payload_cache:
            with np.load(pose_path, allow_pickle=False) as payload:
                self._payload_cache[pose_path] = {key: payload[key] for key in payload.files}
        return self._payload_cache[pose_path]

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.window_records[index]
        payload = self._load_payload(record.pose_path)
        packet_counter = payload["packet_counter"].astype(np.int64)
        relative_positions = payload["relative_positions"].astype(np.float32)
        root_heading_6d = payload["root_heading_6d"].astype(np.float32)

        end_frame = record.start_frame + min(self.window_frames, record.valid_frames)
        window_relative_positions = relative_positions[record.start_frame:end_frame]
        window_root_heading_6d = rebase_root_heading_6d(root_heading_6d[record.start_frame:end_frame])
        window_packet_counter = packet_counter[record.start_frame:end_frame]
        pose = flatten_pose_window(window_relative_positions, window_root_heading_6d)

        participant = record.pose_path.parent.name
        segment_id = record.pose_path.stem.removeprefix("segment_")
        return {
            "pose": torch.from_numpy(pose),
            "meta": {
                "participant": participant,
                "segment_id": segment_id,
                "pose_path": str(record.pose_path),
                "feature_path": str(record.feature_path),
                "start_frame_20hz": int(record.start_frame),
                "valid_frames": int(pose.shape[0]),
                "packet_start_20hz": int(window_packet_counter[0]) if window_packet_counter.size > 0 else -1,
                "packet_end_20hz": int(window_packet_counter[-1]) if window_packet_counter.size > 0 else -1,
            },
        }


def dataset_from_pose_records(records: list[PoseWindowRecord]) -> PseudoPoseWindowDataset:
    if not records:
        raise ValueError("No temporal VAE windows available")
    return PseudoPoseWindowDataset(window_records=records, window_frames=records[0].valid_frames)


def limit_pose_dataset_windows(dataset: PseudoPoseWindowDataset, *, max_windows: int) -> PseudoPoseWindowDataset:
    if max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    return dataset_from_pose_records(dataset.window_records[:max_windows])


def compute_pose_normalization_stats(dataset: PseudoPoseWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([dataset[index]["pose"].numpy() for index in range(len(dataset))], axis=0)
    mean = stacked.reshape(-1, stacked.shape[-1]).mean(axis=0).astype(np.float32)
    std = stacked.reshape(-1, stacked.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def normalize_pose_tensor(pose: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (pose - mean.view(1, 1, -1)) / std.view(1, 1, -1)


def denormalize_pose_tensor(pose: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return pose * std.view(1, 1, -1) + mean.view(1, 1, -1)


def compute_heading_forward_loss(
    prediction_heading: torch.Tensor,
    target_heading: torch.Tensor,
) -> torch.Tensor:
    pred_rotation = rot6d_to_rotation_matrix(prediction_heading)
    target_rotation = rot6d_to_rotation_matrix(target_heading)
    pred_forward = pred_rotation[..., 0]
    target_forward = target_rotation[..., 0]
    dot = torch.sum(pred_forward * target_forward, dim=-1).clamp(-1.0, 1.0)
    return torch.mean(1.0 - dot)


def build_position_joint_weights(
    *,
    device: torch.device,
    dtype: torch.dtype,
    distal_joint_scale: float,
) -> torch.Tensor:
    weights = torch.ones((len(JOINT_ORDER),), device=device, dtype=dtype)
    distal_joint_names = ("left_hand", "right_hand", "left_foot", "right_foot")
    for joint_name in distal_joint_names:
        weights[JOINT_ORDER.index(joint_name)] = float(distal_joint_scale)
    return weights


def compute_position_recon_mse(
    prediction_positions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    distal_joint_scale: float,
) -> torch.Tensor:
    pred = prediction_positions.reshape(prediction_positions.shape[0], prediction_positions.shape[1], len(JOINT_ORDER), 3)
    target = target_positions.reshape(target_positions.shape[0], target_positions.shape[1], len(JOINT_ORDER), 3)
    joint_weights = build_position_joint_weights(
        device=prediction_positions.device,
        dtype=prediction_positions.dtype,
        distal_joint_scale=distal_joint_scale,
    ).view(1, 1, len(JOINT_ORDER), 1)
    squared_error = (pred - target) ** 2
    return torch.sum(squared_error * joint_weights) / (squared_error.numel() / 3.0 * torch.sum(joint_weights[0, 0, :, 0]) / len(JOINT_ORDER))


def compute_position_velocity_loss(
    prediction_positions: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    distal_joint_scale: float,
) -> torch.Tensor:
    if prediction_positions.shape[1] <= 1:
        return prediction_positions.new_zeros(())
    pred = prediction_positions.reshape(prediction_positions.shape[0], prediction_positions.shape[1], len(JOINT_ORDER), 3)
    target = target_positions.reshape(target_positions.shape[0], target_positions.shape[1], len(JOINT_ORDER), 3)
    pred_velocity = pred[:, 1:] - pred[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    joint_weights = build_position_joint_weights(
        device=prediction_positions.device,
        dtype=prediction_positions.dtype,
        distal_joint_scale=distal_joint_scale,
    ).view(1, 1, len(JOINT_ORDER), 1)
    squared_error = (pred_velocity - target_velocity) ** 2
    return torch.sum(squared_error * joint_weights) / (squared_error.numel() / 3.0 * torch.sum(joint_weights[0, 0, :, 0]) / len(JOINT_ORDER))


def resolve_position_smoothing_kernel(name: str) -> tuple[float, ...] | None:
    kernel = POSITION_SMOOTHING_KERNELS.get(str(name))
    if str(name) not in POSITION_SMOOTHING_KERNELS:
        raise ValueError(f"Unsupported position smoothing kernel: {name}")
    return kernel


def apply_position_temporal_filter(
    pose: torch.Tensor,
    *,
    kernel_weights: tuple[float, ...] | None,
) -> torch.Tensor:
    if kernel_weights is None or pose.shape[1] <= 1:
        return pose
    kernel = torch.as_tensor(kernel_weights, dtype=pose.dtype, device=pose.device)
    kernel = kernel / torch.sum(kernel)
    pad = int(kernel.shape[0] // 2)

    positions = pose[..., :POSE_POSITION_DIM].reshape(pose.shape[0], pose.shape[1], len(JOINT_ORDER), 3)
    channels_last = positions.permute(0, 2, 3, 1).reshape(-1, 1, pose.shape[1])
    padded = torch.nn.functional.pad(channels_last, (pad, pad), mode="replicate")
    filtered = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1))
    filtered = filtered.reshape(pose.shape[0], len(JOINT_ORDER), 3, pose.shape[1]).permute(0, 3, 1, 2).reshape(
        pose.shape[0], pose.shape[1], POSE_POSITION_DIM
    )

    output = pose.clone()
    output[..., :POSE_POSITION_DIM] = filtered
    return output


def compute_temporal_smoothness_loss(
    prediction_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    if prediction_positions.shape[1] <= 2:
        return prediction_positions.new_zeros(())
    pred_positions = prediction_positions.reshape(prediction_positions.shape[0], prediction_positions.shape[1], 10, 3)
    target_positions = target_positions.reshape(target_positions.shape[0], target_positions.shape[1], 10, 3)
    pred_acceleration = pred_positions[:, 2:] - 2.0 * pred_positions[:, 1:-1] + pred_positions[:, :-2]
    target_acceleration = target_positions[:, 2:] - 2.0 * target_positions[:, 1:-1] + target_positions[:, :-2]
    return torch.mean((pred_acceleration - target_acceleration) ** 2)


def compute_heading_velocity_acceleration_losses(
    prediction_heading: torch.Tensor,
    target_heading: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_rotation = rot6d_to_rotation_matrix(prediction_heading)
    target_rotation = rot6d_to_rotation_matrix(target_heading)
    pred_forward = pred_rotation[..., 0]
    target_forward = target_rotation[..., 0]

    if prediction_heading.shape[1] <= 1:
        velocity_loss = prediction_heading.new_zeros(())
    else:
        pred_velocity = pred_forward[:, 1:] - pred_forward[:, :-1]
        target_velocity = target_forward[:, 1:] - target_forward[:, :-1]
        velocity_loss = torch.mean((pred_velocity - target_velocity) ** 2)

    if prediction_heading.shape[1] <= 2:
        acceleration_loss = prediction_heading.new_zeros(())
    else:
        pred_acceleration = pred_forward[:, 2:] - 2.0 * pred_forward[:, 1:-1] + pred_forward[:, :-2]
        target_acceleration = target_forward[:, 2:] - 2.0 * target_forward[:, 1:-1] + target_forward[:, :-2]
        acceleration_loss = torch.mean((pred_acceleration - target_acceleration) ** 2)

    return velocity_loss, acceleration_loss


def compute_temporal_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    reconstruction_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    position_loss_weight: float,
    position_velocity_loss_weight: float,
    heading_loss_weight: float,
    heading_forward_loss_weight: float,
    distal_joint_scale: float,
    temporal_smoothness_loss_weight: float,
    heading_velocity_loss_weight: float,
    heading_acceleration_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    position_recon_mse = compute_position_recon_mse(
        reconstruction[..., :POSE_POSITION_DIM],
        target[..., :POSE_POSITION_DIM],
        distal_joint_scale=distal_joint_scale,
    )
    heading_recon_mse = torch.mean((reconstruction[..., POSE_POSITION_DIM:] - target[..., POSE_POSITION_DIM:]) ** 2)
    recon_mse = torch.mean((reconstruction - target) ** 2)
    position_velocity_loss = compute_position_velocity_loss(
        reconstruction_raw[..., :POSE_POSITION_DIM],
        target_raw[..., :POSE_POSITION_DIM],
        distal_joint_scale=distal_joint_scale,
    )
    heading_forward_loss = compute_heading_forward_loss(
        reconstruction_raw[..., POSE_POSITION_DIM:],
        target_raw[..., POSE_POSITION_DIM:],
    )
    temporal_smoothness_loss = compute_temporal_smoothness_loss(
        reconstruction_raw[..., :POSE_POSITION_DIM],
        target_raw[..., :POSE_POSITION_DIM],
    )
    heading_velocity_loss, heading_acceleration_loss = compute_heading_velocity_acceleration_losses(
        reconstruction_raw[..., POSE_POSITION_DIM:],
        target_raw[..., POSE_POSITION_DIM:],
    )
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    weighted_recon_loss = (
        float(position_loss_weight) * position_recon_mse
        + float(position_velocity_loss_weight) * position_velocity_loss
        + float(heading_loss_weight) * heading_recon_mse
        + float(heading_forward_loss_weight) * heading_forward_loss
        + float(temporal_smoothness_loss_weight) * temporal_smoothness_loss
        + float(heading_velocity_loss_weight) * heading_velocity_loss
        + float(heading_acceleration_loss_weight) * heading_acceleration_loss
    )
    loss = weighted_recon_loss + float(beta) * kl
    return loss, {
        "recon_loss": float(loss.item()),
        "weighted_recon_loss": float(weighted_recon_loss.item()),
        "recon_mse": float(recon_mse.item()),
        "position_recon_mse": float(position_recon_mse.item()),
        "position_velocity_loss": float(position_velocity_loss.item()),
        "heading_recon_mse": float(heading_recon_mse.item()),
        "heading_forward_loss": float(heading_forward_loss.item()),
        "temporal_smoothness_loss": float(temporal_smoothness_loss.item()),
        "heading_velocity_loss": float(heading_velocity_loss.item()),
        "heading_acceleration_loss": float(heading_acceleration_loss.item()),
        "kl_loss": float(kl.item()),
    }


def compute_pose_recon_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_positions = prediction[..., :POSE_POSITION_DIM].reshape(prediction.shape[0], prediction.shape[1], 10, 3)
    target_positions = target[..., :POSE_POSITION_DIM].reshape(target.shape[0], target.shape[1], 10, 3)
    pred_heading = prediction[..., POSE_POSITION_DIM:]
    target_heading = target[..., POSE_POSITION_DIM:]

    position_mse = torch.mean((pred_positions - target_positions) ** 2)
    heading_mse = torch.mean((pred_heading - target_heading) ** 2)
    mpjpe = torch.linalg.norm(pred_positions - target_positions, dim=-1).mean()
    position_rmse = torch.sqrt(position_mse)
    heading_rmse = torch.sqrt(heading_mse)

    pred_rotation = rot6d_to_rotation_matrix(pred_heading)
    target_rotation = rot6d_to_rotation_matrix(target_heading)
    pred_forward = pred_rotation[..., 0]
    target_forward = target_rotation[..., 0]
    dot = torch.sum(pred_forward * target_forward, dim=-1).clamp(-1.0, 1.0)
    heading_error_deg = torch.rad2deg(torch.acos(dot)).mean()

    if prediction.shape[1] <= 2:
        jerk_error = prediction.new_zeros(())
    else:
        pred_jerk = pred_positions[:, 2:] - 2.0 * pred_positions[:, 1:-1] + pred_positions[:, :-2]
        target_jerk = target_positions[:, 2:] - 2.0 * target_positions[:, 1:-1] + target_positions[:, :-2]
        jerk_error = torch.linalg.norm(pred_jerk - target_jerk, dim=-1).mean()

    return {
        "position_mse": float(position_mse.item()),
        "heading_mse": float(heading_mse.item()),
        "position_rmse": float(position_rmse.item()),
        "heading_rmse": float(heading_rmse.item()),
        "recon_mpjpe": float(mpjpe.item()),
        "root_heading_error_deg": float(heading_error_deg.item()),
        "jerk_error": float(jerk_error.item()),
    }


def merge_metric_sums(total: dict[str, float], update: dict[str, float], *, weight: float) -> dict[str, float]:
    total["count"] = total.get("count", 0.0) + float(weight)
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value) * float(weight)
    return total


def finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float]:
    count = metric_sums.get("count", 0.0)
    if count <= 0.0:
        return {
            "recon_loss": 0.0,
            "weighted_recon_loss": 0.0,
            "recon_mse": 0.0,
            "position_recon_mse": 0.0,
            "position_velocity_loss": 0.0,
            "heading_recon_mse": 0.0,
            "heading_forward_loss": 0.0,
            "temporal_smoothness_loss": 0.0,
            "heading_velocity_loss": 0.0,
            "heading_acceleration_loss": 0.0,
            "kl_loss": 0.0,
            "position_mse": 0.0,
            "heading_mse": 0.0,
            "position_rmse": 0.0,
            "heading_rmse": 0.0,
            "recon_mpjpe": 0.0,
            "root_heading_error_deg": 0.0,
            "jerk_error": 0.0,
        }
    return {
        key: float(value / count)
        for key, value in metric_sums.items()
        if key != "count"
    }


def all_tensors_finite(*tensors: torch.Tensor) -> bool:
    for tensor in tensors:
        if not torch.isfinite(tensor).all():
            return False
    return True


def model_parameters_finite(model: torch.nn.Module) -> bool:
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            return False
    return True


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_train_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    position_loss_weight: float,
    position_velocity_loss_weight: float,
    heading_loss_weight: float,
    heading_forward_loss_weight: float,
    distal_joint_scale: float,
    temporal_smoothness_loss_weight: float,
    heading_velocity_loss_weight: float,
    heading_acceleration_loss_weight: float,
    normalize_pose: bool,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    skipped_nonfinite_batches = 0.0
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        pose = batch["pose"]
        target = normalize_pose_tensor(pose, pose_mean, pose_std) if normalize_pose else pose
        reconstruction, mu, logvar = model(target)
        if not all_tensors_finite(reconstruction, mu, logvar):
            skipped_nonfinite_batches += 1.0
            optimizer.zero_grad(set_to_none=True)
            continue
        reconstruction_raw = denormalize_pose_tensor(reconstruction, pose_mean, pose_std) if normalize_pose else reconstruction
        if not all_tensors_finite(reconstruction_raw):
            skipped_nonfinite_batches += 1.0
            optimizer.zero_grad(set_to_none=True)
            continue
        pose_raw = pose
        loss, loss_metrics = compute_temporal_vae_loss(
            reconstruction=reconstruction,
            target=target,
            reconstruction_raw=reconstruction_raw,
            target_raw=pose_raw,
            mu=mu,
            logvar=logvar,
            beta=beta,
            position_loss_weight=position_loss_weight,
            position_velocity_loss_weight=position_velocity_loss_weight,
            heading_loss_weight=heading_loss_weight,
            heading_forward_loss_weight=heading_forward_loss_weight,
            distal_joint_scale=distal_joint_scale,
            temporal_smoothness_loss_weight=temporal_smoothness_loss_weight,
            heading_velocity_loss_weight=heading_velocity_loss_weight,
            heading_acceleration_loss_weight=heading_acceleration_loss_weight,
        )
        if not torch.isfinite(loss):
            skipped_nonfinite_batches += 1.0
            optimizer.zero_grad(set_to_none=True)
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(max_grad_norm),
            error_if_nonfinite=False,
        )
        if not torch.isfinite(grad_norm):
            skipped_nonfinite_batches += 1.0
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        if not model_parameters_finite(model):
            raise RuntimeError("Temporal VAE parameters became non-finite after optimizer.step()")

        batch_metrics = loss_metrics | compute_pose_recon_metrics(
            prediction=reconstruction_raw.detach(),
            target=pose_raw.detach(),
        )
        merge_metric_sums(metric_sums, batch_metrics, weight=pose.shape[0])
    metrics = finalize_metric_sums(metric_sums)
    metrics["skipped_nonfinite_batches"] = float(skipped_nonfinite_batches)
    return metrics


@torch.no_grad()
def evaluate_temporal_vae(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    position_loss_weight: float,
    position_velocity_loss_weight: float,
    heading_loss_weight: float,
    heading_forward_loss_weight: float,
    distal_joint_scale: float,
    temporal_smoothness_loss_weight: float,
    heading_velocity_loss_weight: float,
    heading_acceleration_loss_weight: float,
    normalize_pose: bool,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    position_smoothing_kernel: tuple[float, ...] | None = None,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        pose = batch["pose"]
        target = normalize_pose_tensor(pose, pose_mean, pose_std) if normalize_pose else pose
        reconstruction, mu, logvar = model(target)
        if not all_tensors_finite(reconstruction, mu, logvar):
            raise RuntimeError("Temporal VAE produced non-finite outputs during evaluation")
        reconstruction_raw = denormalize_pose_tensor(reconstruction, pose_mean, pose_std) if normalize_pose else reconstruction
        reconstruction_raw = apply_position_temporal_filter(
            reconstruction_raw,
            kernel_weights=position_smoothing_kernel,
        )
        if not all_tensors_finite(reconstruction_raw):
            raise RuntimeError("Temporal VAE produced non-finite reconstruction during evaluation")
        _, loss_metrics = compute_temporal_vae_loss(
            reconstruction=reconstruction,
            target=target,
            reconstruction_raw=reconstruction_raw,
            target_raw=pose,
            mu=mu,
            logvar=logvar,
            beta=beta,
            position_loss_weight=position_loss_weight,
            position_velocity_loss_weight=position_velocity_loss_weight,
            heading_loss_weight=heading_loss_weight,
            heading_forward_loss_weight=heading_forward_loss_weight,
            distal_joint_scale=distal_joint_scale,
            temporal_smoothness_loss_weight=temporal_smoothness_loss_weight,
            heading_velocity_loss_weight=heading_velocity_loss_weight,
            heading_acceleration_loss_weight=heading_acceleration_loss_weight,
        )
        batch_metrics = loss_metrics | compute_pose_recon_metrics(prediction=reconstruction_raw, target=pose)
        merge_metric_sums(metric_sums, batch_metrics, weight=pose.shape[0])
    return finalize_metric_sums(metric_sums)


@torch.no_grad()
def export_temporal_vae_samples(
    *,
    model: torch.nn.Module,
    dataset: PseudoPoseWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    normalize_pose: bool,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    position_smoothing_kernel: tuple[float, ...] | None,
) -> None:
    if sample_count <= 0:
        return
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(sample_count), len(dataset))
    rng = np.random.default_rng(sample_seed + epoch)
    if count == len(dataset):
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.sort(rng.choice(len(dataset), size=count, replace=False))

    samples = [dataset[int(index)] for index in indices]
    pose = torch.stack([sample["pose"] for sample in samples], dim=0).to(device)
    target = normalize_pose_tensor(pose, pose_mean, pose_std) if normalize_pose else pose
    reconstruction, mu, logvar = model(target)
    if not all_tensors_finite(reconstruction, mu, logvar):
        raise RuntimeError("Temporal VAE produced non-finite outputs during sample export")
    reconstruction_raw = denormalize_pose_tensor(reconstruction, pose_mean, pose_std) if normalize_pose else reconstruction
    reconstruction_raw = apply_position_temporal_filter(
        reconstruction_raw,
        kernel_weights=position_smoothing_kernel,
    )
    if not all_tensors_finite(reconstruction_raw):
        raise RuntimeError("Temporal VAE produced non-finite reconstruction during sample export")
    meta_json = json.dumps([sample["meta"] for sample in samples])

    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        target=pose.detach().cpu().numpy().astype(np.float32),
        reconstruction=reconstruction_raw.detach().cpu().numpy().astype(np.float32),
        mu=mu.detach().cpu().numpy().astype(np.float32),
        logvar=logvar.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: TemporalVaeTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    pose_mean: np.ndarray,
    pose_std: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "config": asdict(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "pose_mean": pose_mean.astype(np.float32),
            "pose_std": pose_std.astype(np.float32),
        },
        path,
    )


def run_temporal_vae_training(
    *,
    output_dir: Path,
    config: TemporalVaeTrainingConfig,
    train_window_index_csv: Path,
    val_window_index_csv: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    train_records = read_pose_window_records_from_csv(train_window_index_csv)
    val_records = read_pose_window_records_from_csv(val_window_index_csv)

    if config.overfit_windows > 0:
        overfit_records = select_pose_window_records(
            train_records,
            max_windows=config.overfit_windows,
            shuffle=config.shuffle_train_windows,
            subset_seed=config.window_subset_seed,
        )
        train_selected_records = list(overfit_records)
        val_selected_records = list(overfit_records)
        train_dataset = dataset_from_pose_records(overfit_records)
        val_dataset = dataset_from_pose_records(overfit_records)
    else:
        train_selected_records = select_pose_window_records(
            train_records,
            max_windows=config.max_train_windows,
            shuffle=config.shuffle_train_windows,
            subset_seed=config.window_subset_seed,
        )
        val_selected_records = select_pose_window_records(
            val_records,
            max_windows=config.max_val_windows,
            shuffle=config.shuffle_val_windows,
            subset_seed=config.window_subset_seed + 1,
        )
        train_dataset = dataset_from_pose_records(train_selected_records)
        val_dataset = dataset_from_pose_records(val_selected_records)

    write_pose_window_subset_csv(output_dir / "train_window_subset.csv", train_selected_records)
    write_pose_window_subset_csv(output_dir / "val_window_subset.csv", val_selected_records)

    pose_mean_np, pose_std_np = compute_pose_normalization_stats(train_dataset)
    pose_mean = torch.from_numpy(pose_mean_np).to(device=device, dtype=torch.float32)
    pose_std = torch.from_numpy(pose_std_np).to(device=device, dtype=torch.float32)
    np.savez_compressed(
        output_dir / "normalization_stats.npz",
        pose_mean=pose_mean_np.astype(np.float32),
        pose_std=pose_std_np.astype(np.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = TemporalPoseVAE(
        input_dim=36,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        window_frames=config.window_frames,
        dropout=config.dropout,
        decoder_mode=config.decoder_mode,
        logvar_min=config.logvar_min,
        logvar_max=config.logvar_max,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    args_payload = asdict(config) | {
        "train_window_index_csv": str(train_window_index_csv),
        "val_window_index_csv": str(val_window_index_csv),
        "device_resolved": str(device),
    }
    write_json(output_dir / "args.json", args_payload)
    sample_position_smoothing_kernel = resolve_position_smoothing_kernel(config.sample_position_smoothing_kernel)

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_metric = float("inf")
    best_epoch = 0
    best_record: dict[str, float] | None = None
    best_heading_metric = float("inf")
    best_heading_epoch = 0
    best_heading_record: dict[str, float] | None = None
    best_jerk_metric = float("inf")
    best_jerk_epoch = 0
    best_jerk_record: dict[str, float] | None = None
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=config.beta,
            position_loss_weight=config.position_loss_weight,
            position_velocity_loss_weight=config.position_velocity_loss_weight,
            heading_loss_weight=config.heading_loss_weight,
            heading_forward_loss_weight=config.heading_forward_loss_weight,
            distal_joint_scale=config.distal_joint_scale,
            temporal_smoothness_loss_weight=config.temporal_smoothness_loss_weight,
            heading_velocity_loss_weight=config.heading_velocity_loss_weight,
            heading_acceleration_loss_weight=config.heading_acceleration_loss_weight,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            max_grad_norm=config.max_grad_norm,
        )
        val_metrics = evaluate_temporal_vae(
            model=model,
            dataloader=val_loader,
            device=device,
            beta=config.beta,
            position_loss_weight=config.position_loss_weight,
            position_velocity_loss_weight=config.position_velocity_loss_weight,
            heading_loss_weight=config.heading_loss_weight,
            heading_forward_loss_weight=config.heading_forward_loss_weight,
            distal_joint_scale=config.distal_joint_scale,
            temporal_smoothness_loss_weight=config.temporal_smoothness_loss_weight,
            heading_velocity_loss_weight=config.heading_velocity_loss_weight,
            heading_acceleration_loss_weight=config.heading_acceleration_loss_weight,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            position_smoothing_kernel=None,
        )
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        append_jsonl(metrics_path, record)

        export_temporal_vae_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_reconstructions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            position_smoothing_kernel=sample_position_smoothing_kernel,
        )
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            output_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            pose_mean=pose_mean_np,
            pose_std=pose_std_np,
        )
        if record["val_recon_mpjpe"] <= best_metric:
            best_metric = record["val_recon_mpjpe"]
            best_epoch = epoch
            best_record = dict(record)
            save_checkpoint(
                output_dir / "best_by_mpjpe.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                pose_mean=pose_mean_np,
                pose_std=pose_std_np,
            )
            save_checkpoint(
                output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                pose_mean=pose_mean_np,
                pose_std=pose_std_np,
            )
        if record["val_root_heading_error_deg"] <= best_heading_metric:
            best_heading_metric = record["val_root_heading_error_deg"]
            best_heading_epoch = epoch
            best_heading_record = dict(record)
            save_checkpoint(
                output_dir / "best_by_heading.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                pose_mean=pose_mean_np,
                pose_std=pose_std_np,
            )
        if record["val_jerk_error"] <= best_jerk_metric:
            best_jerk_metric = record["val_jerk_error"]
            best_jerk_epoch = epoch
            best_jerk_record = dict(record)
            save_checkpoint(
                output_dir / "best_by_jerk.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                pose_mean=pose_mean_np,
                pose_std=pose_std_np,
            )

    return {
        "best_epoch": int(best_epoch),
        "best_val_recon_mpjpe": float(best_metric),
        "best_val_root_heading_error_deg": float(0.0 if best_record is None else best_record["val_root_heading_error_deg"]),
        "best_val_heading_rmse": float(0.0 if best_record is None else best_record["val_heading_rmse"]),
        "best_epoch_by_heading": int(best_heading_epoch),
        "best_val_recon_mpjpe_by_heading": float(0.0 if best_heading_record is None else best_heading_record["val_recon_mpjpe"]),
        "best_val_root_heading_error_deg_by_heading": float(best_heading_metric if best_heading_record is not None else 0.0),
        "best_val_heading_rmse_by_heading": float(0.0 if best_heading_record is None else best_heading_record["val_heading_rmse"]),
        "best_epoch_by_jerk": int(best_jerk_epoch),
        "best_val_recon_mpjpe_by_jerk": float(0.0 if best_jerk_record is None else best_jerk_record["val_recon_mpjpe"]),
        "best_val_root_heading_error_deg_by_jerk": float(0.0 if best_jerk_record is None else best_jerk_record["val_root_heading_error_deg"]),
        "best_val_jerk_error": float(best_jerk_metric if best_jerk_record is not None else 0.0),
        "output_dir": str(output_dir),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal VAE on 20Hz pseudo-pose windows")
    parser.add_argument("--train-window-index-csv", type=Path, required=True)
    parser.add_argument("--val-window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--val-stride-frames", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--position-loss-weight", type=float, default=30.0 / 36.0)
    parser.add_argument("--position-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--heading-loss-weight", type=float, default=6.0 / 36.0)
    parser.add_argument("--heading-forward-loss-weight", type=float, default=0.0)
    parser.add_argument("--distal-joint-scale", type=float, default=1.0)
    parser.add_argument("--temporal-smoothness-loss-weight", type=float, default=0.0)
    parser.add_argument("--heading-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--heading-acceleration-loss-weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-mode", type=str, default="transpose_conv", choices=("transpose_conv", "upsample_conv"))
    parser.add_argument("--logvar-min", type=float, default=-8.0)
    parser.add_argument("--logvar-max", type=float, default=8.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler-type", type=str, default="none", choices=("none", "step", "cosine"))
    parser.add_argument("--scheduler-step-size", type=int, default=1)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--overfit-windows", type=int, default=0)
    parser.add_argument("--shuffle-train-windows", action="store_true")
    parser.add_argument("--shuffle-val-windows", action="store_true")
    parser.add_argument("--window-subset-seed", type=int, default=0)
    parser.add_argument("--sample-export-count", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--no-normalize-pose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_temporal_vae_training(
        output_dir=args.output_dir,
        config=TemporalVaeTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            window_frames=args.window_frames,
            train_stride_frames=args.train_stride_frames,
            val_stride_frames=args.val_stride_frames,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta=args.beta,
            position_loss_weight=args.position_loss_weight,
            position_velocity_loss_weight=args.position_velocity_loss_weight,
            heading_loss_weight=args.heading_loss_weight,
            heading_forward_loss_weight=args.heading_forward_loss_weight,
            distal_joint_scale=args.distal_joint_scale,
            temporal_smoothness_loss_weight=args.temporal_smoothness_loss_weight,
            heading_velocity_loss_weight=args.heading_velocity_loss_weight,
            heading_acceleration_loss_weight=args.heading_acceleration_loss_weight,
            dropout=args.dropout,
            decoder_mode=args.decoder_mode,
            logvar_min=args.logvar_min,
            logvar_max=args.logvar_max,
            max_grad_norm=args.max_grad_norm,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            scheduler_type=args.scheduler_type,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_t_max=args.scheduler_t_max,
            max_train_windows=args.max_train_windows,
            max_val_windows=args.max_val_windows,
            overfit_windows=args.overfit_windows,
            shuffle_train_windows=args.shuffle_train_windows,
            shuffle_val_windows=args.shuffle_val_windows,
            window_subset_seed=args.window_subset_seed,
            sample_export_count=args.sample_export_count,
            sample_seed=args.sample_seed,
            sample_position_smoothing_kernel=args.sample_position_smoothing_kernel,
            normalize_pose=not args.no_normalize_pose,
        ),
        train_window_index_csv=args.train_window_index_csv,
        val_window_index_csv=args.val_window_index_csv,
    )
    print(summary)


if __name__ == "__main__":
    main()
