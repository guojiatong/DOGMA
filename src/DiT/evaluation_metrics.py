from __future__ import annotations

from typing import Any

import torch

from train_imu_masked_recon import rot6d_to_rotation_matrix
from train_temporal_vae import POSE_POSITION_DIM, ROOT_HEADING_DIM, ROOT_TRANSLATION_DIM


def future_slice(past_frames: int, future_window_frames: int, frame_count: int) -> slice:
    start = min(max(int(past_frames), 0), int(frame_count))
    end = min(start + max(int(future_window_frames), 0), int(frame_count))
    return slice(start, end)


def has_root_translation(pose_dim: int) -> bool:
    return int(pose_dim) >= POSE_POSITION_DIM + ROOT_HEADING_DIM + ROOT_TRANSLATION_DIM


def compute_future_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    if prediction.ndim != 3 or target.ndim != 3:
        raise ValueError(f"Expected [B,T,C] tensors, got {prediction.shape} and {target.shape}")
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if prediction.shape[1] == 0:
        raise ValueError("Future metric tensors must contain at least one frame")

    pred_positions = prediction[..., :POSE_POSITION_DIM].reshape(prediction.shape[0], prediction.shape[1], 10, 3)
    target_positions = target[..., :POSE_POSITION_DIM].reshape(target.shape[0], target.shape[1], 10, 3)
    mpjpe = torch.linalg.norm(pred_positions - target_positions, dim=-1).mean()

    heading_slice = slice(POSE_POSITION_DIM, POSE_POSITION_DIM + ROOT_HEADING_DIM)
    pred_heading = prediction[..., heading_slice]
    target_heading = target[..., heading_slice]
    pred_forward = rot6d_to_rotation_matrix(pred_heading)[..., :2, 0]
    target_forward = rot6d_to_rotation_matrix(target_heading)[..., :2, 0]
    pred_forward = pred_forward / torch.clamp(torch.linalg.norm(pred_forward, dim=-1, keepdim=True), min=1e-8)
    target_forward = target_forward / torch.clamp(torch.linalg.norm(target_forward, dim=-1, keepdim=True), min=1e-8)
    heading_dot = torch.sum(pred_forward * target_forward, dim=-1).clamp(-1.0, 1.0)
    heading_error_deg = torch.rad2deg(torch.acos(heading_dot)).mean()

    ade: torch.Tensor | None = None
    fde: torch.Tensor | None = None
    if has_root_translation(prediction.shape[-1]):
        translation_slice = slice(POSE_POSITION_DIM + ROOT_HEADING_DIM, POSE_POSITION_DIM + ROOT_HEADING_DIM + ROOT_TRANSLATION_DIM)
        translation_error = torch.linalg.norm(
            prediction[..., translation_slice] - target[..., translation_slice],
            dim=-1,
        )
        ade = translation_error.mean()
        fde = translation_error[:, -1].mean()

    return {
        "mpjpe": float(mpjpe.item()),
        "heading_error_deg": float(heading_error_deg.item()),
        "ade": None if ade is None else float(ade.item()),
        "fde": None if fde is None else float(fde.item()),
    }


def compute_diversity_at_k(samples: list[torch.Tensor], *, k: int = 10) -> float:
    selected = [sample.detach() for sample in samples[: int(k)]]
    if len(selected) <= 1:
        return 0.0
    stacked = torch.cat(selected, dim=0)
    positions = stacked[..., :POSE_POSITION_DIM].reshape(stacked.shape[0], stacked.shape[1], 10, 3)
    distances: list[torch.Tensor] = []
    for first in range(positions.shape[0] - 1):
        for second in range(first + 1, positions.shape[0]):
            distances.append(torch.linalg.norm(positions[first] - positions[second], dim=-1).mean())
    if not distances:
        return 0.0
    return float(torch.stack(distances).mean().item())


def mean_metric_dict(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {
            "mpjpe": 0.0,
            "heading_error_deg": 0.0,
            "diversity_at_10": 0.0,
            "ade": None,
            "fde": None,
        }
    output: dict[str, float | None] = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        output[key] = None if not values else float(sum(float(value) for value in values) / len(values))
    return output


def _flatten_summary_metrics(
    metrics: dict[str, float | None],
    extra: dict[str, Any],
) -> tuple[dict[str, float | None], dict[str, Any]]:
    summary = dict(extra)
    flattened_metrics = dict(metrics)

    motion_distribution_metrics = summary.pop("motion_distribution_metrics", None)
    if isinstance(motion_distribution_metrics, dict):
        fid_value = motion_distribution_metrics.get("fid_motion_encoder")
        if fid_value is not None:
            flattened_metrics["fid"] = float(fid_value)

    trimmed_metrics = summary.pop("translation_metrics_p95_trimmed", None)
    if isinstance(trimmed_metrics, dict):
        ade_p95 = trimmed_metrics.get("ade")
        fde_p95 = trimmed_metrics.get("fde")
        if ade_p95 is not None:
            flattened_metrics["ade_p95"] = float(ade_p95)
        if fde_p95 is not None:
            flattened_metrics["fde_p95"] = float(fde_p95)

    return flattened_metrics, summary


def build_metrics_summary(
    *,
    rows: list[dict[str, float | None]],
    sample_count: int,
    future_frames: int,
    include_root_translation: bool,
    extra: dict[str, Any],
) -> dict[str, Any]:
    del sample_count, future_frames, include_root_translation
    metrics, summary = _flatten_summary_metrics(mean_metric_dict(rows), extra)
    return {
        **summary,
        "metrics": metrics,
        "window_metrics": rows,
    }
