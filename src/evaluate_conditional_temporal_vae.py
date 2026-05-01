#!/usr/bin/env python3
"""
Evaluate a conditional temporal VAE checkpoint on held-out windows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from models.pose_generative import ConditionalFuturePoseVAE
from train_imu_masked_recon import write_json
from train_temporal_vae import (
    compute_pose_recon_metrics,
    read_pose_window_records_from_csv,
    resolve_device,
    resolve_position_smoothing_kernel,
)
from train_conditional_temporal_vae import (
    ConditionalFuturePoseDataset,
    ConditionalTemporalVaeTrainingConfig,
    _move_tensor_batch,
    evaluate_conditional_temporal_vae,
    predict_conditional_future_pose,
)


WINDOW_METRIC_COLUMNS = (
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
    "recon_mpjpe",
    "root_heading_error_deg",
    "position_rmse",
    "heading_rmse",
    "jerk_error",
)

PARTICIPANT_METRIC_COLUMNS = (
    "participant",
    "window_count",
    "recon_mpjpe",
    "root_heading_error_deg",
    "position_rmse",
    "heading_rmse",
    "jerk_error",
)


def _meta_item(meta_batch: dict[str, Any], index: int) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in meta_batch.items():
        if isinstance(value, torch.Tensor):
            item = value[index].item()
        elif isinstance(value, list):
            item = value[index]
        else:
            item = value
        if isinstance(item, Path):
            item = str(item)
        meta[key] = item
    return meta


def _build_model_from_config(config: ConditionalTemporalVaeTrainingConfig, device: torch.device) -> ConditionalFuturePoseVAE:
    return ConditionalFuturePoseVAE(
        pose_dim=36,
        condition_dim=130,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        future_frames=config.future_window_frames,
        dropout=config.dropout,
        decoder_mode=config.decoder_mode,
        use_condition_skip_decoder=config.use_condition_skip_decoder,
        use_split_pose_heads=config.use_split_pose_heads,
    ).to(device)


def load_conditional_temporal_vae_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[ConditionalFuturePoseVAE, ConditionalTemporalVaeTrainingConfig, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ConditionalTemporalVaeTrainingConfig(**checkpoint["config"])
    model = _build_model_from_config(config, device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    checkpoint["missing_keys"] = list(load_result.missing_keys)
    checkpoint["unexpected_keys"] = list(load_result.unexpected_keys)
    model.eval()
    pose_mean = torch.as_tensor(checkpoint["pose_mean"], dtype=torch.float32, device=device)
    pose_std = torch.as_tensor(checkpoint["pose_std"], dtype=torch.float32, device=device)
    condition_mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    condition_std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    return model, config, pose_mean, pose_std, condition_mean, condition_std, checkpoint


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_conditional_temporal_vae_checkpoint(
    *,
    checkpoint_path: Path,
    window_index_csv: Path,
    output_dir: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "auto",
    max_windows: int = 0,
    position_smoothing_kernel: str = "tri5",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_resolved = resolve_device(device)
    model, config, pose_mean, pose_std, condition_mean, condition_std, checkpoint = load_conditional_temporal_vae_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)

    dataset = ConditionalFuturePoseDataset(
        window_records=[
            record
            for record in read_pose_window_records_from_csv(window_index_csv)
            if record.start_frame >= config.past_frames and record.valid_frames >= config.future_window_frames
        ],
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        future_heading_anchor=config.future_heading_anchor,
    )
    if max_windows > 0 and len(dataset) > max_windows:
        dataset = ConditionalFuturePoseDataset(
            window_records=dataset.window_records[:max_windows],
            past_frames=config.past_frames,
            future_window_frames=config.future_window_frames,
            future_heading_anchor=config.future_heading_anchor,
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    global_metrics = evaluate_conditional_temporal_vae(
        model=model,
        dataloader=dataloader,
        device=device_resolved,
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
        condition_mean=condition_mean,
        condition_std=condition_std,
        position_smoothing_kernel=smoothing_kernel,
    )

    participant_sums: dict[str, dict[str, float]] = {}
    window_rows: list[dict[str, Any]] = []
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device_resolved)
        prediction = predict_conditional_future_pose(
            model=model,
            condition=(batch["condition"] - condition_mean) / condition_std,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            position_smoothing_kernel=smoothing_kernel,
        )
        meta_batch = batch["meta"]
        for index in range(batch["target_pose"].shape[0]):
            meta = _meta_item(meta_batch, index)
            metrics = compute_pose_recon_metrics(
                prediction=prediction[index : index + 1],
                target=batch["target_pose"][index : index + 1],
            )
            row = {
                "participant": meta["participant"],
                "segment_id": meta["segment_id"],
                "pose_path": meta["pose_path"],
                "feature_path": meta["feature_path"],
                "start_frame_20hz": int(meta["start_frame_20hz"]),
                "valid_frames": int(meta["valid_frames"]),
                "packet_start_20hz": int(meta["packet_start_20hz"]),
                "packet_end_20hz": int(meta["packet_end_20hz"]),
                "recon_mpjpe": float(metrics["recon_mpjpe"]),
                "root_heading_error_deg": float(metrics["root_heading_error_deg"]),
                "position_rmse": float(metrics["position_rmse"]),
                "heading_rmse": float(metrics["heading_rmse"]),
                "jerk_error": float(metrics["jerk_error"]),
            }
            window_rows.append(row)
            participant = str(row["participant"])
            if participant not in participant_sums:
                participant_sums[participant] = {"count": 0.0}
            participant_sums[participant]["count"] += 1.0
            for key in ("recon_mpjpe", "root_heading_error_deg", "position_rmse", "heading_rmse", "jerk_error"):
                participant_sums[participant][key] = participant_sums[participant].get(key, 0.0) + float(row[key])

    participant_rows: list[dict[str, Any]] = []
    for participant in sorted(participant_sums):
        total = participant_sums[participant]
        count = max(total["count"], 1.0)
        participant_rows.append(
            {
                "participant": participant,
                "window_count": int(total["count"]),
                "recon_mpjpe": float(total["recon_mpjpe"] / count),
                "root_heading_error_deg": float(total["root_heading_error_deg"] / count),
                "position_rmse": float(total["position_rmse"] / count),
                "heading_rmse": float(total["heading_rmse"] / count),
                "jerk_error": float(total["jerk_error"] / count),
            }
        )

    _write_csv(output_dir / "window_metrics.csv", WINDOW_METRIC_COLUMNS, window_rows)
    _write_csv(output_dir / "participant_metrics.csv", PARTICIPANT_METRIC_COLUMNS, participant_rows)
    sorted_by_mpjpe = sorted(window_rows, key=lambda row: row["recon_mpjpe"], reverse=True)
    sorted_by_heading = sorted(window_rows, key=lambda row: row["root_heading_error_deg"], reverse=True)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_index_csv": str(window_index_csv),
        "device_resolved": str(device_resolved),
        "position_smoothing_kernel": position_smoothing_kernel,
        "future_heading_anchor": config.future_heading_anchor,
        "window_count": len(window_rows),
        "participant_count": len(participant_rows),
        "global_metrics": global_metrics,
        "top5_worst_recon_mpjpe": sorted_by_mpjpe[:5],
        "top5_worst_root_heading_error_deg": sorted_by_heading[:5],
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a conditional temporal VAE checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_conditional_temporal_vae_checkpoint(
        checkpoint_path=args.checkpoint,
        window_index_csv=args.window_index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_windows=args.max_windows,
        position_smoothing_kernel=args.position_smoothing_kernel,
    )
    print(summary)


if __name__ == "__main__":
    main()
