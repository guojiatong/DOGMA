#!/usr/bin/env python3
"""
Train a conditional temporal VAE:
past 8s raw IMU -> future 4s pseudo-pose.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.pose_generative import ConditionalFuturePoseVAE
from train_imu_masked_recon import append_jsonl, build_scheduler, resolve_device, set_global_seed, write_json
from train_latent_diffusion import CONDITION_DIM, compute_condition_normalization_stats, normalize_condition_tensor
from train_temporal_vae import (
    POSE_POSITION_DIM,
    PoseWindowRecord,
    ROOT_HEADING_DIM,
    WINDOW_SUBSET_COLUMNS,
    _move_tensor_batch,
    apply_position_temporal_filter,
    compute_heading_forward_loss,
    compute_heading_velocity_acceleration_losses,
    compute_pose_recon_metrics,
    compute_position_recon_mse,
    compute_position_velocity_loss,
    compute_temporal_smoothness_loss,
    denormalize_pose_tensor,
    flatten_pose_window,
    normalize_pose_tensor,
    read_pose_window_records_from_csv,
    rebase_root_heading_6d,
    resolve_position_smoothing_kernel,
    select_pose_window_records,
    write_pose_window_subset_csv,
)


@dataclass(frozen=True)
class ConditionalTemporalVaeTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    past_frames: int = 160
    future_window_frames: int = 80
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
    future_heading_anchor: str = "future_start"
    use_condition_skip_decoder: bool = False
    use_split_pose_heads: bool = False
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


class ConditionalFuturePoseDataset(Dataset):
    def __init__(
        self,
        *,
        window_records: list[PoseWindowRecord],
        past_frames: int,
        future_window_frames: int,
        future_heading_anchor: str = "future_start",
    ) -> None:
        self.window_records = [
            record
            for record in window_records
            if record.start_frame >= int(past_frames) and record.valid_frames >= int(future_window_frames)
        ]
        self.past_frames = int(past_frames)
        self.future_window_frames = int(future_window_frames)
        self.future_heading_anchor = str(future_heading_anchor)
        if self.future_heading_anchor not in {"future_start", "context_end"}:
            raise ValueError(
                f"future_heading_anchor must be 'future_start' or 'context_end', got {self.future_heading_anchor}"
            )
        self._feature_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._pose_cache: dict[Path, dict[str, np.ndarray]] = {}
        if not self.window_records:
            raise ValueError("No conditional temporal VAE windows available")

    def _load_feature_payload(self, feature_path: Path) -> dict[str, np.ndarray]:
        if feature_path not in self._feature_cache:
            with np.load(feature_path, allow_pickle=False) as payload:
                self._feature_cache[feature_path] = {key: payload[key] for key in payload.files}
        return self._feature_cache[feature_path]

    def _load_pose_payload(self, pose_path: Path) -> dict[str, np.ndarray]:
        if pose_path not in self._pose_cache:
            with np.load(pose_path, allow_pickle=False) as payload:
                self._pose_cache[pose_path] = {key: payload[key] for key in payload.files}
        return self._pose_cache[pose_path]

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.window_records[index]
        feature_payload = self._load_feature_payload(record.feature_path)
        pose_payload = self._load_pose_payload(record.pose_path)

        past_start = int(record.start_frame - self.past_frames)
        past_end = int(record.start_frame)
        future_end = int(record.start_frame + self.future_window_frames)

        condition = (
            feature_payload["feature"][past_start:past_end].reshape(self.past_frames, CONDITION_DIM).astype(np.float32)
        )
        relative_positions = pose_payload["relative_positions"][record.start_frame:future_end].astype(np.float32)
        future_heading_source = pose_payload["root_heading_6d"][record.start_frame:future_end].astype(np.float32)
        if self.future_heading_anchor == "context_end":
            root_heading_6d = rebase_root_heading_6d(
                future_heading_source,
                reference_root_heading_6d=pose_payload["root_heading_6d"][past_end - 1 : past_end].astype(np.float32),
            )
        else:
            root_heading_6d = rebase_root_heading_6d(future_heading_source)
        target_pose = flatten_pose_window(relative_positions, root_heading_6d).astype(np.float32)
        packet_counter = pose_payload["packet_counter"][record.start_frame:future_end].astype(np.int64)

        return {
            "condition": torch.from_numpy(condition),
            "target_pose": torch.from_numpy(target_pose),
            "meta": {
                "participant": record.pose_path.parent.name,
                "segment_id": record.pose_path.stem.removeprefix("segment_"),
                "pose_path": str(record.pose_path),
                "feature_path": str(record.feature_path),
                "start_frame_20hz": int(record.start_frame),
                "past_start_frame_20hz": int(past_start),
                "past_end_frame_20hz": int(past_end - 1),
                "future_end_frame_20hz": int(future_end - 1),
                "valid_frames": int(self.future_window_frames),
                "future_heading_anchor": self.future_heading_anchor,
                "packet_start_20hz": int(packet_counter[0]),
                "packet_end_20hz": int(packet_counter[-1]),
            },
        }


def compute_pose_target_normalization_stats(dataset: ConditionalFuturePoseDataset) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([dataset[index]["target_pose"].numpy() for index in range(len(dataset))], axis=0)
    mean = stacked.reshape(-1, stacked.shape[-1]).mean(axis=0).astype(np.float32)
    std = stacked.reshape(-1, stacked.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def compute_gaussian_kl_to_prior(
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    posterior_var = torch.exp(posterior_logvar)
    prior_var = torch.exp(prior_logvar)
    term = (
        prior_logvar
        - posterior_logvar
        + (posterior_var + (posterior_mu - prior_mu).pow(2)) / torch.clamp(prior_var, min=1e-8)
        - 1.0
    )
    return 0.5 * torch.mean(term)


def compute_conditional_temporal_vae_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    prediction_raw: torch.Tensor,
    target_raw: torch.Tensor,
    posterior_mu: torch.Tensor,
    posterior_logvar: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_logvar: torch.Tensor,
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
        prediction[..., :POSE_POSITION_DIM],
        target[..., :POSE_POSITION_DIM],
        distal_joint_scale=distal_joint_scale,
    )
    heading_recon_mse = torch.mean((prediction[..., POSE_POSITION_DIM:] - target[..., POSE_POSITION_DIM:]) ** 2)
    recon_mse = torch.mean((prediction - target) ** 2)
    position_velocity_loss = compute_position_velocity_loss(
        prediction_raw[..., :POSE_POSITION_DIM],
        target_raw[..., :POSE_POSITION_DIM],
        distal_joint_scale=distal_joint_scale,
    )
    heading_forward_loss = compute_heading_forward_loss(
        prediction_raw[..., POSE_POSITION_DIM:],
        target_raw[..., POSE_POSITION_DIM:],
    )
    temporal_smoothness_loss = compute_temporal_smoothness_loss(
        prediction_raw[..., :POSE_POSITION_DIM],
        target_raw[..., :POSE_POSITION_DIM],
    )
    heading_velocity_loss, heading_acceleration_loss = compute_heading_velocity_acceleration_losses(
        prediction_raw[..., POSE_POSITION_DIM:],
        target_raw[..., POSE_POSITION_DIM:],
    )
    kl_loss = compute_gaussian_kl_to_prior(
        posterior_mu=posterior_mu,
        posterior_logvar=posterior_logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
    )
    weighted_recon_loss = (
        float(position_loss_weight) * position_recon_mse
        + float(position_velocity_loss_weight) * position_velocity_loss
        + float(heading_loss_weight) * heading_recon_mse
        + float(heading_forward_loss_weight) * heading_forward_loss
        + float(temporal_smoothness_loss_weight) * temporal_smoothness_loss
        + float(heading_velocity_loss_weight) * heading_velocity_loss
        + float(heading_acceleration_loss_weight) * heading_acceleration_loss
    )
    loss = weighted_recon_loss + float(beta) * kl_loss
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
        "kl_loss": float(kl_loss.item()),
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
    return {key: float(value / count) for key, value in metric_sums.items() if key != "count"}


def predict_conditional_future_pose(
    *,
    model: ConditionalFuturePoseVAE,
    condition: torch.Tensor,
    normalize_pose: bool,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    position_smoothing_kernel: tuple[float, ...] | None = None,
) -> torch.Tensor:
    prediction = model.predict(condition, sample=False)
    prediction_raw = denormalize_pose_tensor(prediction, pose_mean, pose_std) if normalize_pose else prediction
    return apply_position_temporal_filter(prediction_raw, kernel_weights=position_smoothing_kernel)


def run_train_epoch(
    *,
    model: ConditionalFuturePoseVAE,
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
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_pose_raw = batch["target_pose"]
        target_pose = normalize_pose_tensor(target_pose_raw, pose_mean, pose_std) if normalize_pose else target_pose_raw
        optimizer.zero_grad(set_to_none=True)
        reconstruction, posterior_mu, posterior_logvar, prior_mu, prior_logvar = model(condition, target_pose)
        reconstruction_raw = denormalize_pose_tensor(reconstruction, pose_mean, pose_std) if normalize_pose else reconstruction
        loss, loss_metrics = compute_conditional_temporal_vae_loss(
            prediction=reconstruction,
            target=target_pose,
            prediction_raw=reconstruction_raw,
            target_raw=target_pose_raw,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
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
        loss.backward()
        optimizer.step()
        batch_metrics = loss_metrics | compute_pose_recon_metrics(
            prediction=reconstruction_raw.detach(),
            target=target_pose_raw.detach(),
        )
        merge_metric_sums(metric_sums, batch_metrics, weight=target_pose_raw.shape[0])
    return finalize_metric_sums(metric_sums)


@torch.no_grad()
def evaluate_conditional_temporal_vae(
    *,
    model: ConditionalFuturePoseVAE,
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
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    position_smoothing_kernel: tuple[float, ...] | None = None,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_pose_raw = batch["target_pose"]
        target_pose = normalize_pose_tensor(target_pose_raw, pose_mean, pose_std) if normalize_pose else target_pose_raw
        reconstruction, posterior_mu, posterior_logvar, prior_mu, prior_logvar = model(condition, target_pose)
        prediction_raw = predict_conditional_future_pose(
            model=model,
            condition=condition,
            normalize_pose=normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            position_smoothing_kernel=position_smoothing_kernel,
        )
        _, loss_metrics = compute_conditional_temporal_vae_loss(
            prediction=reconstruction,
            target=target_pose,
            prediction_raw=prediction_raw,
            target_raw=target_pose_raw,
            posterior_mu=posterior_mu,
            posterior_logvar=posterior_logvar,
            prior_mu=prior_mu,
            prior_logvar=prior_logvar,
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
        batch_metrics = loss_metrics | compute_pose_recon_metrics(prediction=prediction_raw, target=target_pose_raw)
        merge_metric_sums(metric_sums, batch_metrics, weight=target_pose_raw.shape[0])
    return finalize_metric_sums(metric_sums)


@torch.no_grad()
def export_conditional_temporal_vae_samples(
    *,
    model: ConditionalFuturePoseVAE,
    dataset: ConditionalFuturePoseDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    normalize_pose: bool,
    pose_mean: torch.Tensor,
    pose_std: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    position_smoothing_kernel: tuple[float, ...] | None,
) -> None:
    if sample_count <= 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(sample_count), len(dataset))
    rng = np.random.default_rng(sample_seed + epoch)
    indices = np.arange(len(dataset), dtype=np.int64) if count == len(dataset) else np.sort(
        rng.choice(len(dataset), size=count, replace=False)
    )
    samples = [dataset[int(index)] for index in indices]
    condition_raw = torch.stack([sample["condition"] for sample in samples], dim=0).to(device)
    target_pose = torch.stack([sample["target_pose"] for sample in samples], dim=0).to(device)
    condition = normalize_condition_tensor(condition_raw, condition_mean, condition_std)
    prediction = predict_conditional_future_pose(
        model=model,
        condition=condition,
        normalize_pose=normalize_pose,
        pose_mean=pose_mean,
        pose_std=pose_std,
        position_smoothing_kernel=position_smoothing_kernel,
    )
    meta_json = json.dumps([sample["meta"] for sample in samples])
    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        condition=condition_raw.detach().cpu().numpy().astype(np.float32),
        target_pose=target_pose.detach().cpu().numpy().astype(np.float32),
        prediction=prediction.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: ConditionalFuturePoseVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: ConditionalTemporalVaeTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    pose_mean: np.ndarray,
    pose_std: np.ndarray,
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
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
            "condition_mean": condition_mean.astype(np.float32),
            "condition_std": condition_std.astype(np.float32),
        },
        path,
    )


def run_conditional_temporal_vae_training(
    *,
    output_dir: Path,
    config: ConditionalTemporalVaeTrainingConfig,
    train_window_index_csv: Path,
    val_window_index_csv: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    train_records = [
        record
        for record in read_pose_window_records_from_csv(train_window_index_csv)
        if record.start_frame >= config.past_frames and record.valid_frames >= config.future_window_frames
    ]
    val_records = [
        record
        for record in read_pose_window_records_from_csv(val_window_index_csv)
        if record.start_frame >= config.past_frames and record.valid_frames >= config.future_window_frames
    ]
    if config.overfit_windows > 0:
        overfit_records = select_pose_window_records(
            train_records,
            max_windows=config.overfit_windows,
            shuffle=config.shuffle_train_windows,
            subset_seed=config.window_subset_seed,
        )
        train_selected_records = list(overfit_records)
        val_selected_records = list(overfit_records)
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

    write_pose_window_subset_csv(output_dir / "train_window_subset.csv", train_selected_records)
    write_pose_window_subset_csv(output_dir / "val_window_subset.csv", val_selected_records)

    train_dataset = ConditionalFuturePoseDataset(
        window_records=train_selected_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        future_heading_anchor=config.future_heading_anchor,
    )
    val_dataset = ConditionalFuturePoseDataset(
        window_records=val_selected_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        future_heading_anchor=config.future_heading_anchor,
    )

    pose_mean_np, pose_std_np = compute_pose_target_normalization_stats(train_dataset)
    condition_mean_np, condition_std_np = compute_condition_normalization_stats(train_dataset)
    pose_mean = torch.from_numpy(pose_mean_np).to(device=device, dtype=torch.float32)
    pose_std = torch.from_numpy(pose_std_np).to(device=device, dtype=torch.float32)
    condition_mean = torch.from_numpy(condition_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    condition_std = torch.from_numpy(condition_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    np.savez_compressed(
        output_dir / "normalization_stats.npz",
        pose_mean=pose_mean_np.astype(np.float32),
        pose_std=pose_std_np.astype(np.float32),
        condition_mean=condition_mean_np.astype(np.float32),
        condition_std=condition_std_np.astype(np.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = ConditionalFuturePoseVAE(
        pose_dim=POSE_POSITION_DIM + ROOT_HEADING_DIM,
        condition_dim=CONDITION_DIM,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        future_frames=config.future_window_frames,
        dropout=config.dropout,
        decoder_mode=config.decoder_mode,
        use_condition_skip_decoder=config.use_condition_skip_decoder,
        use_split_pose_heads=config.use_split_pose_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
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
            condition_mean=condition_mean,
            condition_std=condition_std,
        )
        val_metrics = evaluate_conditional_temporal_vae(
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
            condition_mean=condition_mean,
            condition_std=condition_std,
            position_smoothing_kernel=sample_position_smoothing_kernel,
        )
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        append_jsonl(metrics_path, record)
        export_conditional_temporal_vae_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_predictions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            condition_mean=condition_mean,
            condition_std=condition_std,
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
            condition_mean=condition_mean_np,
            condition_std=condition_std_np,
        )
        if record["val_recon_mpjpe"] <= best_metric:
            best_metric = record["val_recon_mpjpe"]
            best_epoch = epoch
            best_record = dict(record)
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
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
            )
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
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
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
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
            )

    return {
        "best_epoch": int(best_epoch),
        "best_val_recon_mpjpe": float(best_metric),
        "best_val_root_heading_error_deg": float(0.0 if best_record is None else best_record["val_root_heading_error_deg"]),
        "best_val_heading_rmse": float(0.0 if best_record is None else best_record["val_heading_rmse"]),
        "best_heading_epoch": int(best_heading_epoch),
        "best_heading_val_root_heading_error_deg": float(
            0.0 if best_heading_record is None else best_heading_record["val_root_heading_error_deg"]
        ),
        "output_dir": str(output_dir),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional temporal VAE on 20Hz pseudo-pose windows")
    parser.add_argument("--train-window-index-csv", type=Path, required=True)
    parser.add_argument("--val-window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--past-frames", type=int, default=160)
    parser.add_argument("--future-window-frames", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--position-loss-weight", type=float, default=30.0 / 36.0)
    parser.add_argument("--position-velocity-loss-weight", type=float, default=0.1)
    parser.add_argument("--heading-loss-weight", type=float, default=0.5)
    parser.add_argument("--heading-forward-loss-weight", type=float, default=0.5)
    parser.add_argument("--distal-joint-scale", type=float, default=2.5)
    parser.add_argument("--temporal-smoothness-loss-weight", type=float, default=0.0)
    parser.add_argument("--heading-velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--heading-acceleration-loss-weight", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--decoder-mode", type=str, default="transpose_conv", choices=("transpose_conv", "upsample_conv"))
    parser.add_argument("--future-heading-anchor", type=str, default="context_end", choices=("future_start", "context_end"))
    parser.add_argument("--use-condition-skip-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-split-pose-heads", action=argparse.BooleanOptionalAction, default=True)
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
    summary = run_conditional_temporal_vae_training(
        output_dir=args.output_dir,
        config=ConditionalTemporalVaeTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            past_frames=args.past_frames,
            future_window_frames=args.future_window_frames,
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
            future_heading_anchor=args.future_heading_anchor,
            use_condition_skip_decoder=args.use_condition_skip_decoder,
            use_split_pose_heads=args.use_split_pose_heads,
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
