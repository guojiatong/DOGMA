#!/usr/bin/env python3
"""
Train a temporal VAE directly on raw 20Hz IMU feature windows.
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

from models.pose_generative import TemporalPoseVAE
from raw_imu_ablation import (
    RAW_IMU_INPUT_DIM,
    compute_raw_imu_component_mse,
    compute_raw_imu_recon_metrics,
    flatten_feature_window,
    normalize_feature_tensor,
    denormalize_feature_tensor,
)
from train_imu_masked_recon import append_jsonl, build_scheduler, resolve_device, set_global_seed, write_json
from train_temporal_vae import (
    PoseWindowRecord,
    read_pose_window_records_from_csv,
    select_pose_window_records,
    write_pose_window_subset_csv,
)


@dataclass(frozen=True)
class RawImuTemporalVaeTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_frames: int = 160
    hidden_dim: int = 128
    latent_dim: int = 64
    beta: float = 1e-3
    rot_loss_weight: float = 1.0
    gyr_loss_weight: float = 0.25
    freeacc_loss_weight: float = 0.1
    interp_loss_weight: float = 0.05
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
    normalize_feature: bool = True


class RawImuWindowDataset(Dataset):
    def __init__(self, *, window_records: list[PoseWindowRecord], window_frames: int) -> None:
        self.window_records = list(window_records)
        self.window_frames = int(window_frames)
        self._feature_cache: dict[Path, dict[str, np.ndarray]] = {}
        if not self.window_records:
            raise ValueError("No raw IMU VAE windows available")

    @classmethod
    def from_window_index_csv(
        cls,
        *,
        window_index_csv: Path,
        max_windows: int = 0,
        shuffle: bool = False,
        subset_seed: int = 0,
    ) -> "RawImuWindowDataset":
        records = read_pose_window_records_from_csv(window_index_csv)
        records = select_pose_window_records(
            records,
            max_windows=max_windows,
            shuffle=shuffle,
            subset_seed=subset_seed,
        )
        if not records:
            raise ValueError("No raw IMU VAE windows available")
        return cls(window_records=records, window_frames=records[0].valid_frames)

    def _load_feature_payload(self, feature_path: Path) -> dict[str, np.ndarray]:
        if feature_path not in self._feature_cache:
            with np.load(feature_path, allow_pickle=False) as payload:
                self._feature_cache[feature_path] = {key: payload[key] for key in payload.files}
        return self._feature_cache[feature_path]

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.window_records[index]
        payload = self._load_feature_payload(record.feature_path)
        packet_counter = payload["packet_counter"].astype(np.int64)
        feature = payload["feature"].astype(np.float32)
        end_frame = int(record.start_frame + min(self.window_frames, record.valid_frames))
        feature_window = feature[record.start_frame:end_frame]
        packet_window = packet_counter[record.start_frame:end_frame]
        participant = record.feature_path.parent.name
        segment_id = record.feature_path.stem.removeprefix("segment_")
        return {
            "feature": torch.from_numpy(flatten_feature_window(feature_window)),
            "meta": {
                "participant": participant,
                "segment_id": segment_id,
                "pose_path": str(record.pose_path),
                "feature_path": str(record.feature_path),
                "start_frame_20hz": int(record.start_frame),
                "valid_frames": int(feature_window.shape[0]),
                "packet_start_20hz": int(packet_window[0]) if packet_window.size > 0 else -1,
                "packet_end_20hz": int(packet_window[-1]) if packet_window.size > 0 else -1,
            },
        }


def dataset_from_records(records: list[PoseWindowRecord]) -> RawImuWindowDataset:
    if not records:
        raise ValueError("No raw IMU VAE windows available")
    return RawImuWindowDataset(window_records=records, window_frames=records[0].valid_frames)


def compute_feature_normalization_stats(dataset: RawImuWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack([dataset[index]["feature"].numpy() for index in range(len(dataset))], axis=0)
    mean = stacked.reshape(-1, stacked.shape[-1]).mean(axis=0).astype(np.float32)
    std = stacked.reshape(-1, stacked.shape[-1]).std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def compute_raw_imu_vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float,
    rot_loss_weight: float,
    gyr_loss_weight: float,
    freeacc_loss_weight: float,
    interp_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    component_mse = compute_raw_imu_component_mse(reconstruction, target)
    weighted_recon_loss = (
        float(rot_loss_weight) * component_mse["rot_mse"]
        + float(gyr_loss_weight) * component_mse["gyr_mse"]
        + float(freeacc_loss_weight) * component_mse["freeacc_mse"]
        + float(interp_loss_weight) * component_mse["interp_mse"]
    )
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = weighted_recon_loss + float(beta) * kl
    return loss, {
        "recon_loss": float(loss.item()),
        "weighted_recon_loss": float(weighted_recon_loss.item()),
        "feature_mse": float(component_mse["feature_mse"].item()),
        "rot_mse": float(component_mse["rot_mse"].item()),
        "gyr_mse": float(component_mse["gyr_mse"].item()),
        "freeacc_mse": float(component_mse["freeacc_mse"].item()),
        "interp_mse": float(component_mse["interp_mse"].item()),
        "kl_loss": float(kl.item()),
    }


def merge_metric_sums(total: dict[str, float], update: dict[str, float], *, weight: float) -> None:
    total["count"] = total.get("count", 0.0) + float(weight)
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value) * float(weight)


def finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float]:
    count = metric_sums.get("count", 0.0)
    if count <= 0.0:
        return {
            "recon_loss": 0.0,
            "weighted_recon_loss": 0.0,
            "feature_mse": 0.0,
            "feature_rmse": 0.0,
            "rot_mse": 0.0,
            "rot_rmse": 0.0,
            "gyr_mse": 0.0,
            "gyr_rmse": 0.0,
            "freeacc_mse": 0.0,
            "freeacc_rmse": 0.0,
            "interp_mse": 0.0,
            "interp_rmse": 0.0,
            "kl_loss": 0.0,
        }
    return {key: float(value / count) for key, value in metric_sums.items() if key != "count"}


def all_tensors_finite(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def model_parameters_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def run_train_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    rot_loss_weight: float,
    gyr_loss_weight: float,
    freeacc_loss_weight: float,
    interp_loss_weight: float,
    normalize_feature_flag: bool,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    skipped_nonfinite_batches = 0.0
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        feature = batch["feature"]
        target = normalize_feature_tensor(feature, feature_mean, feature_std) if normalize_feature_flag else feature
        reconstruction, mu, logvar = model(target)
        if not all_tensors_finite(reconstruction, mu, logvar):
            skipped_nonfinite_batches += 1.0
            continue
        reconstruction_raw = (
            denormalize_feature_tensor(reconstruction, feature_mean, feature_std) if normalize_feature_flag else reconstruction
        )
        if not all_tensors_finite(reconstruction_raw):
            skipped_nonfinite_batches += 1.0
            continue
        loss, loss_metrics = compute_raw_imu_vae_loss(
            reconstruction=reconstruction,
            target=target,
            mu=mu,
            logvar=logvar,
            beta=beta,
            rot_loss_weight=rot_loss_weight,
            gyr_loss_weight=gyr_loss_weight,
            freeacc_loss_weight=freeacc_loss_weight,
            interp_loss_weight=interp_loss_weight,
        )
        if not torch.isfinite(loss):
            skipped_nonfinite_batches += 1.0
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
            raise RuntimeError("Raw IMU temporal VAE parameters became non-finite after optimizer.step()")
        batch_metrics = loss_metrics | compute_raw_imu_recon_metrics(reconstruction_raw.detach(), feature.detach())
        merge_metric_sums(metric_sums, batch_metrics, weight=feature.shape[0])
    metrics = finalize_metric_sums(metric_sums)
    metrics["skipped_nonfinite_batches"] = float(skipped_nonfinite_batches)
    return metrics


@torch.no_grad()
def evaluate_raw_imu_temporal_vae(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    beta: float,
    rot_loss_weight: float,
    gyr_loss_weight: float,
    freeacc_loss_weight: float,
    interp_loss_weight: float,
    normalize_feature_flag: bool,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        feature = batch["feature"]
        target = normalize_feature_tensor(feature, feature_mean, feature_std) if normalize_feature_flag else feature
        reconstruction, mu, logvar = model(target)
        if not all_tensors_finite(reconstruction, mu, logvar):
            raise RuntimeError("Raw IMU temporal VAE produced non-finite outputs during evaluation")
        reconstruction_raw = (
            denormalize_feature_tensor(reconstruction, feature_mean, feature_std) if normalize_feature_flag else reconstruction
        )
        if not all_tensors_finite(reconstruction_raw):
            raise RuntimeError("Raw IMU temporal VAE produced non-finite reconstruction during evaluation")
        _, loss_metrics = compute_raw_imu_vae_loss(
            reconstruction=reconstruction,
            target=target,
            mu=mu,
            logvar=logvar,
            beta=beta,
            rot_loss_weight=rot_loss_weight,
            gyr_loss_weight=gyr_loss_weight,
            freeacc_loss_weight=freeacc_loss_weight,
            interp_loss_weight=interp_loss_weight,
        )
        batch_metrics = loss_metrics | compute_raw_imu_recon_metrics(reconstruction_raw, feature)
        merge_metric_sums(metric_sums, batch_metrics, weight=feature.shape[0])
    return finalize_metric_sums(metric_sums)


@torch.no_grad()
def export_raw_imu_temporal_vae_samples(
    *,
    model: torch.nn.Module,
    dataset: RawImuWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    normalize_feature_flag: bool,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> None:
    if sample_count <= 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(sample_count), len(dataset))
    rng = np.random.default_rng(sample_seed + epoch)
    if count == len(dataset):
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.sort(rng.choice(len(dataset), size=count, replace=False))

    samples = [dataset[int(index)] for index in indices]
    feature = torch.stack([sample["feature"] for sample in samples], dim=0).to(device)
    target = normalize_feature_tensor(feature, feature_mean, feature_std) if normalize_feature_flag else feature
    reconstruction, mu, logvar = model(target)
    if not all_tensors_finite(reconstruction, mu, logvar):
        raise RuntimeError("Raw IMU temporal VAE produced non-finite outputs during sample export")
    reconstruction_raw = denormalize_feature_tensor(reconstruction, feature_mean, feature_std) if normalize_feature_flag else reconstruction
    if not all_tensors_finite(reconstruction_raw):
        raise RuntimeError("Raw IMU temporal VAE produced non-finite reconstruction during sample export")
    meta_json = json.dumps([sample["meta"] for sample in samples])
    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        target_feature=feature.detach().cpu().numpy().astype(np.float32),
        reconstruction_feature=reconstruction_raw.detach().cpu().numpy().astype(np.float32),
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
    config: RawImuTemporalVaeTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
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
            "feature_mean": feature_mean.astype(np.float32),
            "feature_std": feature_std.astype(np.float32),
        },
        path,
    )


def load_raw_imu_temporal_vae_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[TemporalPoseVAE, RawImuTemporalVaeTrainingConfig, torch.Tensor, torch.Tensor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = RawImuTemporalVaeTrainingConfig(**checkpoint["config"])
    model = TemporalPoseVAE(
        input_dim=RAW_IMU_INPUT_DIM,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        window_frames=config.window_frames,
        dropout=config.dropout,
        decoder_mode=config.decoder_mode,
        logvar_min=config.logvar_min,
        logvar_max=config.logvar_max,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_mean = torch.as_tensor(checkpoint["feature_mean"], dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(checkpoint["feature_std"], dtype=torch.float32, device=device)
    return model, config, feature_mean, feature_std, checkpoint


def run_raw_imu_temporal_vae_training(
    *,
    output_dir: Path,
    config: RawImuTemporalVaeTrainingConfig,
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

    train_dataset = dataset_from_records(train_selected_records)
    val_dataset = dataset_from_records(val_selected_records)
    write_pose_window_subset_csv(output_dir / "train_window_subset.csv", train_selected_records)
    write_pose_window_subset_csv(output_dir / "val_window_subset.csv", val_selected_records)

    feature_mean_np, feature_std_np = compute_feature_normalization_stats(train_dataset)
    feature_mean = torch.from_numpy(feature_mean_np).to(device=device, dtype=torch.float32)
    feature_std = torch.from_numpy(feature_std_np).to(device=device, dtype=torch.float32)
    np.savez_compressed(
        output_dir / "normalization_stats.npz",
        feature_mean=feature_mean_np.astype(np.float32),
        feature_std=feature_std_np.astype(np.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = TemporalPoseVAE(
        input_dim=RAW_IMU_INPUT_DIM,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        window_frames=config.window_frames,
        dropout=config.dropout,
        decoder_mode=config.decoder_mode,
        logvar_min=config.logvar_min,
        logvar_max=config.logvar_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    args_payload = asdict(config) | {
        "train_window_index_csv": str(train_window_index_csv),
        "val_window_index_csv": str(val_window_index_csv),
        "device_resolved": str(device),
        "input_dim": int(RAW_IMU_INPUT_DIM),
    }
    write_json(output_dir / "args.json", args_payload)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_metric = float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            beta=config.beta,
            rot_loss_weight=config.rot_loss_weight,
            gyr_loss_weight=config.gyr_loss_weight,
            freeacc_loss_weight=config.freeacc_loss_weight,
            interp_loss_weight=config.interp_loss_weight,
            normalize_feature_flag=config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            max_grad_norm=config.max_grad_norm,
        )
        val_metrics = evaluate_raw_imu_temporal_vae(
            model=model,
            dataloader=val_loader,
            device=device,
            beta=config.beta,
            rot_loss_weight=config.rot_loss_weight,
            gyr_loss_weight=config.gyr_loss_weight,
            freeacc_loss_weight=config.freeacc_loss_weight,
            interp_loss_weight=config.interp_loss_weight,
            normalize_feature_flag=config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        append_jsonl(metrics_path, record)
        export_raw_imu_temporal_vae_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_reconstructions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            normalize_feature_flag=config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
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
            feature_mean=feature_mean_np,
            feature_std=feature_std_np,
        )
        if record["val_feature_rmse"] <= best_metric:
            best_metric = record["val_feature_rmse"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                feature_mean=feature_mean_np,
                feature_std=feature_std_np,
            )
    return {
        "best_epoch": int(best_epoch),
        "best_val_feature_rmse": float(best_metric),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal VAE directly on raw IMU windows")
    parser.add_argument("--train-window-index-csv", type=Path, required=True)
    parser.add_argument("--val-window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-frames", type=int, default=160)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--rot-loss-weight", type=float, default=1.0)
    parser.add_argument("--gyr-loss-weight", type=float, default=0.25)
    parser.add_argument("--freeacc-loss-weight", type=float, default=0.1)
    parser.add_argument("--interp-loss-weight", type=float, default=0.05)
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
    parser.add_argument("--disable-normalize-feature", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_raw_imu_temporal_vae_training(
        output_dir=args.output_dir,
        config=RawImuTemporalVaeTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            window_frames=args.window_frames,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta=args.beta,
            rot_loss_weight=args.rot_loss_weight,
            gyr_loss_weight=args.gyr_loss_weight,
            freeacc_loss_weight=args.freeacc_loss_weight,
            interp_loss_weight=args.interp_loss_weight,
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
            normalize_feature=not args.disable_normalize_feature,
        ),
        train_window_index_csv=args.train_window_index_csv,
        val_window_index_csv=args.val_window_index_csv,
    )
    print(summary)


if __name__ == "__main__":
    main()
