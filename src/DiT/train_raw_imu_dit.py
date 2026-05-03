#!/usr/bin/env python3
"""
Train a conditional DiT directly in raw-IMU feature space.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

VAE_DIR = Path(__file__).resolve().parents[1]
if str(VAE_DIR) not in sys.path:
    sys.path.insert(0, str(VAE_DIR))

from train_dit import (  # noqa: E402
    CONDITION_DIM,
    DEFAULT_CONFIG_DIR,
    DiTTrainingConfig,
    _accumulate_metric_sums,
    _extract,
    _finalize_metric_sums,
    _move_tensor_batch,
    build_beta_schedule,
    build_dit_model,
    build_sampling_generator,
    compute_condition_normalization_stats,
    normalize_condition_tensor,
)
from train_imu_masked_recon import append_jsonl, build_scheduler, resolve_device, set_global_seed, write_json  # noqa: E402
from raw_imu_ablation import compute_raw_imu_recon_metrics  # noqa: E402
from train_raw_imu_latent_diffusion import RawImuLatentDiffusionWindowDataset  # noqa: E402
from train_temporal_vae import (  # noqa: E402
    DEFAULT_FEATURE_ROOT,
    DEFAULT_POSE_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    apply_data_config_to_args,
    load_pose_window_records,
    select_pose_window_records,
    write_pose_window_subset_csv,
)


DEFAULT_TRAIN_WINDOW_INDEX_CSV = DEFAULT_CONFIG_DIR / "vae_window_index_train.csv"
DEFAULT_VAL_WINDOW_INDEX_CSV = DEFAULT_CONFIG_DIR / "vae_window_index_val.csv"


def compute_target_feature_normalization_stats(
    dataset: RawImuLatentDiffusionWindowDataset,
) -> tuple[np.ndarray, np.ndarray]:
    sums: np.ndarray | None = None
    sums_sq: np.ndarray | None = None
    count = 0
    for index in range(len(dataset)):
        target_feature = dataset[index]["target_feature"].numpy().astype(np.float64)
        flattened = target_feature.reshape(-1, target_feature.shape[-1])
        if sums is None:
            sums = flattened.sum(axis=0)
            sums_sq = np.square(flattened).sum(axis=0)
        else:
            sums += flattened.sum(axis=0)
            sums_sq += np.square(flattened).sum(axis=0)
        count += flattened.shape[0]
    if sums is None or sums_sq is None or count <= 0:
        raise ValueError("No target feature frames available for normalization")
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalize_target_tensor(
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return (target - target_mean) / target_std


def denormalize_target_tensor(
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return target * target_std + target_mean


def build_diffusion_buffers(
    *,
    diffusion_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    betas = build_beta_schedule(diffusion_steps=diffusion_steps, beta_start=beta_start, beta_end=beta_end).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def q_sample(
    *,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (
        _extract(diffusion_buffers["sqrt_alphas_cumprod"], timesteps, x_start.ndim) * x_start
        + _extract(diffusion_buffers["sqrt_one_minus_alphas_cumprod"], timesteps, x_start.ndim) * noise
    )


def predict_x0_from_noise(
    *,
    noisy_target: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_alpha_cumprod = _extract(diffusion_buffers["sqrt_alphas_cumprod"], timesteps, noisy_target.ndim)
    sqrt_one_minus = _extract(diffusion_buffers["sqrt_one_minus_alphas_cumprod"], timesteps, noisy_target.ndim)
    return (noisy_target - sqrt_one_minus * predicted_noise) / torch.clamp(sqrt_alpha_cumprod, min=1e-8)


def load_raw_imu_dit_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    DiTTrainingConfig,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = DiTTrainingConfig(**checkpoint["config"])
    condition_mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    condition_std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    target_mean = torch.as_tensor(checkpoint["target_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    target_std = torch.as_tensor(checkpoint["target_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    target_frames = int(checkpoint["target_frames"])
    target_dim = int(checkpoint["target_dim"])
    model = build_dit_model(
        pose_dim=target_dim,
        condition_frames=config.past_frames,
        condition_hidden_dim=config.condition_hidden_dim,
        pose_frames=target_frames,
        dropout=config.dropout,
        denoiser_num_blocks=config.denoiser_num_blocks,
        fusion_mode=config.fusion_mode,
        dit_num_heads=config.dit_num_heads,
        dit_mlp_ratio=config.dit_mlp_ratio,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion_buffers = build_diffusion_buffers(
        diffusion_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    return model, config, condition_mean, condition_std, target_mean, target_std, diffusion_buffers, checkpoint


def compute_noise_prediction_loss(
    *,
    model: torch.nn.Module,
    condition: torch.Tensor,
    target_feature_normalized: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    target_feature_raw: torch.Tensor | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    past_frames: int = 0,
    future_frames: int = 0,
    track_x0_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    timesteps = torch.randint(
        low=0,
        high=int(diffusion_buffers["betas"].shape[0]),
        size=(target_feature_normalized.shape[0],),
        device=target_feature_normalized.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target_feature_normalized)
    noisy_target = q_sample(
        x_start=target_feature_normalized,
        timesteps=timesteps,
        noise=noise,
        diffusion_buffers=diffusion_buffers,
    )
    predicted_noise = model(noisy_target, condition, timesteps)
    noise_mse = torch.mean((predicted_noise - noise) ** 2)
    metrics = {
        "noise_mse": float(noise_mse.item()),
        "target_std": float(target_feature_normalized.std(unbiased=False).item()),
    }
    if track_x0_metrics:
        if target_feature_raw is None or target_mean is None or target_std is None:
            raise ValueError("x0 metrics require target_feature_raw, target_mean, and target_std")
        x0_prediction = predict_x0_from_noise(
            noisy_target=noisy_target,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
            diffusion_buffers=diffusion_buffers,
        )
        denoised_feature = denormalize_target_tensor(x0_prediction, target_mean, target_std)
        future_start = int(past_frames)
        future_end = int(past_frames + future_frames)
        future_metrics = compute_raw_imu_recon_metrics(
            denoised_feature[:, future_start:future_end],
            target_feature_raw[:, future_start:future_end],
        )
        metrics.update({f"x0_future_{key}": float(value) for key, value in future_metrics.items()})
    return noise_mse, metrics


def run_train_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    past_frames: int,
    future_frames: int,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_feature_raw = batch["target_feature"]
        target_feature = normalize_target_tensor(target_feature_raw, target_mean, target_std)
        loss, metrics = compute_noise_prediction_loss(
            model=model,
            condition=condition,
            target_feature_normalized=target_feature,
            diffusion_buffers=diffusion_buffers,
            target_feature_raw=target_feature_raw,
            target_mean=target_mean,
            target_std=target_std,
            past_frames=past_frames,
            future_frames=future_frames,
            track_x0_metrics=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _accumulate_metric_sums(metric_sums, metrics, target_feature_raw.shape[0])
    return _finalize_metric_sums(metric_sums)


@torch.no_grad()
def evaluate_raw_imu_dit(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    past_frames: int,
    future_frames: int,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_feature_raw = batch["target_feature"]
        target_feature = normalize_target_tensor(target_feature_raw, target_mean, target_std)
        _, metrics = compute_noise_prediction_loss(
            model=model,
            condition=condition,
            target_feature_normalized=target_feature,
            diffusion_buffers=diffusion_buffers,
            target_feature_raw=target_feature_raw,
            target_mean=target_mean,
            target_std=target_std,
            past_frames=past_frames,
            future_frames=future_frames,
            track_x0_metrics=True,
        )
        _accumulate_metric_sums(metric_sums, metrics, target_feature_raw.shape[0])
    return _finalize_metric_sums(metric_sums)


@torch.no_grad()
def sample_feature_diffusion(
    *,
    model: torch.nn.Module,
    condition: torch.Tensor,
    target_shape: tuple[int, int, int],
    diffusion_buffers: dict[str, torch.Tensor],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    current = torch.randn(target_shape, device=condition.device, dtype=torch.float32, generator=generator)
    total_steps = int(diffusion_buffers["betas"].shape[0])
    for step in range(total_steps - 1, -1, -1):
        timesteps = torch.full((target_shape[0],), step, dtype=torch.long, device=condition.device)
        predicted_noise = model(current, condition, timesteps)
        alpha = _extract(diffusion_buffers["alphas"], timesteps, current.ndim)
        alpha_cumprod = _extract(diffusion_buffers["alphas_cumprod"], timesteps, current.ndim)
        beta = _extract(diffusion_buffers["betas"], timesteps, current.ndim)
        if step > 0:
            noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
        else:
            noise = torch.zeros_like(current)
        current = (1.0 / torch.sqrt(alpha)) * (
            current - ((1.0 - alpha) / torch.sqrt(torch.clamp(1.0 - alpha_cumprod, min=1e-8))) * predicted_noise
        ) + torch.sqrt(beta) * noise
    return current


@torch.no_grad()
def export_raw_imu_dit_samples(
    *,
    model: torch.nn.Module,
    dataset: RawImuLatentDiffusionWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
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
    condition = torch.stack([sample["condition"] for sample in samples], dim=0).to(device)
    target_feature_raw = torch.stack([sample["target_feature"] for sample in samples], dim=0).to(device)
    condition = normalize_condition_tensor(condition, condition_mean, condition_std)
    target_feature = normalize_target_tensor(target_feature_raw, target_mean, target_std)
    sampled_feature_normalized = sample_feature_diffusion(
        model=model,
        condition=condition,
        target_shape=tuple(target_feature.shape),
        diffusion_buffers=diffusion_buffers,
        generator=torch.Generator(device=device.type).manual_seed(int(sample_seed + epoch)),
    )
    sampled_feature = denormalize_target_tensor(sampled_feature_normalized, target_mean, target_std)
    meta_json = json.dumps([sample["meta"] for sample in samples])
    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        condition=condition.detach().cpu().numpy().astype(np.float32),
        target_feature=target_feature_raw.detach().cpu().numpy().astype(np.float32),
        sampled_feature=sampled_feature.detach().cpu().numpy().astype(np.float32),
        sampled_feature_normalized=sampled_feature_normalized.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: DiTTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    target_frames: int,
    target_dim: int,
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
            "condition_mean": condition_mean.astype(np.float32),
            "condition_std": condition_std.astype(np.float32),
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
            "target_frames": int(target_frames),
            "target_dim": int(target_dim),
        },
        path,
    )


def run_raw_imu_dit_training(
    *,
    output_dir: Path,
    config: DiTTrainingConfig,
    train_window_index_csv: Path | None,
    val_window_index_csv: Path | None,
    split_manifest_path: Path | None,
    pose_root: Path,
    feature_root: Path,
    train_split: str = "train",
    val_split: str = "val",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    combined_window_frames = int(config.past_frames + config.future_window_frames)
    shared_stride = int(config.stride_frames)
    train_stride = shared_stride if shared_stride > 0 else int(config.train_stride_frames)
    val_stride = shared_stride if shared_stride > 0 else int(config.val_stride_frames)
    train_records = load_pose_window_records(
        window_index_csv=train_window_index_csv,
        split_manifest_path=split_manifest_path,
        split=train_split,
        window_frames=combined_window_frames,
        stride_frames=train_stride,
        pose_root=pose_root,
        feature_root=feature_root,
    )
    val_records = load_pose_window_records(
        window_index_csv=val_window_index_csv,
        split_manifest_path=split_manifest_path,
        split=val_split,
        window_frames=combined_window_frames,
        stride_frames=val_stride,
        pose_root=pose_root,
        feature_root=feature_root,
    )
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

    train_dataset = RawImuLatentDiffusionWindowDataset(
        window_records=train_selected_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
    )
    val_dataset = RawImuLatentDiffusionWindowDataset(
        window_records=val_selected_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
    )
    condition_mean_np, condition_std_np = compute_condition_normalization_stats(train_dataset)
    target_mean_np, target_std_np = compute_target_feature_normalization_stats(train_dataset)
    condition_mean = torch.from_numpy(condition_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    condition_std = torch.from_numpy(condition_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    target_mean = torch.from_numpy(target_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    target_std = torch.from_numpy(target_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    np.savez_compressed(
        output_dir / "condition_normalization_stats.npz",
        condition_mean=condition_mean_np.astype(np.float32),
        condition_std=condition_std_np.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "target_feature_normalization_stats.npz",
        target_mean=target_mean_np.astype(np.float32),
        target_std=target_std_np.astype(np.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    sample_shape = train_dataset[0]["target_feature"].shape
    target_frames = int(sample_shape[0])
    target_dim = int(sample_shape[1])
    model = build_dit_model(
        pose_dim=target_dim,
        condition_frames=config.past_frames,
        condition_hidden_dim=config.condition_hidden_dim,
        pose_frames=target_frames,
        dropout=config.dropout,
        denoiser_num_blocks=config.denoiser_num_blocks,
        fusion_mode=config.fusion_mode,
        dit_num_heads=config.dit_num_heads,
        dit_mlp_ratio=config.dit_mlp_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer=optimizer, config=config)
    diffusion_buffers = build_diffusion_buffers(
        diffusion_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )

    args_payload = asdict(config) | {
        "train_window_index_csv": None if train_window_index_csv is None else str(train_window_index_csv),
        "val_window_index_csv": None if val_window_index_csv is None else str(val_window_index_csv),
        "split_manifest_path": None if split_manifest_path is None else str(split_manifest_path),
        "pose_root": str(pose_root),
        "feature_root": str(feature_root),
        "train_split": train_split,
        "val_split": val_split,
        "train_stride_frames_resolved": int(train_stride),
        "val_stride_frames_resolved": int(val_stride),
        "combined_target_window_frames": target_frames,
        "target_feature_dim": target_dim,
        "target_normalization": "train_split_per_dim",
        "device_resolved": str(device),
    }
    write_json(output_dir / "args.json", args_payload)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_metric = float("inf")
    best_epoch = 0
    best_feature_rmse = float("inf")
    best_feature_epoch = 0
    epoch_progress = tqdm(range(1, config.epochs + 1), desc="epochs")
    for epoch in epoch_progress:
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            condition_mean=condition_mean,
            condition_std=condition_std,
            target_mean=target_mean,
            target_std=target_std,
            diffusion_buffers=diffusion_buffers,
            past_frames=config.past_frames,
            future_frames=config.future_window_frames,
        )
        val_metrics = evaluate_raw_imu_dit(
            model=model,
            dataloader=val_loader,
            device=device,
            condition_mean=condition_mean,
            condition_std=condition_std,
            target_mean=target_mean,
            target_std=target_std,
            diffusion_buffers=diffusion_buffers,
            past_frames=config.past_frames,
            future_frames=config.future_window_frames,
        )
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        epoch_progress.set_postfix(
            train_noise_mse=f"{record['train_noise_mse']:.4f}",
            val_noise_mse=f"{record['val_noise_mse']:.4f}",
        )
        append_jsonl(metrics_path, record)
        export_raw_imu_dit_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_predictions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            condition_mean=condition_mean,
            condition_std=condition_std,
            target_mean=target_mean,
            target_std=target_std,
            diffusion_buffers=diffusion_buffers,
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
            condition_mean=condition_mean_np,
            condition_std=condition_std_np,
            target_mean=target_mean_np,
            target_std=target_std_np,
            target_frames=target_frames,
            target_dim=target_dim,
        )
        if record["val_noise_mse"] <= best_metric:
            best_metric = record["val_noise_mse"]
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
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
                target_mean=target_mean_np,
                target_std=target_std_np,
                target_frames=target_frames,
                target_dim=target_dim,
            )
        if record.get("val_x0_future_feature_rmse", float("inf")) <= best_feature_rmse:
            best_feature_rmse = float(record["val_x0_future_feature_rmse"])
            best_feature_epoch = epoch
            save_checkpoint(
                output_dir / "best_by_feature_rmse.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
                target_mean=target_mean_np,
                target_std=target_std_np,
                target_frames=target_frames,
                target_dim=target_dim,
            )
    return {
        "best_epoch": int(best_epoch),
        "best_val_noise_mse": float(best_metric),
        "best_feature_epoch": int(best_feature_epoch),
        "best_val_x0_future_feature_rmse": float(best_feature_rmse),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct raw-IMU DiT")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--train-window-index-csv", type=Path, default=None)
    parser.add_argument("--val-window-index-csv", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--past-frames", type=int, default=120)
    parser.add_argument("--future-window-frames", type=int, default=40)
    parser.add_argument("--stride-frames", type=int, default=0)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--val-stride-frames", type=int, default=120)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument("--denoiser-num-blocks", type=int, default=4)
    parser.add_argument("--fusion-mode", type=str, default="add", choices=("add", "gated", "concat"))
    parser.add_argument("--dit-num-heads", type=int, default=8)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler-type", type=str, default="cosine", choices=("none", "step", "cosine"))
    parser.add_argument("--scheduler-step-size", type=int, default=1)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=100)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--overfit-windows", type=int, default=0)
    parser.add_argument("--shuffle-train-windows", action="store_true")
    parser.add_argument("--shuffle-val-windows", action="store_true")
    parser.add_argument("--window-subset-seed", type=int, default=0)
    parser.add_argument("--sample-export-count", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    return apply_data_config_to_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    summary = run_raw_imu_dit_training(
        output_dir=args.output_dir,
        config=DiTTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            past_frames=args.past_frames,
            future_window_frames=args.future_window_frames,
            stride_frames=args.stride_frames,
            train_stride_frames=args.train_stride_frames,
            val_stride_frames=args.val_stride_frames,
            condition_hidden_dim=args.condition_hidden_dim,
            denoiser_num_blocks=args.denoiser_num_blocks,
            fusion_mode=args.fusion_mode,
            dit_num_heads=args.dit_num_heads,
            dit_mlp_ratio=args.dit_mlp_ratio,
            diffusion_steps=args.diffusion_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            dropout=args.dropout,
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
        ),
        train_window_index_csv=args.train_window_index_csv,
        val_window_index_csv=args.val_window_index_csv,
        split_manifest_path=args.split_manifest,
        pose_root=args.pose_root,
        feature_root=args.feature_root,
        train_split=args.train_split,
        val_split=args.val_split,
    )
    print(summary)


if __name__ == "__main__":
    main()
