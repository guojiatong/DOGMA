#!/usr/bin/env python3
"""
Train latent DiT on frozen raw-IMU temporal VAE latents.
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

from train_imu_masked_recon import append_jsonl, build_scheduler, resolve_device, set_global_seed, write_json  # noqa: E402
from train_latent_dit import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    LatentDiffusionTrainingConfig,
    _accumulate_metric_sums,
    _finalize_metric_sums,
    build_diffusion_buffers,
    build_latent_diffusion_model,
    build_sampling_generator,
    compute_condition_normalization_stats,
    denormalize_latent_tensor,
    normalize_condition_tensor,
    normalize_latent_tensor,
    predict_x0_from_noise,
    q_sample,
    sample_latent_diffusion,
)
from raw_imu_ablation import compute_raw_imu_recon_metrics  # noqa: E402
from train_raw_imu_latent_diffusion import (  # noqa: E402
    RawImuLatentDiffusionWindowDataset,
    _move_tensor_batch,
    compute_latent_normalization_stats,
    decode_latent_to_feature_tensor,
    encode_feature_to_latent,
)
from train_raw_imu_temporal_vae import RawImuTemporalVaeTrainingConfig, load_raw_imu_temporal_vae_checkpoint  # noqa: E402
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


def compute_noise_prediction_loss(
    *,
    model: torch.nn.Module,
    condition: torch.Tensor,
    target_latent_normalized: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    vae_model: torch.nn.Module | None = None,
    normalize_feature_flag: bool = False,
    feature_mean: torch.Tensor | None = None,
    feature_std: torch.Tensor | None = None,
    latent_mean: torch.Tensor | None = None,
    latent_std: torch.Tensor | None = None,
    target_feature: torch.Tensor | None = None,
    past_frames: int = 0,
    future_frames: int = 0,
    track_x0_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    timesteps = torch.randint(
        low=0,
        high=int(diffusion_buffers["betas"].shape[0]),
        size=(target_latent_normalized.shape[0],),
        device=target_latent_normalized.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target_latent_normalized)
    noisy_latent = q_sample(
        x_start=target_latent_normalized,
        timesteps=timesteps,
        noise=noise,
        diffusion_buffers=diffusion_buffers,
    )
    predicted_noise = model(noisy_latent, condition, timesteps)
    noise_mse = torch.mean((predicted_noise - noise) ** 2)
    metrics = {
        "noise_mse": float(noise_mse.item()),
        "latent_std": float(target_latent_normalized.std(unbiased=False).item()),
    }
    if track_x0_metrics:
        if (
            vae_model is None
            or feature_mean is None
            or feature_std is None
            or latent_mean is None
            or latent_std is None
            or target_feature is None
        ):
            raise ValueError("Decoded x0 feature metrics require VAE, normalization stats, and target_feature")
        x0_prediction = predict_x0_from_noise(
            noisy_latent=noisy_latent,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
            diffusion_buffers=diffusion_buffers,
        )
        decoded_feature = decode_latent_to_feature_tensor(
            vae_model=vae_model,
            latent=x0_prediction,
            normalize_feature_flag=normalize_feature_flag,
            feature_mean=feature_mean,
            feature_std=feature_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        future_start = int(past_frames)
        future_end = int(past_frames + future_frames)
        feature_metrics = compute_raw_imu_recon_metrics(
            decoded_feature[:, future_start:future_end],
            target_feature[:, future_start:future_end],
        )
        metrics.update({f"x0_future_{key}": float(value) for key, value in feature_metrics.items()})
    return noise_mse, metrics


def run_train_epoch(
    *,
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    normalize_feature_flag: bool,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    past_frames: int,
    future_frames: int,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_feature = batch["target_feature"]
        with torch.no_grad():
            target_latent_raw = encode_feature_to_latent(
                vae_model=vae_model,
                target_feature=target_feature,
                normalize_feature_flag=normalize_feature_flag,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )
            target_latent = normalize_latent_tensor(target_latent_raw, latent_mean, latent_std)
        loss, metrics = compute_noise_prediction_loss(
            model=model,
            condition=condition,
            target_latent_normalized=target_latent,
            diffusion_buffers=diffusion_buffers,
            vae_model=vae_model,
            normalize_feature_flag=normalize_feature_flag,
            feature_mean=feature_mean,
            feature_std=feature_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            target_feature=target_feature,
            past_frames=past_frames,
            future_frames=future_frames,
            track_x0_metrics=False,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _accumulate_metric_sums(metric_sums, metrics, target_feature.shape[0])
    return _finalize_metric_sums(metric_sums)


@torch.no_grad()
def evaluate_raw_imu_latent_dit(
    *,
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalize_feature_flag: bool,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    past_frames: int,
    future_frames: int,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_feature = batch["target_feature"]
        target_latent_raw = encode_feature_to_latent(
            vae_model=vae_model,
            target_feature=target_feature,
            normalize_feature_flag=normalize_feature_flag,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        target_latent = normalize_latent_tensor(target_latent_raw, latent_mean, latent_std)
        _, metrics = compute_noise_prediction_loss(
            model=model,
            condition=condition,
            target_latent_normalized=target_latent,
            diffusion_buffers=diffusion_buffers,
            vae_model=vae_model,
            normalize_feature_flag=normalize_feature_flag,
            feature_mean=feature_mean,
            feature_std=feature_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            target_feature=target_feature,
            past_frames=past_frames,
            future_frames=future_frames,
            track_x0_metrics=True,
        )
        _accumulate_metric_sums(metric_sums, metrics, target_feature.shape[0])
    return _finalize_metric_sums(metric_sums)


@torch.no_grad()
def export_raw_imu_latent_dit_samples(
    *,
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    dataset: RawImuLatentDiffusionWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    normalize_feature_flag: bool,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
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
    target_feature = torch.stack([sample["target_feature"] for sample in samples], dim=0).to(device)
    condition_norm = normalize_condition_tensor(condition, condition_mean, condition_std)
    target_latent_raw = encode_feature_to_latent(
        vae_model=vae_model,
        target_feature=target_feature,
        normalize_feature_flag=normalize_feature_flag,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    target_latent = normalize_latent_tensor(target_latent_raw, latent_mean, latent_std)
    sampled_latent = sample_latent_diffusion(
        model=model,
        condition=condition_norm,
        latent_shape=tuple(target_latent.shape),
        diffusion_buffers=diffusion_buffers,
        generator=torch.Generator(device=device.type).manual_seed(int(sample_seed + epoch)),
    )
    sampled_feature = decode_latent_to_feature_tensor(
        vae_model=vae_model,
        latent=sampled_latent,
        normalize_feature_flag=normalize_feature_flag,
        feature_mean=feature_mean,
        feature_std=feature_std,
        latent_mean=latent_mean,
        latent_std=latent_std,
    )
    meta_json = json.dumps([sample["meta"] for sample in samples])
    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        condition=condition.detach().cpu().numpy().astype(np.float32),
        target_feature=target_feature.detach().cpu().numpy().astype(np.float32),
        target_latent=target_latent_raw.detach().cpu().numpy().astype(np.float32),
        sampled_latent=denormalize_latent_tensor(sampled_latent, latent_mean, latent_std).detach().cpu().numpy().astype(np.float32),
        sampled_feature=sampled_feature.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: LatentDiffusionTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    raw_imu_vae_checkpoint_path: str,
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
            "latent_mean": latent_mean.astype(np.float32),
            "latent_std": latent_std.astype(np.float32),
            "raw_imu_vae_checkpoint_path": str(raw_imu_vae_checkpoint_path),
        },
        path,
    )


def load_raw_imu_latent_dit_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    raw_imu_vae_checkpoint_override: Path | None = None,
) -> tuple[
    torch.nn.Module,
    LatentDiffusionTrainingConfig,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.nn.Module,
    RawImuTemporalVaeTrainingConfig,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = LatentDiffusionTrainingConfig(**checkpoint["config"])
    raw_imu_vae_checkpoint_path = Path(checkpoint["raw_imu_vae_checkpoint_path"])
    if raw_imu_vae_checkpoint_override is not None:
        raw_imu_vae_checkpoint_path = Path(raw_imu_vae_checkpoint_override)
    vae_model, vae_config, feature_mean, feature_std, _ = load_raw_imu_temporal_vae_checkpoint(
        checkpoint_path=raw_imu_vae_checkpoint_path,
        device=device,
    )
    latent_frames = int(vae_config.window_frames // 8)
    model = build_latent_diffusion_model(
        latent_dim=vae_config.latent_dim,
        condition_frames=config.past_frames,
        condition_hidden_dim=config.condition_hidden_dim,
        latent_frames=latent_frames,
        dropout=config.dropout,
        denoiser_num_blocks=config.denoiser_num_blocks,
        fusion_mode=config.fusion_mode,
        dit_num_heads=config.dit_num_heads,
        dit_mlp_ratio=config.dit_mlp_ratio,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    condition_mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    condition_std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    latent_mean = torch.as_tensor(checkpoint["latent_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    latent_std = torch.as_tensor(checkpoint["latent_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    diffusion_buffers = build_diffusion_buffers(
        diffusion_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    return (
        model,
        config,
        condition_mean,
        condition_std,
        latent_mean,
        latent_std,
        vae_model,
        vae_config,
        feature_mean,
        feature_std,
        diffusion_buffers,
        checkpoint,
    )


def run_raw_imu_latent_dit_training(
    *,
    output_dir: Path,
    config: LatentDiffusionTrainingConfig,
    train_window_index_csv: Path | None,
    val_window_index_csv: Path | None,
    split_manifest_path: Path | None,
    pose_root: Path,
    feature_root: Path,
    train_split: str = "train",
    val_split: str = "val",
    raw_imu_vae_checkpoint_path: Path,
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
    condition_mean = torch.from_numpy(condition_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    condition_std = torch.from_numpy(condition_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    np.savez_compressed(
        output_dir / "condition_normalization_stats.npz",
        condition_mean=condition_mean_np.astype(np.float32),
        condition_std=condition_std_np.astype(np.float32),
    )

    vae_model, vae_config, feature_mean, feature_std, _ = load_raw_imu_temporal_vae_checkpoint(
        checkpoint_path=raw_imu_vae_checkpoint_path,
        device=device,
    )
    vae_model.eval()
    for parameter in vae_model.parameters():
        parameter.requires_grad_(False)

    latent_mean_np, latent_std_np = compute_latent_normalization_stats(
        dataset=train_dataset,
        vae_model=vae_model,
        normalize_feature_flag=vae_config.normalize_feature,
        feature_mean=feature_mean,
        feature_std=feature_std,
        device=device,
    )
    latent_mean = torch.from_numpy(latent_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    latent_std = torch.from_numpy(latent_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    np.savez_compressed(
        output_dir / "latent_normalization_stats.npz",
        latent_mean=latent_mean_np.astype(np.float32),
        latent_std=latent_std_np.astype(np.float32),
    )

    if int(config.past_frames + config.future_window_frames) != int(vae_config.window_frames):
        raise ValueError(
            "past_frames + future_window_frames must equal frozen VAE window_frames "
            f"({config.past_frames} + {config.future_window_frames} != {vae_config.window_frames})"
        )
    latent_frames = int(vae_config.window_frames // 8)
    model = build_latent_diffusion_model(
        latent_dim=vae_config.latent_dim,
        condition_frames=config.past_frames,
        condition_hidden_dim=config.condition_hidden_dim,
        latent_frames=latent_frames,
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

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

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
        "raw_imu_vae_checkpoint_path": str(raw_imu_vae_checkpoint_path),
        "vae_latent_dim": int(vae_config.latent_dim),
        "vae_latent_frames": int(latent_frames),
        "combined_target_window_frames": int(vae_config.window_frames),
        "latent_normalization": "train_split_per_dim",
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
            vae_model=vae_model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            normalize_feature_flag=vae_config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            condition_mean=condition_mean,
            condition_std=condition_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
            diffusion_buffers=diffusion_buffers,
            past_frames=config.past_frames,
            future_frames=config.future_window_frames,
        )
        val_metrics = evaluate_raw_imu_latent_dit(
            model=model,
            vae_model=vae_model,
            dataloader=val_loader,
            device=device,
            normalize_feature_flag=vae_config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            condition_mean=condition_mean,
            condition_std=condition_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
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
        export_raw_imu_latent_dit_samples(
            model=model,
            vae_model=vae_model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_predictions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            normalize_feature_flag=vae_config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
            condition_mean=condition_mean,
            condition_std=condition_std,
            latent_mean=latent_mean,
            latent_std=latent_std,
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
            latent_mean=latent_mean_np,
            latent_std=latent_std_np,
            raw_imu_vae_checkpoint_path=str(raw_imu_vae_checkpoint_path),
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
                latent_mean=latent_mean_np,
                latent_std=latent_std_np,
                raw_imu_vae_checkpoint_path=str(raw_imu_vae_checkpoint_path),
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
                latent_mean=latent_mean_np,
                latent_std=latent_std_np,
                raw_imu_vae_checkpoint_path=str(raw_imu_vae_checkpoint_path),
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
    parser = argparse.ArgumentParser(description="Train latent raw-IMU DiT")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--train-window-index-csv", type=Path, default=None)
    parser.add_argument("--val-window-index-csv", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--raw-imu-vae-checkpoint", type=Path, required=True)
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
    summary = run_raw_imu_latent_dit_training(
        output_dir=args.output_dir,
        config=LatentDiffusionTrainingConfig(
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
        raw_imu_vae_checkpoint_path=args.raw_imu_vae_checkpoint,
    )
    print(summary)


if __name__ == "__main__":
    main()
