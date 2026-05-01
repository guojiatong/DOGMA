#!/usr/bin/env python3
"""
Evaluate first-layer DOGMA plausibility metrics for latent diffusion checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_imu_masked_recon import write_json
from train_latent_diffusion import (
    LatentDiffusionWindowDataset,
    _move_tensor_batch,
    decode_latent_to_pose,
    load_latent_diffusion_checkpoint,
    normalize_condition_tensor,
    sample_latent_diffusion,
)
from train_temporal_vae import (
    JOINT_ORDER,
    POSE_POSITION_DIM,
    ROOT_HEADING_DIM,
    resolve_position_smoothing_kernel,
    root_heading_6d_to_angles,
)

REAL_WINDOW_COLUMNS = (
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "packet_start_20hz",
    "packet_end_20hz",
    "p95_joint_jitter",
    "p95_joint_jerk",
    "p95_heading_delta",
)

GENERATED_WINDOW_COLUMNS = (
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "packet_start_20hz",
    "packet_end_20hz",
    "sample_index",
    "p95_joint_jitter",
    "p95_joint_jerk",
    "p95_heading_delta",
    "jitter_band_violation",
    "jerk_band_violation",
    "heading_delta_band_violation",
    "rom_band_violation_fraction",
)

DIVERSITY_COLUMNS = (
    "participant",
    "segment_id",
    "start_frame_20hz",
    "diversity_at_k",
)

ROM_JOINT_NAMES = tuple(name for name in JOINT_ORDER if name != "stern")
ROM_AXIS_NAMES = ("yaw", "pitch")
SANITIZE_ABS_MAX = 1e6
EMBEDDING_PROJECTION_ABS_MAX = 1e4


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _meta_item(meta_batch: dict[str, Any], index: int) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key, value in meta_batch.items():
        if isinstance(value, torch.Tensor):
            item = value[index].item()
        elif isinstance(value, list):
            item = value[index]
        else:
            item = value
        meta[key] = str(item) if isinstance(item, Path) else item
    return meta


def _future_pose_components(future_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(future_pose, dtype=np.float64)
    pose = np.nan_to_num(pose, nan=0.0, posinf=SANITIZE_ABS_MAX, neginf=-SANITIZE_ABS_MAX)
    pose = np.clip(pose, -SANITIZE_ABS_MAX, SANITIZE_ABS_MAX)
    if pose.ndim != 2 or pose.shape[1] != POSE_POSITION_DIM + ROOT_HEADING_DIM:
        raise ValueError(f"future pose must have shape [T,36], got {pose.shape}")
    positions = pose[:, :POSE_POSITION_DIM].reshape(pose.shape[0], len(JOINT_ORDER), 3)
    root_heading_6d = pose[:, POSE_POSITION_DIM:]
    return positions, root_heading_6d


def _compute_p95_joint_jitter(positions: np.ndarray) -> float:
    if positions.shape[0] < 4:
        return 0.0
    jitter = positions[3:] - 3.0 * positions[2:-1] + 3.0 * positions[1:-2] - positions[:-3]
    jitter_norm = np.linalg.norm(jitter, axis=-1)
    return float(np.percentile(jitter_norm, 95.0))


def _compute_p95_joint_jerk(positions: np.ndarray) -> float:
    if positions.shape[0] < 3:
        return 0.0
    jerk = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    return float(np.percentile(jerk_norm, 95.0))


def _compute_p95_heading_delta(root_heading_6d: np.ndarray) -> float:
    if root_heading_6d.shape[0] < 2:
        return 0.0
    heading_angles = root_heading_6d_to_angles(root_heading_6d)
    heading_delta = np.abs(np.diff(heading_angles))
    return float(np.percentile(heading_delta, 95.0))


def _compute_joint_ray_rom(positions: np.ndarray) -> np.ndarray:
    if positions.shape[0] == 0:
        return np.zeros((len(ROM_JOINT_NAMES), len(ROM_AXIS_NAMES)), dtype=np.float64)
    joint_vectors = positions[:, 1:, :]
    xy_norm = np.linalg.norm(joint_vectors[..., :2], axis=-1)
    yaw = np.unwrap(np.arctan2(joint_vectors[..., 1], joint_vectors[..., 0]), axis=0)
    pitch = np.unwrap(np.arctan2(joint_vectors[..., 2], np.maximum(xy_norm, 1e-8)), axis=0)
    rom_yaw = np.max(yaw, axis=0) - np.min(yaw, axis=0)
    rom_pitch = np.max(pitch, axis=0) - np.min(pitch, axis=0)
    return np.stack([rom_yaw, rom_pitch], axis=-1)


def compute_window_plausibility_stats(future_pose: np.ndarray) -> dict[str, Any]:
    positions, root_heading_6d = _future_pose_components(future_pose)
    return {
        "p95_joint_jitter": _compute_p95_joint_jitter(positions),
        "p95_joint_jerk": _compute_p95_joint_jerk(positions),
        "p95_heading_delta": _compute_p95_heading_delta(root_heading_6d),
        "rom_joint_axis": _compute_joint_ray_rom(positions),
    }


def _compute_scalar_band(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def compute_real_bands(real_stats: list[dict[str, Any]]) -> dict[str, Any]:
    jitter = np.asarray([row["p95_joint_jitter"] for row in real_stats], dtype=np.float64)
    jerk = np.asarray([row["p95_joint_jerk"] for row in real_stats], dtype=np.float64)
    heading = np.asarray([row["p95_heading_delta"] for row in real_stats], dtype=np.float64)
    rom = np.stack([row["rom_joint_axis"] for row in real_stats], axis=0).astype(np.float64)
    rom_low = np.quantile(rom, 0.05, axis=0)
    rom_high = np.quantile(rom, 0.95, axis=0)
    return {
        "jitter_band": _compute_scalar_band(jitter),
        "jerk_band": _compute_scalar_band(jerk),
        "heading_delta_band": _compute_scalar_band(heading),
        "rom_low": rom_low,
        "rom_high": rom_high,
    }


def _outside_band(value: float, band: tuple[float, float]) -> int:
    return int(value < band[0] or value > band[1])


def compute_band_violation(value: float, band: tuple[float, float]) -> int:
    return _outside_band(value, band)


def compute_rom_band_violation_fraction(
    rom_joint_axis: np.ndarray,
    *,
    rom_low: np.ndarray,
    rom_high: np.ndarray,
) -> float:
    rom_joint_axis = np.asarray(rom_joint_axis, dtype=np.float64)
    outside = (rom_joint_axis < rom_low) | (rom_joint_axis > rom_high)
    return float(np.mean(outside.astype(np.float64)))


def fit_future_pose_embedding(
    real_future_flat: np.ndarray,
    *,
    max_embedding_dim: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    real_future_flat = np.asarray(real_future_flat, dtype=np.float64)
    if real_future_flat.ndim != 2:
        raise ValueError(f"real_future_flat must have shape [N,D], got {real_future_flat.shape}")
    mean = np.mean(real_future_flat, axis=0, keepdims=True)
    centered = real_future_flat - mean
    max_dim = min(int(max_embedding_dim), centered.shape[0] - 1, centered.shape[1])
    if max_dim <= 0:
        basis = np.zeros((centered.shape[1], 1), dtype=np.float64)
        basis[0, 0] = 1.0
        return mean[0], basis
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:max_dim].T.astype(np.float64)
    basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
    return mean[0], basis


def project_future_pose_embedding(
    future_flat: np.ndarray,
    *,
    embedding_mean: np.ndarray,
    embedding_basis: np.ndarray,
) -> np.ndarray:
    future_flat = np.asarray(future_flat, dtype=np.float64)
    future_flat = np.nan_to_num(future_flat, nan=0.0, posinf=SANITIZE_ABS_MAX, neginf=-SANITIZE_ABS_MAX)
    future_flat = np.clip(future_flat, -SANITIZE_ABS_MAX, SANITIZE_ABS_MAX)
    centered = future_flat - embedding_mean.reshape(1, -1)
    centered = np.nan_to_num(centered, nan=0.0, posinf=EMBEDDING_PROJECTION_ABS_MAX, neginf=-EMBEDDING_PROJECTION_ABS_MAX)
    centered = np.clip(centered, -EMBEDDING_PROJECTION_ABS_MAX, EMBEDDING_PROJECTION_ABS_MAX)
    safe_basis = np.nan_to_num(embedding_basis, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        projected = np.matmul(centered.astype(np.float32), safe_basis.astype(np.float32)).astype(np.float64)
    projected = np.nan_to_num(projected, nan=0.0, posinf=EMBEDDING_PROJECTION_ABS_MAX, neginf=-EMBEDDING_PROJECTION_ABS_MAX)
    return np.clip(projected, -EMBEDDING_PROJECTION_ABS_MAX, EMBEDDING_PROJECTION_ABS_MAX)


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    clipped = np.clip(eigenvalues, 0.0, None)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        reconstructed = (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T
    reconstructed = np.nan_to_num(
        reconstructed,
        nan=0.0,
        posinf=SANITIZE_ABS_MAX,
        neginf=-SANITIZE_ABS_MAX,
    )
    return 0.5 * (reconstructed + reconstructed.T)


def compute_fid_pose(real_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> float:
    real_embeddings = np.asarray(real_embeddings, dtype=np.float64)
    generated_embeddings = np.asarray(generated_embeddings, dtype=np.float64)
    if real_embeddings.ndim != 2 or generated_embeddings.ndim != 2:
        raise ValueError("Embeddings must be rank-2 arrays")
    if real_embeddings.shape[1] != generated_embeddings.shape[1]:
        raise ValueError("Embedding dimensions must match")
    mu_real = np.mean(real_embeddings, axis=0)
    mu_gen = np.mean(generated_embeddings, axis=0)

    def _covariance(embeddings: np.ndarray) -> np.ndarray:
        centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
        if embeddings.shape[0] <= 1:
            return np.zeros((embeddings.shape[1], embeddings.shape[1]), dtype=np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cov = (centered.T @ centered) / float(embeddings.shape[0])
        cov = np.nan_to_num(cov, nan=0.0, posinf=SANITIZE_ABS_MAX, neginf=-SANITIZE_ABS_MAX)
        return 0.5 * (cov + cov.T)

    cov_real = _covariance(real_embeddings)
    cov_gen = _covariance(generated_embeddings)
    sqrt_cov_real = _matrix_sqrt_psd(cov_real)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cov_cross = sqrt_cov_real @ cov_gen @ sqrt_cov_real
    cov_cross = np.nan_to_num(cov_cross, nan=0.0, posinf=SANITIZE_ABS_MAX, neginf=-SANITIZE_ABS_MAX)
    sqrt_cov_cross = _matrix_sqrt_psd(cov_cross)
    mean_distance = float(np.sum((mu_gen - mu_real) ** 2))
    trace_term = float(np.trace(cov_real + cov_gen - 2.0 * sqrt_cov_cross))
    return float(max(mean_distance + trace_term, 0.0))


def compute_diversity_at_k(embeddings: np.ndarray) -> float:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    sample_count = int(embeddings.shape[0])
    if sample_count <= 1:
        return 0.0
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    pairwise = np.linalg.norm(diffs, axis=-1)
    upper = pairwise[np.triu_indices(sample_count, k=1)]
    return float(np.mean(upper))


def _default_run_label(checkpoint_path: Path) -> str:
    text = str(checkpoint_path)
    if "latent_diffusion_fulltrain_autoresearch_2h" in text:
        return f"2h-ft {checkpoint_path.parent.name}"
    run_name = checkpoint_path.parent.name
    if "100e" in run_name:
        return "100e full-train"
    if "24e" in run_name:
        return "24e full-train"
    return run_name


@torch.no_grad()
def evaluate_latent_diffusion_plausibility(
    *,
    checkpoint_path: Path,
    window_index_csv: Path,
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "auto",
    max_windows: int = 0,
    sample_count: int = 10,
    sampling_seed: int = 0,
    position_smoothing_kernel: str = "tri5",
    embedding_dim: int = 64,
    vae_checkpoint_override: Path | None = None,
    run_label: str | None = None,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        config,
        condition_mean,
        condition_std,
        latent_mean,
        latent_std,
        vae_model,
        vae_config,
        pose_mean,
        pose_std,
        diffusion_buffers,
        _checkpoint,
    ) = load_latent_diffusion_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
        vae_checkpoint_override=vae_checkpoint_override,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)
    dataset = LatentDiffusionWindowDataset.from_window_index_csv(
        window_index_csv=window_index_csv,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        max_windows=max_windows,
        shuffle=False,
        subset_seed=0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    future_start = int(config.past_frames)
    future_end = int(config.past_frames + config.future_window_frames)
    latent_frames = int(vae_config.window_frames // 8)
    latent_shape = (0, latent_frames, int(vae_config.latent_dim))

    real_window_rows: list[dict[str, Any]] = []
    real_stats: list[dict[str, Any]] = []
    real_future_flat: list[np.ndarray] = []
    for batch in dataloader:
        target_pose = batch["target_pose"].numpy().astype(np.float64)
        meta_batch = batch["meta"]
        future_pose = target_pose[:, future_start:future_end]
        for index in range(future_pose.shape[0]):
            meta = _meta_item(meta_batch, index)
            stats = compute_window_plausibility_stats(future_pose[index])
            real_stats.append(stats)
            real_future_flat.append(future_pose[index].reshape(-1))
            real_window_rows.append(
                {
                    "participant": meta["participant"],
                    "segment_id": meta["segment_id"],
                    "pose_path": meta["pose_path"],
                    "feature_path": meta["feature_path"],
                    "start_frame_20hz": int(meta["start_frame_20hz"]),
                    "packet_start_20hz": int(meta["packet_start_20hz"]),
                    "packet_end_20hz": int(meta["packet_end_20hz"]),
                    "p95_joint_jitter": float(stats["p95_joint_jitter"]),
                    "p95_joint_jerk": float(stats["p95_joint_jerk"]),
                    "p95_heading_delta": float(stats["p95_heading_delta"]),
                }
            )
    if not real_future_flat:
        raise ValueError("No held-out future windows available for plausibility evaluation")

    real_bands = compute_real_bands(real_stats)
    real_future_flat_array = np.stack(real_future_flat, axis=0).astype(np.float64)
    embedding_mean, embedding_basis = fit_future_pose_embedding(
        real_future_flat_array,
        max_embedding_dim=embedding_dim,
    )
    real_embeddings = project_future_pose_embedding(
        real_future_flat_array,
        embedding_mean=embedding_mean,
        embedding_basis=embedding_basis,
    )

    generated_window_rows: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    generated_embeddings: list[np.ndarray] = []

    for batch in dataloader:
        batch = _move_tensor_batch(batch, device_resolved)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        batch_size_actual = int(condition.shape[0])
        meta_batch = batch["meta"]
        window_embeddings: list[list[np.ndarray]] = [[] for _ in range(batch_size_actual)]
        latent_shape = (batch_size_actual, latent_frames, int(vae_config.latent_dim))
        for sample_index in range(sample_count):
            batch_seed_key = "|".join(
                [
                    str(meta_batch["participant"][0]),
                    str(meta_batch["segment_id"][0]),
                    str(meta_batch["start_frame_20hz"][0]),
                    str(sample_index),
                    str(sampling_seed),
                ]
            )
            batch_seed = zlib.adler32(batch_seed_key.encode("utf-8")) & 0xFFFFFFFF
            generator = torch.Generator(device=device_resolved.type)
            generator.manual_seed(int(batch_seed % (2**31 - 1)))
            sampled_latent = sample_latent_diffusion(
                model=model,
                condition=condition,
                latent_shape=latent_shape,
                diffusion_buffers=diffusion_buffers,
                generator=generator,
            )
            sampled_pose = decode_latent_to_pose(
                vae_model=vae_model,
                latent=sampled_latent,
                normalize_pose=vae_config.normalize_pose,
                pose_mean=pose_mean,
                pose_std=pose_std,
                latent_mean=latent_mean,
                latent_std=latent_std,
                position_smoothing_kernel=smoothing_kernel,
            )
            future_pose = sampled_pose[:, future_start:future_end].detach().cpu().numpy().astype(np.float64)
            future_flat = future_pose.reshape(future_pose.shape[0], -1)
            batch_embeddings = project_future_pose_embedding(
                future_flat,
                embedding_mean=embedding_mean,
                embedding_basis=embedding_basis,
            )
            for index in range(batch_size_actual):
                meta = _meta_item(meta_batch, index)
                stats = compute_window_plausibility_stats(future_pose[index])
                jitter_violation = _outside_band(stats["p95_joint_jitter"], real_bands["jitter_band"])
                jerk_violation = _outside_band(stats["p95_joint_jerk"], real_bands["jerk_band"])
                heading_violation = _outside_band(stats["p95_heading_delta"], real_bands["heading_delta_band"])
                rom_violation_fraction = compute_rom_band_violation_fraction(
                    stats["rom_joint_axis"],
                    rom_low=real_bands["rom_low"],
                    rom_high=real_bands["rom_high"],
                )
                generated_embeddings.append(batch_embeddings[index])
                window_embeddings[index].append(batch_embeddings[index])
                generated_window_rows.append(
                    {
                        "participant": meta["participant"],
                        "segment_id": meta["segment_id"],
                        "pose_path": meta["pose_path"],
                        "feature_path": meta["feature_path"],
                        "start_frame_20hz": int(meta["start_frame_20hz"]),
                        "packet_start_20hz": int(meta["packet_start_20hz"]),
                        "packet_end_20hz": int(meta["packet_end_20hz"]),
                        "sample_index": int(sample_index),
                        "p95_joint_jitter": float(stats["p95_joint_jitter"]),
                        "p95_joint_jerk": float(stats["p95_joint_jerk"]),
                        "p95_heading_delta": float(stats["p95_heading_delta"]),
                        "jitter_band_violation": int(jitter_violation),
                        "jerk_band_violation": int(jerk_violation),
                        "heading_delta_band_violation": int(heading_violation),
                        "rom_band_violation_fraction": float(rom_violation_fraction),
                    }
                )
        for index in range(batch_size_actual):
            meta = _meta_item(meta_batch, index)
            diversity_rows.append(
                {
                    "participant": meta["participant"],
                    "segment_id": meta["segment_id"],
                    "start_frame_20hz": int(meta["start_frame_20hz"]),
                    "diversity_at_k": float(compute_diversity_at_k(np.stack(window_embeddings[index], axis=0))),
                }
            )

    generated_embeddings_array = np.stack(generated_embeddings, axis=0).astype(np.float64)
    diversity_values = np.asarray([row["diversity_at_k"] for row in diversity_rows], dtype=np.float64)
    jitter_violation_rate = float(np.mean([row["jitter_band_violation"] for row in generated_window_rows]))
    jerk_violation_rate = float(np.mean([row["jerk_band_violation"] for row in generated_window_rows]))
    heading_violation_rate = float(np.mean([row["heading_delta_band_violation"] for row in generated_window_rows]))
    rom_violation_rate = float(np.mean([row["rom_band_violation_fraction"] for row in generated_window_rows]))
    fid_pose = compute_fid_pose(real_embeddings, generated_embeddings_array)
    diversity_at_k = float(np.mean(diversity_values)) if diversity_values.size else 0.0

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "run_name": checkpoint_path.parent.name,
        "run_label": run_label or _default_run_label(checkpoint_path),
        "window_index_csv": str(window_index_csv),
        "sample_count": int(sample_count),
        "window_count": int(len(real_window_rows)),
        "generated_window_count": int(len(generated_window_rows)),
        "embedding_dim": int(real_embeddings.shape[1]),
        "position_smoothing_kernel": str(position_smoothing_kernel),
        "real_bands": {
            "jitter_band": [float(real_bands["jitter_band"][0]), float(real_bands["jitter_band"][1])],
            "jerk_band": [float(real_bands["jerk_band"][0]), float(real_bands["jerk_band"][1])],
            "heading_delta_band": [
                float(real_bands["heading_delta_band"][0]),
                float(real_bands["heading_delta_band"][1]),
            ],
            "rom_joint_axis_low": real_bands["rom_low"].tolist(),
            "rom_joint_axis_high": real_bands["rom_high"].tolist(),
            "rom_joint_names": list(ROM_JOINT_NAMES),
            "rom_axis_names": list(ROM_AXIS_NAMES),
        },
        "global_metrics": {
            "fid_pose": float(fid_pose),
            "diversity_at_k": float(diversity_at_k),
            "jitter_band_violation_rate": float(jitter_violation_rate),
            "jerk_band_violation_rate": float(jerk_violation_rate),
            "heading_delta_band_violation_rate": float(heading_violation_rate),
            "rom_band_violation_rate": float(rom_violation_rate),
        },
    }

    _write_csv(output_dir / "real_window_metrics.csv", REAL_WINDOW_COLUMNS, real_window_rows)
    _write_csv(output_dir / "generated_window_metrics.csv", GENERATED_WINDOW_COLUMNS, generated_window_rows)
    _write_csv(output_dir / "diversity_by_window.csv", DIVERSITY_COLUMNS, diversity_rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent diffusion plausibility metrics")
    parser.add_argument("--checkpoint-path", type=Path, required=True, help="Checkpoint best.pt path")
    parser.add_argument("--window-index-csv", type=Path, required=True, help="Held-out future window index CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--device", type=str, default="auto", help="Device name or auto")
    parser.add_argument("--max-windows", type=int, default=0, help="Optional cap on held-out windows")
    parser.add_argument("--sample-count", type=int, default=10, help="Generated samples per condition")
    parser.add_argument("--sampling-seed", type=int, default=23, help="Sampling base seed")
    parser.add_argument(
        "--position-smoothing-kernel",
        type=str,
        default="tri5",
        choices=("none", "tri3", "tri5"),
        help="Position smoothing kernel applied after VAE decoding",
    )
    parser.add_argument("--embedding-dim", type=int, default=64, help="Future-only PCA embedding dim for FID")
    parser.add_argument(
        "--vae-checkpoint-override",
        type=Path,
        default=None,
        help="Optional override for the frozen temporal VAE checkpoint",
    )
    parser.add_argument("--run-label", type=str, default=None, help="Optional display label saved into summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_latent_diffusion_plausibility(
        checkpoint_path=args.checkpoint_path,
        window_index_csv=args.window_index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_windows=args.max_windows,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        position_smoothing_kernel=args.position_smoothing_kernel,
        embedding_dim=args.embedding_dim,
        vae_checkpoint_override=args.vae_checkpoint_override,
        run_label=args.run_label,
    )
    print(summary["global_metrics"])


if __name__ == "__main__":
    main()
