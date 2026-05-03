#!/usr/bin/env python3
"""
Evaluate a latent raw-IMU DiT checkpoint on held-out windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

VAE_DIR = Path(__file__).resolve().parents[1]
if str(VAE_DIR) not in sys.path:
    sys.path.insert(0, str(VAE_DIR))

from train_imu_masked_recon import write_json  # noqa: E402
from raw_imu_ablation import compute_raw_imu_recon_metrics  # noqa: E402
from train_dit import _move_tensor_batch  # noqa: E402
from train_latent_dit import build_sampling_generator, normalize_condition_tensor, sample_latent_diffusion  # noqa: E402
from train_raw_imu_latent_diffusion import (  # noqa: E402
    RawImuLatentDiffusionWindowDataset,
    decode_latent_to_feature_tensor,
    encode_feature_to_latent,
)
from train_raw_imu_latent_dit import DEFAULT_CONFIG_DIR, DEFAULT_VAL_WINDOW_INDEX_CSV, load_raw_imu_latent_dit_checkpoint  # noqa: E402
from train_temporal_vae import DEFAULT_FEATURE_ROOT, DEFAULT_POSE_ROOT, DEFAULT_SPLIT_MANIFEST, apply_data_config_to_args, load_pose_window_records  # noqa: E402

DEFAULT_EVAL_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "eval_raw_imu_latent_dit"

WINDOW_METRIC_COLUMNS = (
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
    "sample_count",
    "sample0_feature_rmse",
    "sample0_rot_rmse",
    "sample0_gyr_rmse",
    "sample0_freeacc_rmse",
    "sample0_interp_rmse",
    "sample0_latent_mse",
    "min_feature_rmse_at_k",
    "min_rot_rmse_at_k",
    "min_gyr_rmse_at_k",
    "min_freeacc_rmse_at_k",
    "min_interp_rmse_at_k",
    "min_latent_mse_at_k",
    "best_sample_index_by_feature_rmse",
    "best_sample_index_by_latent_mse",
)

PARTICIPANT_METRIC_COLUMNS = (
    "participant",
    "window_count",
    "sample0_feature_rmse",
    "sample0_rot_rmse",
    "sample0_gyr_rmse",
    "sample0_freeacc_rmse",
    "sample0_interp_rmse",
    "sample0_latent_mse",
    "min_feature_rmse_at_k",
    "min_rot_rmse_at_k",
    "min_gyr_rmse_at_k",
    "min_freeacc_rmse_at_k",
    "min_interp_rmse_at_k",
    "min_latent_mse_at_k",
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
        meta[key] = str(item) if isinstance(item, Path) else item
    return meta


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate_raw_imu_latent_dit_checkpoint(
    *,
    checkpoint_path: Path,
    window_index_csv: Path | None,
    split_manifest_path: Path | None,
    pose_root: Path,
    feature_root: Path,
    eval_split: str,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
    max_windows: int = 0,
    sample_count: int = 5,
    sampling_seed: int = 0,
    raw_imu_vae_checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").unlink(missing_ok=True)
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
        feature_mean,
        feature_std,
        diffusion_buffers,
        checkpoint,
    ) = load_raw_imu_latent_dit_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
        raw_imu_vae_checkpoint_override=raw_imu_vae_checkpoint_override,
    )
    window_records = load_pose_window_records(
        window_index_csv=window_index_csv,
        split_manifest_path=split_manifest_path,
        split=eval_split,
        window_frames=int(config.past_frames + config.future_window_frames),
        stride_frames=int(config.past_frames),
        pose_root=pose_root,
        feature_root=feature_root,
    )
    if max_windows > 0:
        window_records = window_records[:max_windows]
    dataset = RawImuLatentDiffusionWindowDataset(
        window_records=window_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    window_rows: list[dict[str, Any]] = []
    target_feature_exports: list[np.ndarray] = []
    pred_feature_exports: list[np.ndarray] = []
    pred_feature_sample_exports: list[np.ndarray] = []
    meta_exports: list[dict[str, Any]] = []
    progress = tqdm(dataloader, desc="eval")
    for batch in progress:
        batch = _move_tensor_batch(batch, device_resolved)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_feature = batch["target_feature"]
        target_latent = encode_feature_to_latent(
            vae_model=vae_model,
            target_feature=target_feature,
            normalize_feature_flag=vae_config.normalize_feature,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        meta_batch = batch["meta"]
        batch_progress = tqdm(range(target_feature.shape[0]), desc="batch_windows", leave=False)
        for index in batch_progress:
            meta = _meta_item(meta_batch, index)
            tracks = {
                "feature_rmse": [],
                "rot_rmse": [],
                "gyr_rmse": [],
                "freeacc_rmse": [],
                "interp_rmse": [],
                "latent_mse": [],
            }
            condition_single = condition[index : index + 1]
            target_feature_single = target_feature[index : index + 1]
            target_latent_single = target_latent[index : index + 1]
            latent_shape = tuple(target_latent_single.shape)
            sampled_feature_candidates: list[torch.Tensor] = []
            for sample_index in range(sample_count):
                generator = build_sampling_generator(
                    device=device_resolved,
                    base_seed=sampling_seed,
                    meta=meta,
                    sample_index=sample_index,
                )
                sampled_latent = sample_latent_diffusion(
                    model=model,
                    condition=condition_single,
                    latent_shape=latent_shape,
                    diffusion_buffers=diffusion_buffers,
                    generator=generator,
                )
                sampled_feature = decode_latent_to_feature_tensor(
                    vae_model=vae_model,
                    latent=sampled_latent,
                    normalize_feature_flag=vae_config.normalize_feature,
                    feature_mean=feature_mean,
                    feature_std=feature_std,
                    latent_mean=latent_mean,
                    latent_std=latent_std,
                )
                feature_metrics = compute_raw_imu_recon_metrics(sampled_feature, target_feature_single)
                latent_mse = torch.mean((sampled_latent[0] - target_latent_single[0]) ** 2).item()
                sampled_feature_candidates.append(sampled_feature.detach().cpu())
                tracks["feature_rmse"].append(float(feature_metrics["feature_rmse"]))
                tracks["rot_rmse"].append(float(feature_metrics["rot_rmse"]))
                tracks["gyr_rmse"].append(float(feature_metrics["gyr_rmse"]))
                tracks["freeacc_rmse"].append(float(feature_metrics["freeacc_rmse"]))
                tracks["interp_rmse"].append(float(feature_metrics["interp_rmse"]))
                tracks["latent_mse"].append(float(latent_mse))
            best_sample_index_by_feature = min(range(sample_count), key=lambda k: tracks["feature_rmse"][k])
            best_sample_index_by_latent = min(range(sample_count), key=lambda k: tracks["latent_mse"][k])
            best_feature = sampled_feature_candidates[best_sample_index_by_feature]
            target_feature_exports.append(target_feature_single[0].detach().cpu().numpy().astype(np.float32))
            pred_feature_exports.append(best_feature[0].numpy().astype(np.float32))
            pred_feature_sample_exports.append(
                np.stack([candidate[0].numpy().astype(np.float32) for candidate in sampled_feature_candidates], axis=0)
            )
            meta_exports.append(meta)
            window_rows.append(
                {
                    "participant": meta["participant"],
                    "segment_id": meta["segment_id"],
                    "pose_path": meta["pose_path"],
                    "feature_path": meta["feature_path"],
                    "start_frame_20hz": int(meta["start_frame_20hz"]),
                    "valid_frames": int(config.future_window_frames),
                    "packet_start_20hz": int(meta["packet_start_20hz"]),
                    "packet_end_20hz": int(meta["packet_end_20hz"]),
                    "sample_count": int(sample_count),
                    "sample0_feature_rmse": float(tracks["feature_rmse"][0]),
                    "sample0_rot_rmse": float(tracks["rot_rmse"][0]),
                    "sample0_gyr_rmse": float(tracks["gyr_rmse"][0]),
                    "sample0_freeacc_rmse": float(tracks["freeacc_rmse"][0]),
                    "sample0_interp_rmse": float(tracks["interp_rmse"][0]),
                    "sample0_latent_mse": float(tracks["latent_mse"][0]),
                    "min_feature_rmse_at_k": float(min(tracks["feature_rmse"])),
                    "min_rot_rmse_at_k": float(min(tracks["rot_rmse"])),
                    "min_gyr_rmse_at_k": float(min(tracks["gyr_rmse"])),
                    "min_freeacc_rmse_at_k": float(min(tracks["freeacc_rmse"])),
                    "min_interp_rmse_at_k": float(min(tracks["interp_rmse"])),
                    "min_latent_mse_at_k": float(min(tracks["latent_mse"])),
                    "best_sample_index_by_feature_rmse": int(best_sample_index_by_feature),
                    "best_sample_index_by_latent_mse": int(best_sample_index_by_latent),
                }
            )
            batch_progress.set_postfix(done=index + 1, total=target_feature.shape[0])
        if window_rows:
            progress.set_postfix(windows=len(window_rows))

    metric_keys = (
        "sample0_feature_rmse",
        "sample0_rot_rmse",
        "sample0_gyr_rmse",
        "sample0_freeacc_rmse",
        "sample0_interp_rmse",
        "sample0_latent_mse",
        "min_feature_rmse_at_k",
        "min_rot_rmse_at_k",
        "min_gyr_rmse_at_k",
        "min_freeacc_rmse_at_k",
        "min_interp_rmse_at_k",
        "min_latent_mse_at_k",
    )
    participant_sums: dict[str, dict[str, float]] = {}
    for row in window_rows:
        participant = str(row["participant"])
        if participant not in participant_sums:
            participant_sums[participant] = {"count": 0.0}
        participant_sums[participant]["count"] += 1.0
        for key in metric_keys:
            participant_sums[participant][key] = participant_sums[participant].get(key, 0.0) + float(row[key])

    participant_rows: list[dict[str, Any]] = []
    for participant in sorted(participant_sums):
        total = participant_sums[participant]
        count = max(total["count"], 1.0)
        participant_rows.append(
            {
                "participant": participant,
                "window_count": int(total["count"]),
                **{key: float(total[key] / count) for key in metric_keys},
            }
        )
    global_metrics = {key: float(sum(float(row[key]) for row in window_rows) / max(len(window_rows), 1)) for key in metric_keys}
    _write_csv(output_dir / "window_metrics.csv", WINDOW_METRIC_COLUMNS, window_rows)
    _write_csv(output_dir / "participant_metrics.csv", PARTICIPANT_METRIC_COLUMNS, participant_rows)
    np.savez_compressed(
        output_dir / "feature_predictions.npz",
        target_feature=np.stack(target_feature_exports, axis=0).astype(np.float32),
        pred_feature=np.stack(pred_feature_exports, axis=0).astype(np.float32),
        pred_feature_samples=np.stack(pred_feature_sample_exports, axis=0).astype(np.float32),
        meta_json=np.asarray(json.dumps(meta_exports)),
    )
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_index_csv": None if window_index_csv is None else str(window_index_csv),
        "split_manifest_path": None if split_manifest_path is None else str(split_manifest_path),
        "pose_root": str(pose_root),
        "feature_root": str(feature_root),
        "eval_split": eval_split,
        "device_resolved": str(device_resolved),
        "sample_count": int(sample_count),
        "sampling_seed": int(sampling_seed),
        "window_count": len(window_rows),
        "participant_count": len(participant_rows),
        "global_metrics": global_metrics,
    }
    write_json(output_dir / "metrics.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent raw-IMU DiT on held-out windows")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-index-csv", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--eval-split", type=str, default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--raw-imu-vae-checkpoint", type=Path, default=None)
    return apply_data_config_to_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    summary = evaluate_raw_imu_latent_dit_checkpoint(
        checkpoint_path=args.checkpoint,
        window_index_csv=args.window_index_csv,
        split_manifest_path=args.split_manifest,
        pose_root=args.pose_root,
        feature_root=args.feature_root,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_windows=args.max_windows,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        raw_imu_vae_checkpoint_override=args.raw_imu_vae_checkpoint,
    )
    print(summary)


if __name__ == "__main__":
    main()
