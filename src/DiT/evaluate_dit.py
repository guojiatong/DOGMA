#!/usr/bin/env python3
"""
Evaluate a direct pose-space DiT checkpoint on held-out windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation_extras import (
    DEFAULT_MOTION_ENCODER_CHECKPOINT,
    compute_trimmed_metric_means,
    evaluate_pose_distribution_metrics,
)
from evaluation_metrics import build_metrics_summary, compute_future_metrics, future_slice
from train_dit import (
    DEFAULT_CONFIG_DIR,
    DiTWindowDataset,
    _move_tensor_batch,
    build_sampling_generator,
    denormalize_target_tensor,
    load_dit_checkpoint,
    normalize_condition_tensor,
    sample_pose_diffusion,
)
from train_imu_masked_recon import write_json
from train_temporal_vae import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_POSE_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    apply_position_temporal_filter,
    apply_data_config_to_args,
    load_pose_window_records,
    resolve_position_smoothing_kernel,
)

DEFAULT_EVAL_OUTPUT_DIR = DEFAULT_CONFIG_DIR / "eval_dit"


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


@torch.no_grad()
def evaluate_dit_checkpoint(
    *,
    checkpoint_path: Path,
    window_index_csv: Path | None,
    split_manifest_path: Path | None,
    pose_root: Path,
    feature_root: Path,
    eval_split: str,
    output_dir: Path,
    batch_size: int = 8,
    num_workers: int = 0,
    device: str = "auto",
    max_windows: int = 0,
    sample_count: int = 10,
    sampling_seed: int = 0,
    position_smoothing_kernel: str = "tri5",
    motion_encoder_checkpoint: Path = DEFAULT_MOTION_ENCODER_CHECKPOINT,
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("window_metrics.csv", "participant_metrics.csv", "summary.json"):
        (output_dir / stale_name).unlink(missing_ok=True)
    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        model,
        config,
        condition_mean,
        condition_std,
        target_mean,
        target_std,
        diffusion_buffers,
        checkpoint,
    ) = load_dit_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)

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
    dataset = DiTWindowDataset(
        window_records=window_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        include_root_translation=config.include_root_translation,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metric_rows: list[dict[str, float | None]] = []
    target_pose_exports: list[np.ndarray] = []
    pred_pose_exports: list[np.ndarray] = []
    sampled_pose_candidate_exports: list[list[np.ndarray]] = []
    meta_exports: list[dict[str, Any]] = []
    fs = future_slice(config.past_frames, config.future_window_frames, config.past_frames + config.future_window_frames)
    progress = tqdm(dataloader, desc="eval")
    for batch in progress:
        batch = _move_tensor_batch(batch, device_resolved)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_pose = batch["target_pose"]
        target_pose_normalized = (target_pose - target_mean) / target_std

        meta_batch = batch["meta"]
        batch_progress = tqdm(
            range(target_pose.shape[0]),
            desc="batch_windows",
            leave=False,
        )
        for index in batch_progress:
            meta = _meta_item(meta_batch, index)
            condition_single = condition[index : index + 1]
            target_pose_single = target_pose[index : index + 1]
            target_pose_future = target_pose_single[:, fs]
            target_pose_norm_single = target_pose_normalized[index : index + 1]
            pose_shape = tuple(target_pose_norm_single.shape)
            sampled_pose_candidates: list[torch.Tensor] = []
            sampled_future_candidates: list[torch.Tensor] = []
            sample_metric_rows: list[dict[str, float | None]] = []
            for sample_index in range(sample_count):
                generator = build_sampling_generator(
                    device=device_resolved,
                    base_seed=sampling_seed,
                    meta=meta,
                    sample_index=sample_index,
                )
                sampled_pose_normalized = sample_pose_diffusion(
                    model=model,
                    condition=condition_single,
                    pose_shape=pose_shape,
                    diffusion_buffers=diffusion_buffers,
                    generator=generator,
                )
                sampled_pose = denormalize_target_tensor(
                    sampled_pose_normalized,
                    target_mean,
                    target_std,
                )
                sampled_pose = apply_position_temporal_filter(
                    sampled_pose,
                    kernel_weights=smoothing_kernel,
                )
                sampled_pose_future = sampled_pose[:, fs]
                pose_metrics = compute_future_metrics(
                    prediction=sampled_pose_future,
                    target=target_pose_future,
                )
                sampled_pose_candidates.append(sampled_pose.detach().cpu())
                sampled_future_candidates.append(sampled_pose_future.detach().cpu())
                sample_metric_rows.append(pose_metrics)
            if not sampled_pose_candidates:
                raise RuntimeError("Expected at least one sampled pose during evaluation export")
            best_sample_index_by_mpjpe = min(range(sample_count), key=lambda k: float(sample_metric_rows[k]["mpjpe"]))
            best_future_pose = sampled_future_candidates[best_sample_index_by_mpjpe]
            best_metrics = dict(sample_metric_rows[best_sample_index_by_mpjpe])
            metric_rows.append(best_metrics)
            target_pose_exports.append(target_pose_future[0].detach().cpu().numpy().astype(np.float32))
            pred_pose_exports.append(best_future_pose[0].numpy().astype(np.float32))
            sampled_pose_candidate_exports.append(
                [candidate[0].numpy().astype(np.float32) for candidate in sampled_future_candidates]
            )
            meta_exports.append(meta)
            batch_progress.set_postfix(done=index + 1, total=target_pose.shape[0])
        if metric_rows:
            progress.set_postfix(windows=len(metric_rows))

    target_pose_array = np.stack(target_pose_exports, axis=0).astype(np.float32)
    pred_pose_array = np.stack(pred_pose_exports, axis=0).astype(np.float32)
    per_window_extras, global_distribution_metrics = evaluate_pose_distribution_metrics(
        target_pose=target_pose_array,
        pred_pose=pred_pose_array,
        sampled_pose_candidates=sampled_pose_candidate_exports,
        motion_encoder_checkpoint=motion_encoder_checkpoint,
        device=device_resolved,
    )
    for row, extra_row in zip(metric_rows, per_window_extras):
        row.update(extra_row)

    np.savez_compressed(
        output_dir / "pose_predictions.npz",
        target_pose=target_pose_array,
        pred_pose=pred_pose_array,
        pred_pose_samples=np.stack(
            [np.stack(candidates, axis=0).astype(np.float32) for candidates in sampled_pose_candidate_exports],
            axis=0,
        ).astype(np.float32),
        meta_json=np.asarray(json.dumps(meta_exports)),
    )

    summary = build_metrics_summary(
        rows=metric_rows,
        sample_count=sample_count,
        future_frames=config.future_window_frames,
        include_root_translation=config.include_root_translation,
        extra={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "window_index_csv": None if window_index_csv is None else str(window_index_csv),
            "split_manifest_path": None if split_manifest_path is None else str(split_manifest_path),
            "pose_root": str(pose_root),
            "feature_root": str(feature_root),
            "eval_split": eval_split,
            "device_resolved": str(device_resolved),
            "position_smoothing_kernel": position_smoothing_kernel,
            "sampling_seed": int(sampling_seed),
            "window_count": len(metric_rows),
            "motion_distribution_metrics": global_distribution_metrics,
            "translation_metrics_p95_trimmed": compute_trimmed_metric_means(metric_rows),
        },
    )
    write_json(output_dir / "metrics.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct pose-space DiT on held-out windows")
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
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--motion-encoder-checkpoint", type=Path, default=DEFAULT_MOTION_ENCODER_CHECKPOINT)
    return apply_data_config_to_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    summary = evaluate_dit_checkpoint(
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
        position_smoothing_kernel=args.position_smoothing_kernel,
        motion_encoder_checkpoint=args.motion_encoder_checkpoint,
    )
    print(summary)


if __name__ == "__main__":
    main()
