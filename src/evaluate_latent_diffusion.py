#!/usr/bin/env python3
"""
Evaluate a latent diffusion checkpoint on held-out windows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from evaluate_latent_diffusion_plausibility import evaluate_latent_diffusion_plausibility
from train_imu_masked_recon import write_json
from train_latent_diffusion import (
    LatentDiffusionWindowDataset,
    _move_tensor_batch,
    build_sampling_generator,
    decode_latent_to_pose,
    encode_pose_to_latent,
    load_latent_diffusion_checkpoint,
    normalize_condition_tensor,
    normalize_latent_tensor,
    sample_latent_diffusion,
)
from train_temporal_vae import compute_pose_recon_metrics, resolve_position_smoothing_kernel


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
    "sample0_recon_mpjpe",
    "sample0_root_heading_error_deg",
    "sample0_jerk_error",
    "sample0_latent_mse",
    "min_recon_mpjpe_at_k",
    "min_root_heading_error_deg_at_k",
    "min_jerk_error_at_k",
    "min_latent_mse_at_k",
    "best_sample_index_by_mpjpe",
    "best_sample_index_by_heading",
    "best_sample_index_by_latent_mse",
)

PARTICIPANT_METRIC_COLUMNS = (
    "participant",
    "window_count",
    "sample0_recon_mpjpe",
    "sample0_root_heading_error_deg",
    "sample0_jerk_error",
    "sample0_latent_mse",
    "min_recon_mpjpe_at_k",
    "min_root_heading_error_deg_at_k",
    "min_jerk_error_at_k",
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
def evaluate_latent_diffusion_checkpoint(
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
    vae_checkpoint_override: Path | None = None,
    compute_plausibility: bool = True,
    plausibility_embedding_dim: int = 64,
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
        checkpoint,
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

    window_rows: list[dict[str, Any]] = []
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device_resolved)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_pose = batch["target_pose"]
        target_latent_raw = encode_pose_to_latent(
            vae_model=vae_model,
            target_pose=target_pose,
            normalize_pose=vae_config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
        )
        target_latent = normalize_latent_tensor(target_latent_raw, latent_mean, latent_std)

        meta_batch = batch["meta"]
        for index in range(target_pose.shape[0]):
            meta = _meta_item(meta_batch, index)
            tracks = {
                "recon_mpjpe": [],
                "root_heading_error_deg": [],
                "jerk_error": [],
                "latent_mse": [],
            }
            condition_single = condition[index : index + 1]
            target_pose_single = target_pose[index : index + 1]
            target_latent_single = target_latent[index : index + 1]
            future_start = int(config.past_frames)
            future_end = int(config.past_frames + config.future_window_frames)
            latent_shape = tuple(target_latent_single.shape)
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
                pose_metrics = compute_pose_recon_metrics(
                    prediction=sampled_pose[:, future_start:future_end],
                    target=target_pose_single[:, future_start:future_end],
                )
                latent_mse = torch.mean((sampled_latent[0] - target_latent_single[0]) ** 2).item()
                tracks["recon_mpjpe"].append(float(pose_metrics["recon_mpjpe"]))
                tracks["root_heading_error_deg"].append(float(pose_metrics["root_heading_error_deg"]))
                tracks["jerk_error"].append(float(pose_metrics["jerk_error"]))
                tracks["latent_mse"].append(float(latent_mse))
            best_sample_index_by_mpjpe = min(range(sample_count), key=lambda k: tracks["recon_mpjpe"][k])
            best_sample_index_by_heading = min(range(sample_count), key=lambda k: tracks["root_heading_error_deg"][k])
            best_sample_index_by_latent = min(range(sample_count), key=lambda k: tracks["latent_mse"][k])
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
                    "sample0_recon_mpjpe": float(tracks["recon_mpjpe"][0]),
                    "sample0_root_heading_error_deg": float(tracks["root_heading_error_deg"][0]),
                    "sample0_jerk_error": float(tracks["jerk_error"][0]),
                    "sample0_latent_mse": float(tracks["latent_mse"][0]),
                    "min_recon_mpjpe_at_k": float(min(tracks["recon_mpjpe"])),
                    "min_root_heading_error_deg_at_k": float(min(tracks["root_heading_error_deg"])),
                    "min_jerk_error_at_k": float(min(tracks["jerk_error"])),
                    "min_latent_mse_at_k": float(min(tracks["latent_mse"])),
                    "best_sample_index_by_mpjpe": int(best_sample_index_by_mpjpe),
                    "best_sample_index_by_heading": int(best_sample_index_by_heading),
                    "best_sample_index_by_latent_mse": int(best_sample_index_by_latent),
                }
            )

    participant_sums: dict[str, dict[str, float]] = {}
    metric_keys = (
        "sample0_recon_mpjpe",
        "sample0_root_heading_error_deg",
        "sample0_jerk_error",
        "sample0_latent_mse",
        "min_recon_mpjpe_at_k",
        "min_root_heading_error_deg_at_k",
        "min_jerk_error_at_k",
        "min_latent_mse_at_k",
    )
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

    global_metrics = {
        key: float(sum(float(row[key]) for row in window_rows) / max(len(window_rows), 1))
        for key in metric_keys
    }

    _write_csv(output_dir / "window_metrics.csv", WINDOW_METRIC_COLUMNS, window_rows)
    _write_csv(output_dir / "participant_metrics.csv", PARTICIPANT_METRIC_COLUMNS, participant_rows)

    sorted_by_mpjpe = sorted(window_rows, key=lambda row: row["min_recon_mpjpe_at_k"], reverse=True)
    sorted_by_heading = sorted(window_rows, key=lambda row: row["min_root_heading_error_deg_at_k"], reverse=True)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_index_csv": str(window_index_csv),
        "device_resolved": str(device_resolved),
        "position_smoothing_kernel": position_smoothing_kernel,
        "sample_count": int(sample_count),
        "sampling_seed": int(sampling_seed),
        "window_count": len(window_rows),
        "participant_count": len(participant_rows),
        "global_metrics": global_metrics,
        "top5_worst_min_recon_mpjpe_at_k": sorted_by_mpjpe[:5],
        "top5_worst_min_root_heading_error_deg_at_k": sorted_by_heading[:5],
    }
    if compute_plausibility:
        plausibility_output_dir = output_dir / "plausibility"
        plausibility_summary = evaluate_latent_diffusion_plausibility(
            checkpoint_path=checkpoint_path,
            window_index_csv=window_index_csv,
            output_dir=plausibility_output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            max_windows=max_windows,
            sample_count=sample_count,
            sampling_seed=sampling_seed,
            position_smoothing_kernel=position_smoothing_kernel,
            embedding_dim=plausibility_embedding_dim,
            vae_checkpoint_override=vae_checkpoint_override,
        )
        summary["plausibility_output_dir"] = str(plausibility_output_dir)
        summary["plausibility_global_metrics"] = plausibility_summary["global_metrics"]
        summary["plausibility_summary_path"] = str(plausibility_output_dir / "summary.json")
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent diffusion on held-out windows")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--skip-plausibility", action="store_true")
    parser.add_argument("--plausibility-embedding-dim", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_latent_diffusion_checkpoint(
        checkpoint_path=args.checkpoint,
        window_index_csv=args.window_index_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        max_windows=args.max_windows,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        position_smoothing_kernel=args.position_smoothing_kernel,
        vae_checkpoint_override=args.vae_checkpoint,
        compute_plausibility=not args.skip_plausibility,
        plausibility_embedding_dim=args.plausibility_embedding_dim,
    )
    print(summary)


if __name__ == "__main__":
    main()
