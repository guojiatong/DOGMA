#!/usr/bin/env python3
"""
Export raw-IMU latent diffusion sample videos by projecting decoded IMU back to pseudo-pose.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export_temporal_vae_failure_pack import (
    build_context_target_payload,
    compare_payloads,
    render_payload_comparison_video,
    save_payload_npz,
    slice_payload,
    stitch_future_payload_with_context,
)
from raw_imu_ablation import (
    flat_feature_window_to_pseudo_pose_payload,
    predicted_rot6d_to_quaternion,
    slice_flat_feature_window_from_path,
    unflatten_feature_window,
)
from train_imu_masked_recon import write_json
from train_latent_diffusion import build_sampling_generator, normalize_condition_tensor, sample_latent_diffusion
from train_raw_imu_latent_diffusion import (
    decode_latent_to_feature_tensor,
    load_raw_imu_latent_diffusion_checkpoint,
)
from train_temporal_vae import flatten_pose_window, rebase_root_heading_6d, compute_pose_recon_metrics
from video_export_paths import build_export_video_path
from visualize_pose import PseudoPosePayload, load_pseudo_pose


COMPARISON_COLUMNS = (
    "rank",
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
    "context_frames",
    "prediction_frames",
    "sample_count",
    "selected_sample_index",
    "selected_sample_recon_mpjpe",
    "selected_sample_root_heading_error_deg",
    "selected_sample_jerk_error",
    "target_vs_sample_mean_mpjpe_mm",
    "target_vs_sample_max_mpjpe_mm",
    "target_vs_sample_mean_heading_error_deg",
    "target_vs_sample_max_heading_error_deg",
    "window_dir",
    "video_path",
)


def load_window_records(window_index_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(window_index_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            feature_path = Path(row["feature_path"])
            rows.append(
                {
                    "participant": row["participant"],
                    "segment_id": row["segment_id"],
                    "pose_path": Path(row["pose_path"]),
                    "feature_path": feature_path,
                    "start_frame_20hz": int(row["start_frame_20hz"]),
                    "valid_frames": int(row["end_frame_20hz"]) - int(row["start_frame_20hz"]) + 1,
                }
            )
    if not rows:
        raise ValueError(f"No windows found in {window_index_csv}")
    return rows


def select_window_records(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    seed: int,
    participant: str | None,
    segment_id: str | None,
    center_start_frame: int | None,
) -> list[dict[str, Any]]:
    filtered = list(rows)
    if participant is not None:
        filtered = [row for row in filtered if row["participant"] == participant]
    if segment_id is not None:
        filtered = [row for row in filtered if str(row["segment_id"]) == str(segment_id)]
    if center_start_frame is not None:
        filtered = sorted(
            filtered,
            key=lambda row: (abs(int(row["start_frame_20hz"]) - int(center_start_frame)), int(row["start_frame_20hz"])),
        )
    else:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(filtered)).tolist()
        filtered = [filtered[int(index)] for index in order]
    if not filtered:
        raise ValueError("No windows matched the requested filters")
    return filtered[:top_k]


def rebase_payload(payload: PseudoPosePayload) -> PseudoPosePayload:
    return PseudoPosePayload(
        packet_counter=payload.packet_counter.copy(),
        relative_positions=payload.relative_positions.copy(),
        root_heading_6d=rebase_root_heading_6d(payload.root_heading_6d).astype(np.float64),
        is_interpolated=payload.is_interpolated.copy(),
        joint_names=list(payload.joint_names),
    )


@torch.no_grad()
def export_raw_imu_latent_diffusion_sample_videos(
    *,
    checkpoint_path: Path,
    window_index_csv: Path,
    output_dir: Path,
    top_k: int = 5,
    sample_count: int = 10,
    sampling_seed: int = 0,
    context_frames: int = 120,
    prediction_frames: int = 40,
    participant: str | None = None,
    segment_id: str | None = None,
    center_start_frame: int | None = None,
    render_space: str = "heading",
    body_model: str = "smal",
    show_skeleton_overlay: bool = False,
    smal_model: Path | None = None,
    smal_mapping: Path | None = None,
    smal_data: Path | None = None,
    smal_family_index: int = 1,
    fps: int = 20,
    point_size: float = 42.0,
    device: str = "auto",
    no_video: bool = False,
    raw_imu_vae_checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if smal_model is None:
        smal_model = workspace_root / "SMAL" / "wolf_alph3.pkl"

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
    ) = load_raw_imu_latent_diffusion_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
        raw_imu_vae_checkpoint_override=raw_imu_vae_checkpoint_override,
    )
    if int(config.past_frames) != int(context_frames) or int(config.future_window_frames) != int(prediction_frames):
        raise ValueError(
            "Requested context/prediction frames do not match checkpoint config "
            f"({context_frames}/{prediction_frames} vs {config.past_frames}/{config.future_window_frames})"
        )
    latent_shape = (1, int(vae_config.window_frames // 8), int(vae_config.latent_dim))
    rows = [
        row
        for row in load_window_records(window_index_csv)
        if row["start_frame_20hz"] >= context_frames and row["valid_frames"] >= prediction_frames
    ]
    selected_rows = select_window_records(
        rows,
        top_k=top_k,
        seed=sampling_seed,
        participant=participant,
        segment_id=segment_id,
        center_start_frame=center_start_frame,
    )

    summary_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected_rows, start=1):
        target_payload_full = load_pseudo_pose(row["pose_path"])
        target_context_future = build_context_target_payload(
            full_payload=target_payload_full,
            future_start_frame=row["start_frame_20hz"],
            context_frames=context_frames,
            future_frames=prediction_frames,
        )
        combined_feature_np, combined_packets = slice_flat_feature_window_from_path(
            feature_path=row["feature_path"],
            start_frame=int(row["start_frame_20hz"] - context_frames),
            num_frames=int(context_frames + prediction_frames),
        )
        reference_quat = predicted_rot6d_to_quaternion(
            unflatten_feature_window(combined_feature_np[:1])[:, :, :6]
        )[0]
        condition_np = combined_feature_np[:context_frames]
        condition = torch.from_numpy(condition_np[None]).to(device=device_resolved, dtype=torch.float32)
        condition = normalize_condition_tensor(condition, condition_mean, condition_std)

        best_metrics: dict[str, float] | None = None
        best_combined_payload: PseudoPosePayload | None = None
        best_sample_index = 0
        for sample_index in range(int(sample_count)):
            generator = build_sampling_generator(
                device=device_resolved,
                base_seed=sampling_seed,
                meta={
                    "participant": row["participant"],
                    "segment_id": row["segment_id"],
                    "start_frame_20hz": row["start_frame_20hz"],
                    "packet_start_20hz": int(target_context_future.packet_counter[0]),
                },
                sample_index=sample_index,
            )
            sampled_latent = sample_latent_diffusion(
                model=model,
                condition=condition,
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
            )[0].detach().cpu().numpy().astype(np.float32)
            predicted_combined_payload = rebase_payload(
                flat_feature_window_to_pseudo_pose_payload(
                    flat_feature_window=sampled_feature,
                    packet_counter=combined_packets,
                    reference_quat=reference_quat,
                )
            )
            target_future = slice_payload(
                target_context_future,
                start_frame=context_frames,
                num_frames=prediction_frames,
                rebase_heading=False,
            )
            predicted_future = slice_payload(
                predicted_combined_payload,
                start_frame=context_frames,
                num_frames=prediction_frames,
                rebase_heading=False,
            )
            target_pose_tensor = torch.from_numpy(
                flatten_pose_window(
                    target_future.relative_positions.astype(np.float32),
                    target_future.root_heading_6d.astype(np.float32),
                )[None]
            )
            predicted_pose_tensor = torch.from_numpy(
                flatten_pose_window(
                    predicted_future.relative_positions.astype(np.float32),
                    predicted_future.root_heading_6d.astype(np.float32),
                )[None]
            )
            pose_metrics = compute_pose_recon_metrics(
                prediction=predicted_pose_tensor,
                target=target_pose_tensor,
            )
            if best_metrics is None or float(pose_metrics["recon_mpjpe"]) <= float(best_metrics["recon_mpjpe"]):
                best_metrics = dict(pose_metrics)
                best_combined_payload = predicted_combined_payload
                best_sample_index = sample_index

        if best_metrics is None or best_combined_payload is None:
            raise RuntimeError("No sample metrics produced for raw IMU latent diffusion export")
        best_future_payload = slice_payload(
            best_combined_payload,
            start_frame=context_frames,
            num_frames=prediction_frames,
            rebase_heading=False,
        )
        best_context_future_payload = stitch_future_payload_with_context(
            context_target_payload=target_context_future,
            future_payload=best_future_payload,
            context_frames=context_frames,
            future_frames=prediction_frames,
        )
        comparison = compare_payloads(
            left_payload=target_context_future,
            right_payload=best_context_future_payload,
            render_space=render_space,
        )
        window_dir = output_dir / (
            f"{rank:02d}__{row['participant']}__segment_{row['segment_id']}__start_{row['start_frame_20hz']}"
        )
        window_dir.mkdir(parents=True, exist_ok=True)
        save_payload_npz(window_dir / "target_context_future_8s.npz", target_context_future)
        save_payload_npz(window_dir / "predicted_combined_from_raw_imu_8s.npz", best_combined_payload)
        save_payload_npz(window_dir / "sample_context_future_8s.npz", best_context_future_payload)

        video_path = ""
        if not no_video:
            video_output_path = build_export_video_path(
                output_dir=output_dir,
                source_tag="raw-imu-ablation",
                participant=row["participant"],
                segment_id=row["segment_id"],
                start_frame_20hz=row["start_frame_20hz"],
                rank=rank,
                extra_tags=("hidden_256_blocks_12",),
            )
            render_payload_comparison_video(
                left_payload=target_context_future,
                right_payload=best_context_future_payload,
                output_path=video_output_path,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Pseudo-pose target",
                right_title="Raw IMU ablation",
                progress_label=f"{row['participant']} seg{row['segment_id']} start{row['start_frame_20hz']}",
                fps=fps,
                point_size=point_size,
                transition_frame=context_frames,
                pre_transition_label=f"PAST {context_frames / 20.0:.1f}S CONTEXT ON BOTH SIDES",
                post_transition_label=f"FUTURE {prediction_frames / 20.0:.1f}S: LEFT TARGET | RIGHT RAW-IMU PREDICTION",
                transition_banner="PREDICTION STARTS",
            )
            video_path = str(video_output_path)

        summary_rows.append(
            {
                "rank": int(rank),
                "participant": row["participant"],
                "segment_id": row["segment_id"],
                "pose_path": str(row["pose_path"]),
                "feature_path": str(row["feature_path"]),
                "start_frame_20hz": int(row["start_frame_20hz"]),
                "valid_frames": int(row["valid_frames"]),
                "packet_start_20hz": int(target_context_future.packet_counter[0]),
                "packet_end_20hz": int(target_context_future.packet_counter[-1]),
                "context_frames": int(context_frames),
                "prediction_frames": int(prediction_frames),
                "sample_count": int(sample_count),
                "selected_sample_index": int(best_sample_index),
                "selected_sample_recon_mpjpe": float(best_metrics["recon_mpjpe"]),
                "selected_sample_root_heading_error_deg": float(best_metrics["root_heading_error_deg"]),
                "selected_sample_jerk_error": float(best_metrics["jerk_error"]),
                "target_vs_sample_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                "target_vs_sample_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                "target_vs_sample_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                "target_vs_sample_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
                "window_dir": str(window_dir),
                "video_path": video_path,
            }
        )

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "window_index_csv": str(window_index_csv),
        "top_k": int(len(summary_rows)),
        "sample_count": int(sample_count),
        "rows": summary_rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export raw IMU latent diffusion sample videos")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=120)
    parser.add_argument("--prediction-frames", type=int, default=40)
    parser.add_argument("--participant", type=str, default=None)
    parser.add_argument("--segment-id", type=str, default=None)
    parser.add_argument("--center-start-frame", type=int, default=None)
    parser.add_argument("--render-space", choices=["root", "heading"], default="heading")
    parser.add_argument("--body-model", choices=["shell", "smal"], default="smal")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=root / "SMAL" / "wolf_alph3.pkl")
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--raw-imu-vae-checkpoint-override", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_raw_imu_latent_diffusion_sample_videos(
        checkpoint_path=args.checkpoint,
        window_index_csv=args.window_index_csv,
        output_dir=args.output_dir,
        top_k=args.top_k,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        context_frames=args.context_frames,
        prediction_frames=args.prediction_frames,
        participant=args.participant,
        segment_id=args.segment_id,
        center_start_frame=args.center_start_frame,
        render_space=args.render_space,
        body_model=args.body_model,
        show_skeleton_overlay=args.show_skeleton_overlay,
        smal_model=args.smal_model,
        smal_mapping=args.smal_mapping,
        smal_data=args.smal_data,
        smal_family_index=args.smal_family_index,
        fps=args.fps,
        point_size=args.point_size,
        device=args.device,
        no_video=args.no_video,
        raw_imu_vae_checkpoint_override=args.raw_imu_vae_checkpoint_override,
    )
    print(summary)


if __name__ == "__main__":
    main()
