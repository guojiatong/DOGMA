#!/usr/bin/env python3
"""
Export the same held-out latent diffusion windows for multiple checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from export_latent_diffusion_sample_videos import load_condition_window, pose_array_to_payload
from export_temporal_vae_failure_pack import (
    build_context_target_payload,
    compare_payloads,
    render_payload_comparison_video,
    save_payload_npz,
    slice_payload,
    stitch_future_payload_with_context,
)
from train_imu_masked_recon import write_json
from train_latent_diffusion import (
    build_sampling_generator,
    decode_latent_to_pose,
    load_latent_diffusion_checkpoint,
    normalize_condition_tensor,
    sample_latent_diffusion,
)
from train_temporal_vae import (
    compute_pose_recon_metrics,
    flatten_pose_window,
    read_pose_window_records_from_csv,
    resolve_position_smoothing_kernel,
)
from video_export_paths import build_export_video_path
from visualize_pose import load_pseudo_pose


SELECTED_WINDOW_COLUMNS = (
    "rank",
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
)

SUMMARY_COLUMNS = (
    "rank",
    "checkpoint_label",
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "valid_frames",
    "packet_start_20hz",
    "packet_end_20hz",
    "sample_count",
    "selected_sample_index",
    "selected_future_recon_mpjpe",
    "selected_future_root_heading_error_deg",
    "selected_future_jerk_error",
    "target_vs_sample_mean_mpjpe_mm",
    "target_vs_sample_max_mpjpe_mm",
    "target_vs_sample_mean_heading_error_deg",
    "target_vs_sample_max_heading_error_deg",
    "window_dir",
    "video_path",
)


@dataclass(frozen=True)
class SelectedWindow:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    start_frame_20hz: int
    valid_frames: int
    packet_start_20hz: int
    packet_end_20hz: int


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    checkpoint_path: Path


def parse_checkpoint_specs(spec_values: list[str]) -> list[CheckpointSpec]:
    specs: list[CheckpointSpec] = []
    for value in spec_values:
        if "::" not in value:
            raise ValueError(f"checkpoint spec must be LABEL::PATH, got {value}")
        label, path_text = value.split("::", 1)
        specs.append(CheckpointSpec(label=label.strip(), checkpoint_path=Path(path_text.strip())))
    if not specs:
        raise ValueError("At least one --checkpoint-spec is required")
    return specs


def infer_window_metadata(record: Any, *, prediction_frames: int) -> SelectedWindow:
    payload = load_pseudo_pose(record.pose_path)
    packet_counter = payload.packet_counter[record.start_frame : record.start_frame + prediction_frames]
    participant = record.pose_path.parent.name
    segment_id = record.pose_path.stem.replace("segment_", "")
    return SelectedWindow(
        participant=participant,
        segment_id=segment_id,
        pose_path=record.pose_path,
        feature_path=record.feature_path,
        start_frame_20hz=int(record.start_frame),
        valid_frames=int(prediction_frames),
        packet_start_20hz=int(packet_counter[0]),
        packet_end_20hz=int(packet_counter[-1]),
    )


def select_random_windows(
    *,
    window_index_csv: Path,
    context_frames: int,
    prediction_frames: int,
    top_k: int,
    seed: int,
) -> list[SelectedWindow]:
    records = read_pose_window_records_from_csv(window_index_csv)
    eligible = [
        infer_window_metadata(record, prediction_frames=prediction_frames)
        for record in records
        if record.start_frame >= context_frames and record.valid_frames >= prediction_frames
    ]
    if len(eligible) < top_k:
        raise ValueError(f"Requested top_k={top_k}, but only {len(eligible)} eligible windows exist")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(eligible), size=top_k, replace=False)
    selected = [eligible[int(index)] for index in np.sort(indices).tolist()]
    return selected


def write_selected_windows_csv(path: Path, rows: list[SelectedWindow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_WINDOW_COLUMNS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": int(rank),
                    "participant": row.participant,
                    "segment_id": row.segment_id,
                    "pose_path": str(row.pose_path),
                    "feature_path": str(row.feature_path),
                    "start_frame_20hz": int(row.start_frame_20hz),
                    "valid_frames": int(row.valid_frames),
                    "packet_start_20hz": int(row.packet_start_20hz),
                    "packet_end_20hz": int(row.packet_end_20hz),
                }
            )


def read_selected_windows_csv(path: Path) -> list[SelectedWindow]:
    rows: list[SelectedWindow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for payload in csv.DictReader(handle):
            rows.append(
                SelectedWindow(
                    participant=str(payload["participant"]),
                    segment_id=str(payload["segment_id"]),
                    pose_path=Path(payload["pose_path"]),
                    feature_path=Path(payload["feature_path"]),
                    start_frame_20hz=int(payload["start_frame_20hz"]),
                    valid_frames=int(payload["valid_frames"]),
                    packet_start_20hz=int(payload["packet_start_20hz"]),
                    packet_end_20hz=int(payload["packet_end_20hz"]),
                )
            )
    return rows


@torch.no_grad()
def export_multicheckpoint_latent_diffusion_samples(
    *,
    checkpoint_specs: list[CheckpointSpec],
    window_index_csv: Path,
    output_dir: Path,
    selected_windows_csv: Path | None = None,
    top_k: int = 6,
    sample_count: int = 10,
    sampling_seed: int = 23,
    selection_seed: int = 23,
    context_frames: int = 120,
    prediction_frames: int = 40,
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
    position_smoothing_kernel: str = "tri5",
    vae_checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    workspace_root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if smal_model is None:
        smal_model = workspace_root / "SMAL" / "wolf_alph3.pkl"

    selected_windows_path = output_dir / "selected_windows.csv"
    if selected_windows_csv is None:
        selected_windows = select_random_windows(
            window_index_csv=window_index_csv,
            context_frames=context_frames,
            prediction_frames=prediction_frames,
            top_k=top_k,
            seed=selection_seed,
        )
        write_selected_windows_csv(selected_windows_path, selected_windows)
    else:
        selected_windows = read_selected_windows_csv(Path(selected_windows_csv))
        if len(selected_windows) < top_k:
            raise ValueError(
                f"selected_windows_csv contains {len(selected_windows)} rows, smaller than requested top_k={top_k}"
            )
        selected_windows = selected_windows[:top_k]
        write_selected_windows_csv(selected_windows_path, selected_windows)

    all_summaries: list[dict[str, Any]] = []
    for checkpoint_spec in checkpoint_specs:
        checkpoint_output_dir = output_dir / checkpoint_spec.label
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)
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
            checkpoint_path=checkpoint_spec.checkpoint_path,
            device=device_resolved,
            vae_checkpoint_override=vae_checkpoint_override,
        )
        if int(config.past_frames) != int(context_frames) or int(config.future_window_frames) != int(prediction_frames):
            raise ValueError(
                "Requested context/prediction frames do not match checkpoint config "
                f"for {checkpoint_spec.label} ({context_frames}/{prediction_frames} vs {config.past_frames}/{config.future_window_frames})"
            )
        smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)
        latent_shape = (1, int(vae_config.window_frames // 8), int(vae_config.latent_dim))

        summary_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(selected_windows, start=1):
            target_payload_full = load_pseudo_pose(row.pose_path)
            target_window = slice_payload(
                target_payload_full,
                start_frame=row.start_frame_20hz,
                num_frames=prediction_frames,
                rebase_heading=True,
            )
            target_context_future = build_context_target_payload(
                full_payload=target_payload_full,
                future_start_frame=row.start_frame_20hz,
                context_frames=context_frames,
                future_frames=prediction_frames,
            )
            condition_np = load_condition_window(
                feature_path=row.feature_path,
                start_frame_20hz=row.start_frame_20hz,
                past_frames=context_frames,
            )
            condition = torch.from_numpy(condition_np[None]).to(device=device_resolved, dtype=torch.float32)
            condition = normalize_condition_tensor(condition, condition_mean, condition_std)

            target_future_pose_tensor = torch.from_numpy(
                flatten_pose_window(
                    target_window.relative_positions.astype(np.float32),
                    target_window.root_heading_6d.astype(np.float32),
                )[None]
            )

            best_index = 0
            best_metrics: dict[str, float] | None = None
            best_payload = None
            for sample_index in range(sample_count):
                generator = build_sampling_generator(
                    device=device_resolved,
                    base_seed=sampling_seed,
                    meta={
                        "participant": row.participant,
                        "segment_id": row.segment_id,
                        "start_frame_20hz": row.start_frame_20hz,
                        "packet_start_20hz": row.packet_start_20hz,
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
                sampled_pose_np = sampled_pose[0].detach().cpu().numpy().astype(np.float64)
                future_pose_np = sampled_pose_np[config.past_frames : config.past_frames + prediction_frames]
                future_payload = pose_array_to_payload(
                    base_payload=target_window,
                    pose_array=future_pose_np,
                )
                future_pose_tensor = torch.from_numpy(
                    flatten_pose_window(
                        future_payload.relative_positions.astype(np.float32),
                        future_payload.root_heading_6d.astype(np.float32),
                    )[None]
                )
                metrics = compute_pose_recon_metrics(
                    prediction=future_pose_tensor,
                    target=target_future_pose_tensor,
                )
                if best_metrics is None or float(metrics["recon_mpjpe"]) < float(best_metrics["recon_mpjpe"]):
                    best_index = sample_index
                    best_metrics = metrics
                    best_payload = future_payload

            assert best_metrics is not None and best_payload is not None
            sample_context_future = stitch_future_payload_with_context(
                context_target_payload=target_context_future,
                future_payload=best_payload,
                context_frames=context_frames,
                future_frames=prediction_frames,
            )
            comparison = compare_payloads(
                left_payload=target_context_future,
                right_payload=sample_context_future,
                render_space=render_space,
            )

            window_dir = checkpoint_output_dir / (
                f"{rank:02d}__{row.participant}__segment_{row.segment_id}__start_{row.start_frame_20hz}"
            )
            window_dir.mkdir(parents=True, exist_ok=True)
            save_payload_npz(window_dir / "target_window_rebased.npz", target_window)
            save_payload_npz(window_dir / "sample_prediction.npz", best_payload)
            save_payload_npz(window_dir / "target_context_future_12s.npz", target_context_future)
            save_payload_npz(window_dir / "sample_context_future_12s.npz", sample_context_future)

            video_path = ""
            if not no_video:
                video_output_path = build_export_video_path(
                    output_dir=checkpoint_output_dir,
                    source_tag="latent-diffusion",
                    participant=row.participant,
                    segment_id=row.segment_id,
                    start_frame_20hz=row.start_frame_20hz,
                    rank=rank,
                    extra_tags=(checkpoint_spec.label, "random", "mpjpe"),
                )
                render_payload_comparison_video(
                    left_payload=target_context_future,
                    right_payload=sample_context_future,
                    output_path=video_output_path,
                    render_space=render_space,
                    body_model=body_model,
                    show_skeleton_overlay=show_skeleton_overlay,
                    smal_model=smal_model,
                    smal_mapping=smal_mapping,
                    smal_data=smal_data,
                    smal_family_index=smal_family_index,
                    left_title="Pseudo-pose target",
                    right_title=f"Latent diffusion ({checkpoint_spec.label})",
                    progress_label=f"{row.participant} seg{row.segment_id} start{row.start_frame_20hz}",
                    fps=fps,
                    point_size=point_size,
                    transition_frame=context_frames,
                    pre_transition_label=f"PAST {context_frames / 20.0:.1f}S CONTEXT ON BOTH SIDES",
                    post_transition_label=f"FUTURE {prediction_frames / 20.0:.1f}S: LEFT TARGET | RIGHT PREDICTION",
                    transition_banner="PREDICTION STARTS",
                )
                video_path = str(video_output_path)

            summary_rows.append(
                {
                    "rank": int(rank),
                    "checkpoint_label": checkpoint_spec.label,
                    "participant": row.participant,
                    "segment_id": row.segment_id,
                    "pose_path": str(row.pose_path),
                    "feature_path": str(row.feature_path),
                    "start_frame_20hz": int(row.start_frame_20hz),
                    "valid_frames": int(row.valid_frames),
                    "packet_start_20hz": int(row.packet_start_20hz),
                    "packet_end_20hz": int(row.packet_end_20hz),
                    "sample_count": int(sample_count),
                    "selected_sample_index": int(best_index),
                    "selected_future_recon_mpjpe": float(best_metrics["recon_mpjpe"]),
                    "selected_future_root_heading_error_deg": float(best_metrics["root_heading_error_deg"]),
                    "selected_future_jerk_error": float(best_metrics["jerk_error"]),
                    "target_vs_sample_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                    "target_vs_sample_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                    "target_vs_sample_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                    "target_vs_sample_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
                    "window_dir": str(window_dir),
                    "video_path": video_path,
                }
            )

        with (checkpoint_output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(summary_rows)
        checkpoint_summary = {
            "checkpoint_label": checkpoint_spec.label,
            "checkpoint_path": str(checkpoint_spec.checkpoint_path),
            "top_k": len(summary_rows),
            "rows": summary_rows,
        }
        write_json(checkpoint_output_dir / "comparison_summary.json", checkpoint_summary)
        all_summaries.append(checkpoint_summary)

    merged_summary = {
        "window_index_csv": str(window_index_csv),
        "selection_seed": int(selection_seed),
        "sampling_seed": int(sampling_seed),
        "top_k": int(top_k),
        "checkpoint_labels": [spec.label for spec in checkpoint_specs],
        "checkpoint_count": len(checkpoint_specs),
        "selected_windows_csv": str(selected_windows_path),
        "checkpoint_summaries": all_summaries,
    }
    write_json(output_dir / "multicheckpoint_summary.json", merged_summary)
    return merged_summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export the same held-out latent diffusion windows for multiple checkpoints")
    parser.add_argument("--checkpoint-spec", type=str, action="append", required=True, help="LABEL::/abs/path/to/best.pt")
    parser.add_argument("--window-index-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selected-windows-csv", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sampling-seed", type=int, default=23)
    parser.add_argument("--selection-seed", type=int, default=23)
    parser.add_argument("--context-frames", type=int, default=120)
    parser.add_argument("--prediction-frames", type=int, default=40)
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
    parser.add_argument("--position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_multicheckpoint_latent_diffusion_samples(
        checkpoint_specs=parse_checkpoint_specs(args.checkpoint_spec),
        window_index_csv=args.window_index_csv,
        output_dir=args.output_dir,
        selected_windows_csv=args.selected_windows_csv,
        top_k=args.top_k,
        sample_count=args.sample_count,
        sampling_seed=args.sampling_seed,
        selection_seed=args.selection_seed,
        context_frames=args.context_frames,
        prediction_frames=args.prediction_frames,
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
        position_smoothing_kernel=args.position_smoothing_kernel,
        vae_checkpoint_override=args.vae_checkpoint,
    )
    print(summary["multicheckpoint_summary"] if "multicheckpoint_summary" in summary else summary["output_dir"] if "output_dir" in summary else summary)


if __name__ == "__main__":
    main()
