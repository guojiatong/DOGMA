#!/usr/bin/env python3
"""
Export curated temporal VAE reconstruction videos on walking-like windows with
better pseudo-pose ground truth quality.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_temporal_vae import load_temporal_vae_checkpoint
from export_temporal_vae_failure_pack import (
    compare_payloads,
    reconstruct_payload_window,
    render_payload_comparison_video,
    resolve_workspace_path,
    save_payload_npz,
    slice_payload,
)
from train_imu_masked_recon import write_json
from train_temporal_vae import resolve_position_smoothing_kernel
from video_export_paths import build_export_video_path
from visualize_pose import compute_motion_metrics, load_pseudo_pose


SUMMARY_COLUMNS = (
    "rank",
    "participant",
    "segment_id",
    "aligned_path",
    "start_frame_20hz",
    "end_frame_20hz",
    "packet_start_20hz",
    "packet_end_20hz",
    "walking_like_fraction",
    "walking_forward_fraction",
    "trotting_fraction",
    "running_fraction",
    "stopped_fraction",
    "mean_foot_motion",
    "mean_joint_motion",
    "p95_joint_step",
    "p95_joint_jerk",
    "p95_root_heading_delta_deg",
    "quality_score",
    "recon_mean_mpjpe_mm",
    "recon_max_mpjpe_mm",
    "recon_mean_heading_error_deg",
    "recon_max_heading_error_deg",
    "window_dir",
    "video_path",
)


@dataclass(frozen=True)
class CuratedWindow:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    aligned_path: Path
    start_frame_20hz: int
    valid_frames: int
    packet_start_20hz: int
    packet_end_20hz: int
    walking_like_fraction: float
    walking_forward_fraction: float
    trotting_fraction: float
    running_fraction: float
    stopped_fraction: float
    mean_joint_motion: float
    mean_foot_motion: float
    p95_joint_step: float
    p95_joint_jerk: float
    p95_root_heading_delta_deg: float
    quality_score: float


def parse_projection_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            participant_id, imu_participant = parts
            if imu_participant:
                mapping[participant_id] = imu_participant
    return mapping


def load_vae_segment_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame = pd.read_csv(path)
    selected = frame[frame["use_for_temporal_vae"] == 1].copy()
    manifest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected.to_dict(orient="records"):
        manifest[(str(row["participant"]), str(row["segment_id"]))] = row
    return manifest


def resolve_aligned_segment_id(aligned_payload: dict[str, Any], aligned_path: Path) -> str:
    meta = aligned_payload.get("meta", {})
    segment_id = meta.get("segment_id")
    if segment_id is not None and str(segment_id) != "":
        return str(segment_id)
    stem = aligned_path.stem
    if "__segment_" in stem:
        return stem.rsplit("__segment_", 1)[1]
    raise ValueError(f"Unable to resolve segment_id from {aligned_path}")


def compute_label_fractions(
    *,
    labels_40hz: np.ndarray,
    label_vocab: list[str],
) -> dict[str, float]:
    if labels_40hz.ndim != 2:
        raise ValueError(f"labels_40hz must be 2D, got {labels_40hz.shape}")
    if labels_40hz.shape[0] == 0:
        return {
            "walking_like_fraction": 0.0,
            "walking_forward_fraction": 0.0,
            "trotting_fraction": 0.0,
            "running_fraction": 0.0,
            "stopped_fraction": 0.0,
        }
    vocab_index = {label: index for index, label in enumerate(label_vocab)}

    def _mean_for(label: str) -> float:
        index = vocab_index.get(label)
        if index is None:
            return 0.0
        return float(np.asarray(labels_40hz[:, index], dtype=np.float64).mean())

    walking_forward_fraction = _mean_for("D_WALKING_FORWARD")
    trotting_fraction = _mean_for("D_TROTTING")
    running_fraction = _mean_for("D_RUNNING") + _mean_for("D_PLAY_RUNNING")
    stopped_fraction = _mean_for("D_STOPPED")

    walking_like_indices = [
        vocab_index[label]
        for label in ("D_WALKING_FORWARD", "D_TROTTING", "D_RUNNING", "D_PLAY_RUNNING")
        if label in vocab_index
    ]
    if walking_like_indices:
        walking_like_fraction = float(np.any(labels_40hz[:, walking_like_indices] > 0, axis=1).mean())
    else:
        walking_like_fraction = 0.0
    return {
        "walking_like_fraction": walking_like_fraction,
        "walking_forward_fraction": walking_forward_fraction,
        "trotting_fraction": trotting_fraction,
        "running_fraction": running_fraction,
        "stopped_fraction": stopped_fraction,
    }


def score_candidate(
    *,
    label_fractions: dict[str, float],
    motion_metrics: dict[str, float | str],
) -> float:
    return float(
        6.0 * label_fractions["walking_forward_fraction"]
        + 3.0 * label_fractions["walking_like_fraction"]
        + 1.5 * label_fractions["trotting_fraction"]
        - 1.0 * label_fractions["running_fraction"]
        - 3.0 * label_fractions["stopped_fraction"]
        + 45.0 * float(motion_metrics["mean_foot_motion"])
        + 18.0 * float(motion_metrics["mean_joint_motion"])
        - 20.0 * float(motion_metrics["p95_joint_jerk"])
        - 0.10 * float(motion_metrics["p95_root_heading_delta_deg"])
    )


def collect_candidate_windows(
    *,
    aligned_root: Path,
    projection_mapping: dict[str, str],
    vae_segment_manifest: dict[tuple[str, str], dict[str, Any]],
    window_frames: int,
    stride_frames: int,
    min_label_overlap_frames: int,
    min_walking_like_fraction: float,
    min_walking_forward_fraction: float,
    max_running_fraction: float,
    max_stopped_fraction: float,
    min_mean_foot_motion: float,
    max_p95_joint_jerk: float,
    max_p95_root_heading_delta_deg: float,
) -> list[CuratedWindow]:
    candidates: list[CuratedWindow] = []
    for aligned_path in sorted(aligned_root.rglob("*.pkl")):
        participant_id = aligned_path.parent.name
        participant = projection_mapping.get(participant_id, "")
        if not participant:
            continue
        with aligned_path.open("rb") as handle:
            aligned_payload = pickle.load(handle)

        segment_id = resolve_aligned_segment_id(aligned_payload, aligned_path)
        manifest_row = vae_segment_manifest.get((participant, segment_id))
        if manifest_row is None:
            continue

        pose_path = resolve_workspace_path(Path(str(manifest_row["pose_path"])), Path(__file__).resolve().parents[1])
        feature_path = resolve_workspace_path(Path(str(manifest_row["feature_path"])), Path(__file__).resolve().parents[1])
        pseudo_payload = load_pseudo_pose(pose_path)
        if pseudo_payload.packet_counter.shape[0] < window_frames:
            continue

        aligned_packet_counter = np.asarray(aligned_payload["packet_counter"], dtype=np.int64)
        labels_40hz = np.asarray(aligned_payload["labels_40hz"], dtype=np.uint8)
        label_vocab = [str(label) for label in aligned_payload["label_vocab"]]

        for start_frame in range(0, pseudo_payload.packet_counter.shape[0] - window_frames + 1, stride_frames):
            packet_start = int(pseudo_payload.packet_counter[start_frame])
            packet_end = int(pseudo_payload.packet_counter[start_frame + window_frames - 1])
            overlap_mask = (aligned_packet_counter >= packet_start) & (aligned_packet_counter <= packet_end)
            overlap_frames = int(overlap_mask.sum())
            if overlap_frames < min_label_overlap_frames:
                continue

            label_fractions = compute_label_fractions(
                labels_40hz=labels_40hz[overlap_mask],
                label_vocab=label_vocab,
            )
            if label_fractions["walking_like_fraction"] < min_walking_like_fraction:
                continue
            if label_fractions["walking_forward_fraction"] < min_walking_forward_fraction:
                continue
            if label_fractions["running_fraction"] > max_running_fraction:
                continue
            if label_fractions["stopped_fraction"] > max_stopped_fraction:
                continue

            gt_window = slice_payload(
                pseudo_payload,
                start_frame=int(start_frame),
                num_frames=int(window_frames),
                rebase_heading=True,
            )
            motion_metrics = compute_motion_metrics(gt_window)
            if float(motion_metrics["mean_foot_motion"]) < min_mean_foot_motion:
                continue
            if float(motion_metrics["p95_joint_jerk"]) > max_p95_joint_jerk:
                continue
            if float(motion_metrics["p95_root_heading_delta_deg"]) > max_p95_root_heading_delta_deg:
                continue

            candidates.append(
                CuratedWindow(
                    participant=participant,
                    segment_id=str(segment_id),
                    pose_path=pose_path,
                    feature_path=feature_path,
                    aligned_path=aligned_path,
                    start_frame_20hz=int(start_frame),
                    valid_frames=int(window_frames),
                    packet_start_20hz=packet_start,
                    packet_end_20hz=packet_end,
                    walking_like_fraction=float(label_fractions["walking_like_fraction"]),
                    walking_forward_fraction=float(label_fractions["walking_forward_fraction"]),
                    trotting_fraction=float(label_fractions["trotting_fraction"]),
                    running_fraction=float(label_fractions["running_fraction"]),
                    stopped_fraction=float(label_fractions["stopped_fraction"]),
                    mean_joint_motion=float(motion_metrics["mean_joint_motion"]),
                    mean_foot_motion=float(motion_metrics["mean_foot_motion"]),
                    p95_joint_step=float(motion_metrics["p95_joint_step"]),
                    p95_joint_jerk=float(motion_metrics["p95_joint_jerk"]),
                    p95_root_heading_delta_deg=float(motion_metrics["p95_root_heading_delta_deg"]),
                    quality_score=score_candidate(label_fractions=label_fractions, motion_metrics=motion_metrics),
                )
            )
    return candidates


def select_curated_windows(
    candidates: list[CuratedWindow],
    *,
    top_k: int,
    max_per_participant: int,
    min_start_distance_frames: int,
) -> list[CuratedWindow]:
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (
            row.quality_score,
            row.walking_forward_fraction,
            row.walking_like_fraction,
            row.mean_foot_motion,
            -row.p95_joint_jerk,
        ),
        reverse=True,
    )
    selected: list[CuratedWindow] = []
    counts_by_participant: dict[str, int] = {}
    starts_by_segment: dict[tuple[str, str], list[int]] = {}
    for candidate in sorted_candidates:
        if max_per_participant > 0 and counts_by_participant.get(candidate.participant, 0) >= max_per_participant:
            continue
        key = (candidate.participant, candidate.segment_id)
        previous_starts = starts_by_segment.get(key, [])
        if any(abs(candidate.start_frame_20hz - previous) < min_start_distance_frames for previous in previous_starts):
            continue
        selected.append(candidate)
        counts_by_participant[candidate.participant] = counts_by_participant.get(candidate.participant, 0) + 1
        starts_by_segment.setdefault(key, []).append(candidate.start_frame_20hz)
        if len(selected) >= top_k:
            break
    return selected


def export_curated_reconstruction_videos(
    *,
    checkpoint_path: Path,
    aligned_root: Path,
    projection_mapping_path: Path,
    vae_segment_manifest_path: Path,
    output_dir: Path,
    top_k: int,
    window_frames: int,
    stride_frames: int,
    min_start_distance_frames: int,
    max_per_participant: int,
    min_label_overlap_frames: int,
    min_walking_like_fraction: float,
    min_walking_forward_fraction: float,
    max_running_fraction: float,
    max_stopped_fraction: float,
    min_mean_foot_motion: float,
    max_p95_joint_jerk: float,
    max_p95_root_heading_delta_deg: float,
    render_space: str,
    body_model: str,
    show_skeleton_overlay: bool,
    smal_model: Path,
    smal_mapping: Path | None,
    smal_data: Path | None,
    smal_family_index: int,
    fps: int,
    point_size: float,
    device: str,
    no_video: bool,
    position_smoothing_kernel: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    projection_mapping = parse_projection_mapping(projection_mapping_path)
    vae_segment_manifest = load_vae_segment_manifest(vae_segment_manifest_path)
    candidates = collect_candidate_windows(
        aligned_root=aligned_root,
        projection_mapping=projection_mapping,
        vae_segment_manifest=vae_segment_manifest,
        window_frames=window_frames,
        stride_frames=stride_frames,
        min_label_overlap_frames=min_label_overlap_frames,
        min_walking_like_fraction=min_walking_like_fraction,
        min_walking_forward_fraction=min_walking_forward_fraction,
        max_running_fraction=max_running_fraction,
        max_stopped_fraction=max_stopped_fraction,
        min_mean_foot_motion=min_mean_foot_motion,
        max_p95_joint_jerk=max_p95_joint_jerk,
        max_p95_root_heading_delta_deg=max_p95_root_heading_delta_deg,
    )
    if not candidates:
        raise ValueError("No curated walking-like windows matched the requested thresholds")
    selected = select_curated_windows(
        candidates,
        top_k=top_k,
        max_per_participant=max_per_participant,
        min_start_distance_frames=min_start_distance_frames,
    )
    if not selected:
        raise ValueError("No curated windows selected after de-duplication")

    device_resolved = torch.device(device) if device != "auto" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, pose_mean, pose_std, checkpoint = load_temporal_vae_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device_resolved,
    )
    smoothing_kernel = resolve_position_smoothing_kernel(position_smoothing_kernel)

    summary_rows: list[dict[str, Any]] = []
    for rank, window in enumerate(selected, start=1):
        full_payload = load_pseudo_pose(window.pose_path)
        gt_window = slice_payload(
            full_payload,
            start_frame=window.start_frame_20hz,
            num_frames=window.valid_frames,
            rebase_heading=True,
        )
        reconstruction = reconstruct_payload_window(
            model=model,
            normalize_pose=config.normalize_pose,
            pose_mean=pose_mean,
            pose_std=pose_std,
            target_payload=gt_window,
            device=device_resolved,
            position_smoothing_kernel=smoothing_kernel,
        )
        comparison = compare_payloads(
            left_payload=gt_window,
            right_payload=reconstruction,
            render_space=render_space,
        )

        window_dir = output_dir / f"{rank:02d}__{window.participant}__segment_{window.segment_id}__start_{window.start_frame_20hz}"
        save_payload_npz(window_dir / "pseudo_gt_window_rebased.npz", gt_window)
        save_payload_npz(window_dir / "vae_reconstruction.npz", reconstruction)
        video_path = build_export_video_path(
            output_dir=output_dir,
            source_tag="temporal-vae-curated-recon",
            participant=window.participant,
            segment_id=window.segment_id,
            start_frame_20hz=window.start_frame_20hz,
            rank=rank,
            extra_tags=("walking-like",),
        )
        if not no_video:
            render_payload_comparison_video(
                left_payload=gt_window,
                right_payload=reconstruction,
                output_path=video_path,
                render_space=render_space,
                body_model=body_model,
                show_skeleton_overlay=show_skeleton_overlay,
                smal_model=smal_model,
                smal_mapping=smal_mapping,
                smal_data=smal_data,
                smal_family_index=smal_family_index,
                left_title="Pseudo-pose target",
                right_title="Temporal VAE reconstruction",
                progress_label=f"Rendering curated temporal VAE reconstruction #{rank}",
                fps=fps,
                point_size=point_size,
            )

        window_summary = {
            **asdict(window),
            "pose_path": str(window.pose_path),
            "feature_path": str(window.feature_path),
            "aligned_path": str(window.aligned_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "position_smoothing_kernel": position_smoothing_kernel,
            "comparison": {
                "mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                "max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                "mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                "max_heading_error_deg": float(comparison["max_heading_error_deg"]),
            },
        }
        write_json(window_dir / "window_summary.json", window_summary)

        summary_rows.append(
            {
                "rank": rank,
                "participant": window.participant,
                "segment_id": window.segment_id,
                "aligned_path": str(window.aligned_path),
                "start_frame_20hz": window.start_frame_20hz,
                "end_frame_20hz": window.start_frame_20hz + window.valid_frames - 1,
                "packet_start_20hz": window.packet_start_20hz,
                "packet_end_20hz": window.packet_end_20hz,
                "walking_like_fraction": window.walking_like_fraction,
                "walking_forward_fraction": window.walking_forward_fraction,
                "trotting_fraction": window.trotting_fraction,
                "running_fraction": window.running_fraction,
                "stopped_fraction": window.stopped_fraction,
                "mean_foot_motion": window.mean_foot_motion,
                "mean_joint_motion": window.mean_joint_motion,
                "p95_joint_step": window.p95_joint_step,
                "p95_joint_jerk": window.p95_joint_jerk,
                "p95_root_heading_delta_deg": window.p95_root_heading_delta_deg,
                "quality_score": window.quality_score,
                "recon_mean_mpjpe_mm": float(comparison["mean_mpjpe_m"] * 1000.0),
                "recon_max_mpjpe_mm": float(comparison["max_mpjpe_m"] * 1000.0),
                "recon_mean_heading_error_deg": float(comparison["mean_heading_error_deg"]),
                "recon_max_heading_error_deg": float(comparison["max_heading_error_deg"]),
                "window_dir": str(window_dir),
                "video_path": "" if no_video else str(video_path),
            }
        )

    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "aligned_root": str(aligned_root),
        "projection_mapping_path": str(projection_mapping_path),
        "vae_segment_manifest_path": str(vae_segment_manifest_path),
        "window_frames": int(window_frames),
        "stride_frames": int(stride_frames),
        "top_k": len(summary_rows),
        "rows": summary_rows,
    }
    write_json(output_dir / "comparison_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export curated walking-like temporal VAE reconstruction videos")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--aligned-root", type=Path, default=root / "Data" / "aligned_actions_v1")
    parser.add_argument("--projection-mapping", type=Path, default=root / "Data" / "projection_folder.txt")
    parser.add_argument(
        "--vae-segment-manifest",
        type=Path,
        default=root / "Data" / "IMU_Only_20Hz_v1" / "manifests" / "vae_segment_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--stride-frames", type=int, default=120)
    parser.add_argument("--min-start-distance-frames", type=int, default=240)
    parser.add_argument("--max-per-participant", type=int, default=2)
    parser.add_argument("--min-label-overlap-frames", type=int, default=400)
    parser.add_argument("--min-walking-like-fraction", type=float, default=0.80)
    parser.add_argument("--min-walking-forward-fraction", type=float, default=0.60)
    parser.add_argument("--max-running-fraction", type=float, default=0.20)
    parser.add_argument("--max-stopped-fraction", type=float, default=0.10)
    parser.add_argument("--min-mean-foot-motion", type=float, default=0.012)
    parser.add_argument("--max-p95-joint-jerk", type=float, default=0.045)
    parser.add_argument("--max-p95-root-heading-delta-deg", type=float, default=6.0)
    parser.add_argument("--render-space", choices=("root", "heading"), default="heading")
    parser.add_argument("--body-model", choices=("shell", "smal"), default="smal")
    parser.add_argument("--show-skeleton-overlay", action="store_true")
    parser.add_argument("--smal-model", type=Path, default=root / "SMAL" / "wolf_alph3.pkl")
    parser.add_argument("--smal-mapping", type=Path, default=None)
    parser.add_argument("--smal-data", type=Path, default=None)
    parser.add_argument("--smal-family-index", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--point-size", type=float, default=42.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--position-smoothing-kernel", choices=("none", "tri3", "tri5"), default="tri5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_curated_reconstruction_videos(
        checkpoint_path=args.checkpoint,
        aligned_root=args.aligned_root,
        projection_mapping_path=args.projection_mapping,
        vae_segment_manifest_path=args.vae_segment_manifest,
        output_dir=args.output_dir,
        top_k=args.top_k,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        min_start_distance_frames=args.min_start_distance_frames,
        max_per_participant=args.max_per_participant,
        min_label_overlap_frames=args.min_label_overlap_frames,
        min_walking_like_fraction=args.min_walking_like_fraction,
        min_walking_forward_fraction=args.min_walking_forward_fraction,
        max_running_fraction=args.max_running_fraction,
        max_stopped_fraction=args.max_stopped_fraction,
        min_mean_foot_motion=args.min_mean_foot_motion,
        max_p95_joint_jerk=args.max_p95_joint_jerk,
        max_p95_root_heading_delta_deg=args.max_p95_root_heading_delta_deg,
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
    )
    print(summary)


if __name__ == "__main__":
    main()
