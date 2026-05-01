#!/usr/bin/env python3
"""
Export 20Hz pseudo-pose targets from filled IMU segments.

The exported coordinates are derived from the same pose construction path used
by visualize_dog_skeleton.py, so downstream VAE/diffusion targets stay aligned
with the visual audit semantics.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from imu_new2_common import FLOAT_DTYPE, IMU_RATE_HZ, JOINT_ORDER
from visualize_dog_skeleton import (
    DEFAULT_GYR_MOTION_SCALE,
    build_heading_rotation_matrices,
    build_relative_positions,
    compute_pose_audit,
    downsample_motion_data,
    load_filled_motion_data,
    rotation_matrices_to_rot6d,
    stack_joint_is_interpolated,
)


PSEUDO_POSE_AUDIT_COLUMNS = (
    "participant",
    "segment_id",
    "feature_path",
    "source_dir",
    "output_path",
    "num_frames",
    "packet_start",
    "packet_end",
    "sample_rate_hz",
    "finite_ratio",
    "interp_ratio",
    "max_joint_interp_ratio",
    "worst_interp_joint",
    "left_right_sign_consistency",
    "distal_below_parent_consistency",
    "computed_preview_status",
    "audit_preview_status",
    "export_decision",
    "skip_reason",
)


def discover_feature_files(feature_root: Path) -> list[Path]:
    return sorted(feature_root.glob("*/segment_*.npz"), key=lambda path: (path.parent.name, path.stem))


def parse_feature_metadata(feature_path: Path) -> tuple[str, str]:
    participant = feature_path.parent.name
    stem = feature_path.stem
    if not stem.startswith("segment_"):
        raise ValueError(f"Expected feature filename segment_<id>.npz, got {feature_path.name}")
    segment_id = stem.removeprefix("segment_")
    if not participant or not segment_id:
        raise ValueError(f"Could not infer participant/segment from {feature_path}")
    return participant, segment_id


def load_status_csv(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None or not path.exists():
        return {}

    statuses: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            participant = str(row.get("participant", "")).strip()
            segment_id = str(row.get("segment_id", "")).strip()
            status = str(row.get("preview_status", row.get("status", ""))).strip().lower()
            if not participant or not segment_id or not status:
                continue
            statuses[(participant, segment_id)] = status
    return statuses


def load_review_allowlist(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()

    allowed: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            participant = str(row.get("participant", "")).strip()
            segment_id = str(row.get("segment_id", "")).strip()
            if participant and segment_id:
                allowed.add((participant, segment_id))
    return allowed


def select_packet_axis(
    *,
    source_packet_counter: np.ndarray,
    target_packet_counter: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    source_packet_counter = np.asarray(source_packet_counter, dtype=np.int64)
    target_packet_counter = np.asarray(target_packet_counter, dtype=np.int64)
    if np.array_equal(source_packet_counter, target_packet_counter):
        return arrays

    source_index_by_packet = {
        int(packet): index for index, packet in enumerate(source_packet_counter.tolist())
    }
    missing_packets = [int(packet) for packet in target_packet_counter.tolist() if int(packet) not in source_index_by_packet]
    if missing_packets:
        preview = missing_packets[:5]
        raise ValueError(f"Feature packet axis is not a subset of pseudo-pose axis; missing packets={preview}")

    selected_indices = np.asarray(
        [source_index_by_packet[int(packet)] for packet in target_packet_counter.tolist()],
        dtype=np.int64,
    )
    return {name: value[selected_indices] for name, value in arrays.items()}


def build_segment_pseudo_pose(
    *,
    segment_dir: Path,
    feature_path: Path,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
) -> dict[str, np.ndarray | float | dict[str, float | str]]:
    packet_counter, sensor_data_by_joint = load_filled_motion_data(segment_dir)
    packet_counter, sensor_data_by_joint, sample_rate_hz = downsample_motion_data(
        packet_counter=packet_counter,
        sensor_data_by_joint=sensor_data_by_joint,
        source_rate_hz=float(IMU_RATE_HZ),
        target_rate_hz=20.0,
    )

    relative_positions = build_relative_positions(
        sensor_data_by_joint=sensor_data_by_joint,
        neutral_pose_mode=neutral_pose_mode,
        gyr_motion_scale=gyr_motion_scale,
    ).astype(FLOAT_DTYPE)
    root_heading_6d = rotation_matrices_to_rot6d(
        build_heading_rotation_matrices(sensor_data_by_joint["stern"]["quat"])
    ).astype(FLOAT_DTYPE)
    is_interpolated = stack_joint_is_interpolated(sensor_data_by_joint)

    with np.load(feature_path, allow_pickle=False) as payload:
        feature_packet_counter = payload["packet_counter"].astype(np.int64)

    aligned = select_packet_axis(
        source_packet_counter=packet_counter,
        target_packet_counter=feature_packet_counter,
        arrays={
            "relative_positions": relative_positions,
            "root_heading_6d": root_heading_6d,
            "is_interpolated": is_interpolated,
        },
    )

    audit_metrics = compute_pose_audit(
        relative_positions=aligned["relative_positions"],
        is_interpolated=aligned["is_interpolated"].astype(bool),
    )

    return {
        "packet_counter": feature_packet_counter,
        "relative_positions": aligned["relative_positions"].astype(FLOAT_DTYPE),
        "root_heading_6d": aligned["root_heading_6d"].astype(FLOAT_DTYPE),
        "is_interpolated": aligned["is_interpolated"].astype(bool),
        "sample_rate_hz": float(sample_rate_hz),
        "audit_metrics": audit_metrics,
    }


def decide_export(
    *,
    status: str,
    key: tuple[str, str],
    review_allowlist: set[tuple[str, str]],
) -> tuple[bool, str]:
    if status == "pass":
        return True, ""
    if status == "review":
        if key in review_allowlist:
            return True, "review_allowlisted"
        return False, "review_not_allowlisted"
    if status == "fail":
        return False, "visual_audit_failed"
    return False, f"invalid_status:{status}"


def write_audit_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PSEUDO_POSE_AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_pseudo_pose_dataset(
    *,
    filled_root: Path,
    feature_root: Path,
    output_root: Path,
    audit_output: Path,
    visual_audit_path: Path | None,
    review_allowlist_path: Path | None,
    max_segments: int | None = None,
    neutral_pose_mode: str = "sequence-median",
    gyr_motion_scale: float = DEFAULT_GYR_MOTION_SCALE,
) -> dict[str, object]:
    filled_root = Path(filled_root)
    feature_root = Path(feature_root)
    output_root = Path(output_root)
    audit_output = Path(audit_output)

    visual_statuses = load_status_csv(visual_audit_path)
    review_allowlist = load_review_allowlist(review_allowlist_path)
    feature_files = discover_feature_files(feature_root)
    if max_segments is not None:
        feature_files = feature_files[:max_segments]

    rows: list[dict[str, object]] = []
    num_exported = 0
    num_errors = 0

    for feature_path in feature_files:
        participant, segment_id = parse_feature_metadata(feature_path)
        key = (participant, segment_id)
        segment_dir = filled_root / participant / segment_id
        output_path = output_root / participant / f"segment_{segment_id}.npz"
        row: dict[str, object] = {
            "participant": participant,
            "segment_id": segment_id,
            "feature_path": str(feature_path),
            "source_dir": str(segment_dir),
            "output_path": str(output_path),
            "num_frames": "",
            "packet_start": "",
            "packet_end": "",
            "sample_rate_hz": "",
            "finite_ratio": "",
            "interp_ratio": "",
            "max_joint_interp_ratio": "",
            "worst_interp_joint": "",
            "left_right_sign_consistency": "",
            "distal_below_parent_consistency": "",
            "computed_preview_status": "",
            "audit_preview_status": "",
            "export_decision": "skipped",
            "skip_reason": "",
        }

        try:
            segment_payload = build_segment_pseudo_pose(
                segment_dir=segment_dir,
                feature_path=feature_path,
                neutral_pose_mode=neutral_pose_mode,
                gyr_motion_scale=gyr_motion_scale,
            )
            packet_counter = segment_payload["packet_counter"]
            relative_positions = segment_payload["relative_positions"]
            root_heading_6d = segment_payload["root_heading_6d"]
            is_interpolated = segment_payload["is_interpolated"]
            sample_rate_hz = float(segment_payload["sample_rate_hz"])
            audit_metrics = dict(segment_payload["audit_metrics"])
            computed_status = str(audit_metrics["preview_status"]).lower()
            audit_status = visual_statuses.get(key, computed_status).lower()
            should_export, skip_reason = decide_export(
                status=audit_status,
                key=key,
                review_allowlist=review_allowlist,
            )

            row.update(
                {
                    "num_frames": int(packet_counter.shape[0]),
                    "packet_start": int(packet_counter[0]),
                    "packet_end": int(packet_counter[-1]),
                    "sample_rate_hz": sample_rate_hz,
                    "finite_ratio": float(audit_metrics["finite_ratio"]),
                    "interp_ratio": float(audit_metrics["interp_ratio"]),
                    "max_joint_interp_ratio": float(audit_metrics["max_joint_interp_ratio"]),
                    "worst_interp_joint": str(audit_metrics["worst_interp_joint"]),
                    "left_right_sign_consistency": float(audit_metrics["left_right_sign_consistency"]),
                    "distal_below_parent_consistency": float(audit_metrics["distal_below_parent_consistency"]),
                    "computed_preview_status": computed_status,
                    "audit_preview_status": audit_status,
                    "export_decision": "exported" if should_export else "skipped",
                    "skip_reason": skip_reason,
                }
            )

            if should_export:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output_path,
                    packet_counter=packet_counter.astype(np.int64),
                    relative_positions=relative_positions.astype(FLOAT_DTYPE),
                    root_heading_6d=root_heading_6d.astype(FLOAT_DTYPE),
                    joint_names=np.asarray(JOINT_ORDER),
                    is_interpolated=is_interpolated.astype(bool),
                )
                num_exported += 1
            elif output_path.exists():
                output_path.unlink()

        except Exception as exc:
            num_errors += 1
            row["skip_reason"] = f"error:{type(exc).__name__}:{exc}"
            if output_path.exists():
                output_path.unlink()

        rows.append(row)

    write_audit_csv(audit_output, rows)
    return {
        "num_feature_files": len(feature_files),
        "num_exported": num_exported,
        "num_skipped": len(feature_files) - num_exported,
        "num_errors": num_errors,
        "output_root": str(output_root),
        "audit_output": str(audit_output),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "Data" / "IMU_Only_20Hz_v1"
    parser = argparse.ArgumentParser(description="Export 20Hz pseudo-pose npz files from filled IMU data")
    parser.add_argument("--filled-root", type=Path, default=root / "Data" / "IMU_New2_Filled")
    parser.add_argument("--feature-root", type=Path, default=data_root / "features_20hz")
    parser.add_argument("--output-root", type=Path, default=data_root / "pseudo_pose_20hz")
    parser.add_argument("--audit-output", type=Path, default=data_root / "visual_audit" / "pseudo_pose_export_audit.csv")
    parser.add_argument("--visual-audit-path", type=Path, default=data_root / "visual_audit" / "pose_visualization_audit.csv")
    parser.add_argument("--review-allowlist-path", type=Path, default=data_root / "manifests" / "review_allowlist.csv")
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--neutral-pose-mode", choices=["first-frame", "sequence-median"], default="sequence-median")
    parser.add_argument("--gyr-motion-scale", type=float, default=DEFAULT_GYR_MOTION_SCALE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_pseudo_pose_dataset(
        filled_root=args.filled_root,
        feature_root=args.feature_root,
        output_root=args.output_root,
        audit_output=args.audit_output,
        visual_audit_path=args.visual_audit_path,
        review_allowlist_path=args.review_allowlist_path,
        max_segments=args.max_segments,
        neutral_pose_mode=args.neutral_pose_mode,
        gyr_motion_scale=args.gyr_motion_scale,
    )
    print(
        "Exported {num_exported}/{num_feature_files} pseudo-pose segments "
        "({num_errors} errors). Audit: {audit_output}".format(**summary)
    )


if __name__ == "__main__":
    main()
