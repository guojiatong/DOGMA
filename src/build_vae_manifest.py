#!/usr/bin/env python3
"""
Build participant-level train/val/test manifests for pseudo-pose temporal VAE.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from build_imu_only_dataset import (
    TEST_PARTICIPANTS,
    TRAIN_PARTICIPANTS,
    VAL_PARTICIPANTS,
    determine_split,
    write_csv,
)


SEGMENT_MANIFEST_COLUMNS = (
    "participant",
    "segment_id",
    "source_dir",
    "feature_path",
    "pose_path",
    "has_all_10_sensors",
    "has_feature_file",
    "has_pose_file",
    "num_frames_20hz",
    "packet_start_40hz",
    "packet_end_40hz",
    "duration_sec_20hz",
    "interp_ratio_20hz",
    "max_joint_interp_ratio_20hz",
    "worst_interp_joint",
    "split",
    "export_decision",
    "skip_reason",
    "computed_preview_status",
    "audit_preview_status",
    "visual_full_preview_status",
    "visual_selected_preview_status",
    "selected_p95_joint_step",
    "selected_max_joint_step",
    "selected_p95_joint_jerk",
    "selected_max_joint_jerk",
    "selected_p95_root_heading_delta_deg",
    "selected_max_root_heading_delta_deg",
    "num_windows",
    "use_for_temporal_vae",
    "use_for_temporal_vae_reason",
)

WINDOW_INDEX_COLUMNS = (
    "participant",
    "segment_id",
    "pose_path",
    "feature_path",
    "start_frame_20hz",
    "end_frame_20hz",
    "packet_start_40hz",
    "packet_end_40hz",
    "interp_ratio",
    "task",
)


def normalize_key(participant: str, segment_id: str) -> tuple[str, str]:
    return str(participant).strip(), str(segment_id).strip()


def read_csv_by_key(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            normalize_key(row.get("participant", ""), row.get("segment_id", "")): row
            for row in reader
        }


def parse_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return int(float(text))


def parse_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return float(text)


def parse_status(value: object) -> str:
    text = str(value).strip().lower()
    return text if text and text != "nan" else ""


def sort_segment_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(row: dict[str, object]) -> tuple[str, int | str]:
        segment_id = str(row["segment_id"])
        return (
            str(row["participant"]),
            int(segment_id) if segment_id.isdigit() else segment_id,
        )

    return sorted(rows, key=sort_key)


def load_pose_window_source(pose_path: Path) -> dict[str, np.ndarray | int | float | str]:
    with np.load(pose_path, allow_pickle=False) as payload:
        packet_counter = payload["packet_counter"].astype(np.int64)
        is_interpolated = payload["is_interpolated"].astype(bool)
        joint_names = [str(name) for name in payload["joint_names"].tolist()]

    if packet_counter.ndim != 1:
        raise ValueError(f"packet_counter must be 1D in {pose_path}")
    if is_interpolated.shape != (packet_counter.shape[0], len(joint_names)):
        raise ValueError(f"is_interpolated shape mismatch in {pose_path}")

    joint_interp_ratio = is_interpolated.mean(axis=0) if is_interpolated.size else np.zeros((len(joint_names),))
    worst_joint_index = int(np.argmax(joint_interp_ratio)) if joint_interp_ratio.size else 0
    return {
        "packet_counter": packet_counter,
        "is_interpolated": is_interpolated,
        "num_frames_20hz": int(packet_counter.shape[0]),
        "packet_start_40hz": int(packet_counter[0]) if packet_counter.size else 0,
        "packet_end_40hz": int(packet_counter[-1]) if packet_counter.size else 0,
        "duration_sec_20hz": float(packet_counter.shape[0]) / 20.0,
        "interp_ratio_20hz": float(is_interpolated.mean()) if is_interpolated.size else 0.0,
        "max_joint_interp_ratio_20hz": float(joint_interp_ratio[worst_joint_index]) if joint_interp_ratio.size else 0.0,
        "worst_interp_joint": joint_names[worst_joint_index] if joint_names else "",
    }


def decide_segment_usage(
    *,
    has_all_10_sensors: bool,
    split: str,
    export_decision: str,
    has_feature_file: bool,
    has_pose_file: bool,
    computed_preview_status: str,
    audit_preview_status: str,
    visual_full_preview_status: str,
    visual_selected_preview_status: str,
    num_frames_20hz: int,
    window_frames: int,
    interp_ratio_20hz: float,
    max_joint_interp_ratio_20hz: float,
    segment_interp_threshold: float,
    segment_joint_interp_threshold: float,
) -> tuple[bool, str]:
    if not has_all_10_sensors:
        return False, "missing_required_sensors"
    if split not in {"train", "val", "test"}:
        return False, "split_unused"
    if export_decision != "exported":
        return False, f"export_decision:{export_decision or 'missing'}"
    if not has_feature_file:
        return False, "missing_feature_file"
    if not has_pose_file:
        return False, "missing_pose_file"
    if computed_preview_status != "pass":
        return False, f"computed_preview_status:{computed_preview_status or 'missing'}"
    if audit_preview_status != "pass":
        return False, f"audit_preview_status:{audit_preview_status or 'missing'}"
    if visual_full_preview_status != "pass":
        return False, f"visual_full_preview_status:{visual_full_preview_status or 'missing'}"
    if visual_selected_preview_status != "pass":
        return False, f"visual_selected_preview_status:{visual_selected_preview_status or 'missing'}"
    if num_frames_20hz < window_frames:
        return False, "segment_too_short"
    if interp_ratio_20hz > segment_interp_threshold:
        return False, "segment_interp_too_high"
    if max_joint_interp_ratio_20hz > segment_joint_interp_threshold:
        return False, "segment_joint_interp_too_high"
    return True, "ok"


def generate_pose_window_rows(
    *,
    pose_path: Path,
    feature_path: Path,
    participant: str,
    segment_id: str,
    window_frames: int,
    stride_frames: int,
    interp_threshold: float,
) -> list[dict[str, object]]:
    payload = load_pose_window_source(pose_path)
    packet_counter = np.asarray(payload["packet_counter"], dtype=np.int64)
    is_interpolated = np.asarray(payload["is_interpolated"], dtype=bool)
    total_frames = int(packet_counter.shape[0])
    if total_frames < window_frames:
        return []

    max_start = total_frames - window_frames
    starts = list(range(0, max_start + 1, stride_frames))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)

    rows: list[dict[str, object]] = []
    for start_frame in starts:
        end_frame = start_frame + window_frames
        window_interp_ratio = float(is_interpolated[start_frame:end_frame].mean())
        if window_interp_ratio > interp_threshold:
            continue
        rows.append(
            {
                "participant": participant,
                "segment_id": segment_id,
                "pose_path": str(pose_path),
                "feature_path": str(feature_path),
                "start_frame_20hz": int(start_frame),
                "end_frame_20hz": int(end_frame - 1),
                "packet_start_40hz": int(packet_counter[start_frame]),
                "packet_end_40hz": int(packet_counter[end_frame - 1]),
                "interp_ratio": window_interp_ratio,
                "task": "temporal_vae",
            }
        )
    return rows


def build_vae_manifest_artifacts(
    *,
    data_root: Path,
    manifests_root: Path | None = None,
    window_frames: int = 240,
    train_stride_frames: int = 20,
    eval_stride_frames: int = 120,
    segment_interp_threshold: float = 0.20,
    segment_joint_interp_threshold: float = 0.20,
    train_window_interp_threshold: float = 0.20,
    eval_window_interp_threshold: float = 0.05,
) -> dict[str, object]:
    data_root = Path(data_root)
    manifests_root = data_root / "manifests" if manifests_root is None else Path(manifests_root)
    manifests_root.mkdir(parents=True, exist_ok=True)

    export_audit_path = data_root / "visual_audit" / "pseudo_pose_export_audit.csv"
    visual_audit_path = data_root / "visual_audit" / "pseudo_pose_visual_audit.csv"
    raw_segment_manifest_path = data_root / "manifests" / "segment_manifest.csv"
    export_rows_by_key = read_csv_by_key(export_audit_path)
    visual_rows_by_key = read_csv_by_key(visual_audit_path)
    raw_segment_rows_by_key = read_csv_by_key(raw_segment_manifest_path) if raw_segment_manifest_path.exists() else {}

    segment_rows: list[dict[str, object]] = []
    window_rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    for key in sorted(export_rows_by_key.keys(), key=lambda item: (item[0], parse_int(item[1], default=10**9))):
        export_row = export_rows_by_key[key]
        visual_row = visual_rows_by_key.get(key, {})
        raw_segment_row = raw_segment_rows_by_key.get(key, {})
        participant, segment_id = key
        split = determine_split(participant)
        has_all_10_sensors = parse_int(raw_segment_row.get("has_all_10_sensors"), default=1) == 1
        source_dir = str(export_row.get("source_dir", "")).strip()
        feature_path_str = str(export_row.get("feature_path", "")).strip()
        pose_path_str = str(export_row.get("output_path", "")).strip()
        feature_path = Path(feature_path_str) if feature_path_str else None
        pose_path = Path(pose_path_str) if pose_path_str else None
        has_feature_file = feature_path is not None and feature_path.exists()
        has_pose_file = pose_path is not None and pose_path.exists()

        num_frames_20hz = parse_int(export_row.get("num_frames"))
        packet_start_40hz = parse_int(export_row.get("packet_start"))
        packet_end_40hz = parse_int(export_row.get("packet_end"))
        duration_sec_20hz = float(num_frames_20hz) / 20.0 if num_frames_20hz > 0 else 0.0
        interp_ratio_20hz = parse_float(export_row.get("interp_ratio"))
        max_joint_interp_ratio_20hz = parse_float(export_row.get("max_joint_interp_ratio"))
        worst_interp_joint = str(export_row.get("worst_interp_joint", "")).strip()
        if has_pose_file and pose_path is not None:
            pose_summary = load_pose_window_source(pose_path)
            num_frames_20hz = int(pose_summary["num_frames_20hz"])
            packet_start_40hz = int(pose_summary["packet_start_40hz"])
            packet_end_40hz = int(pose_summary["packet_end_40hz"])
            duration_sec_20hz = float(pose_summary["duration_sec_20hz"])
            interp_ratio_20hz = float(pose_summary["interp_ratio_20hz"])
            max_joint_interp_ratio_20hz = float(pose_summary["max_joint_interp_ratio_20hz"])
            worst_interp_joint = str(pose_summary["worst_interp_joint"])

        computed_preview_status = parse_status(export_row.get("computed_preview_status"))
        audit_preview_status = parse_status(export_row.get("audit_preview_status"))
        visual_full_preview_status = parse_status(visual_row.get("full_preview_status", visual_row.get("preview_status")))
        visual_selected_preview_status = parse_status(visual_row.get("selected_preview_status", visual_row.get("preview_status")))

        use_for_temporal_vae, use_reason = decide_segment_usage(
            has_all_10_sensors=has_all_10_sensors,
            split=split,
            export_decision=str(export_row.get("export_decision", "")).strip(),
            has_feature_file=has_feature_file,
            has_pose_file=has_pose_file,
            computed_preview_status=computed_preview_status,
            audit_preview_status=audit_preview_status,
            visual_full_preview_status=visual_full_preview_status,
            visual_selected_preview_status=visual_selected_preview_status,
            num_frames_20hz=num_frames_20hz,
            window_frames=window_frames,
            interp_ratio_20hz=interp_ratio_20hz,
            max_joint_interp_ratio_20hz=max_joint_interp_ratio_20hz,
            segment_interp_threshold=segment_interp_threshold,
            segment_joint_interp_threshold=segment_joint_interp_threshold,
        )

        segment_window_rows: list[dict[str, object]] = []
        if use_for_temporal_vae:
            assert pose_path is not None
            assert feature_path is not None
            stride_frames = train_stride_frames if split == "train" else eval_stride_frames
            interp_threshold = train_window_interp_threshold if split == "train" else eval_window_interp_threshold
            segment_window_rows = generate_pose_window_rows(
                pose_path=pose_path,
                feature_path=feature_path,
                participant=participant,
                segment_id=segment_id,
                window_frames=window_frames,
                stride_frames=stride_frames,
                interp_threshold=interp_threshold,
            )
            if not segment_window_rows:
                use_for_temporal_vae = False
                use_reason = "no_windows_after_filter"
            else:
                window_rows_by_split[split].extend(segment_window_rows)

        segment_rows.append(
            {
                "participant": participant,
                "segment_id": segment_id,
                "source_dir": source_dir,
                "feature_path": "" if feature_path is None else str(feature_path),
                "pose_path": "" if pose_path is None else str(pose_path),
                "has_all_10_sensors": int(has_all_10_sensors),
                "has_feature_file": int(has_feature_file),
                "has_pose_file": int(has_pose_file),
                "num_frames_20hz": num_frames_20hz,
                "packet_start_40hz": packet_start_40hz,
                "packet_end_40hz": packet_end_40hz,
                "duration_sec_20hz": duration_sec_20hz,
                "interp_ratio_20hz": interp_ratio_20hz,
                "max_joint_interp_ratio_20hz": max_joint_interp_ratio_20hz,
                "worst_interp_joint": worst_interp_joint,
                "split": split,
                "export_decision": str(export_row.get("export_decision", "")).strip(),
                "skip_reason": str(export_row.get("skip_reason", "")).strip(),
                "computed_preview_status": computed_preview_status,
                "audit_preview_status": audit_preview_status,
                "visual_full_preview_status": visual_full_preview_status,
                "visual_selected_preview_status": visual_selected_preview_status,
                "selected_p95_joint_step": parse_float(visual_row.get("selected_p95_joint_step")),
                "selected_max_joint_step": parse_float(visual_row.get("selected_max_joint_step")),
                "selected_p95_joint_jerk": parse_float(visual_row.get("selected_p95_joint_jerk")),
                "selected_max_joint_jerk": parse_float(visual_row.get("selected_max_joint_jerk")),
                "selected_p95_root_heading_delta_deg": parse_float(visual_row.get("selected_p95_root_heading_delta_deg")),
                "selected_max_root_heading_delta_deg": parse_float(visual_row.get("selected_max_root_heading_delta_deg")),
                "num_windows": len(segment_window_rows),
                "use_for_temporal_vae": int(use_for_temporal_vae),
                "use_for_temporal_vae_reason": use_reason,
            }
        )

    segment_rows = sort_segment_rows(segment_rows)
    for split in window_rows_by_split:
        window_rows_by_split[split] = sort_segment_rows(window_rows_by_split[split])

    write_csv(manifests_root / "vae_segment_manifest.csv", SEGMENT_MANIFEST_COLUMNS, segment_rows)
    write_csv(manifests_root / "vae_window_index_train.csv", WINDOW_INDEX_COLUMNS, window_rows_by_split["train"])
    write_csv(manifests_root / "vae_window_index_val.csv", WINDOW_INDEX_COLUMNS, window_rows_by_split["val"])
    write_csv(manifests_root / "vae_window_index_test.csv", WINDOW_INDEX_COLUMNS, window_rows_by_split["test"])

    split_manifest = {
        "train": list(TRAIN_PARTICIPANTS),
        "val": list(VAL_PARTICIPANTS),
        "test": list(TEST_PARTICIPANTS),
        "unused": sorted({row["participant"] for row in segment_rows if row["split"] == "unused"}),
    }
    (manifests_root / "vae_split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )

    return {
        "num_segments_total": len(segment_rows),
        "num_segments_for_temporal_vae": int(sum(int(row["use_for_temporal_vae"]) for row in segment_rows)),
        "num_train_windows": len(window_rows_by_split["train"]),
        "num_val_windows": len(window_rows_by_split["val"]),
        "num_test_windows": len(window_rows_by_split["test"]),
        "manifests_root": str(manifests_root),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "Data" / "IMU_Only_20Hz_v1"
    parser = argparse.ArgumentParser(description="Build pseudo-pose temporal VAE manifests")
    parser.add_argument("--data-root", type=Path, default=data_root)
    parser.add_argument("--manifests-root", type=Path, default=None)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--eval-stride-frames", type=int, default=120)
    parser.add_argument("--segment-interp-threshold", type=float, default=0.20)
    parser.add_argument("--segment-joint-interp-threshold", type=float, default=0.20)
    parser.add_argument("--train-window-interp-threshold", type=float, default=0.20)
    parser.add_argument("--eval-window-interp-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_vae_manifest_artifacts(
        data_root=args.data_root,
        manifests_root=args.manifests_root,
        window_frames=args.window_frames,
        train_stride_frames=args.train_stride_frames,
        eval_stride_frames=args.eval_stride_frames,
        segment_interp_threshold=args.segment_interp_threshold,
        segment_joint_interp_threshold=args.segment_joint_interp_threshold,
        train_window_interp_threshold=args.train_window_interp_threshold,
        eval_window_interp_threshold=args.eval_window_interp_threshold,
    )
    print(summary)


if __name__ == "__main__":
    main()
