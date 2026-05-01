#!/usr/bin/env python3
"""
Align filled IMU_New2 data with D_* action labels from annotation json files.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
import pickle
from pathlib import Path

import numpy as np

from imu_new2_common import (
    DOG_SENSOR_IDS,
    IMU_RATE_HZ,
    JOINT_ORDER,
    build_packet_labels,
    collect_d_label_vocab,
    load_clip_labels,
    read_filled_sensor_pickle,
)

MIN_DURATION_THRESHOLD_SEC = 15.0
STERN_SENSOR_ID = "00B49A94"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align filled IMU_New2 data with D_* annotation labels and export aligned pkls.",
    )
    parser.add_argument(
        "--imu-root",
        type=Path,
        default=Path("Data/IMU_New2_Filled"),
        help="Filled IMU root. Default: Data/IMU_New2_Filled",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=Path("Data/annotation"),
        help="Annotation root. Default: Data/annotation",
    )
    parser.add_argument(
        "--projection-txt",
        type=Path,
        default=Path("Data/projection.txt"),
        help="Projection mapping txt. Default: Data/projection.txt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/aligned_actions_v1"),
        help="Output root for aligned pkl files. Default: Data/aligned_actions_v1",
    )
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=None,
        help="Optional audit csv path. Default: <output-root>/alignment_audit.csv",
    )
    return parser.parse_args()


def read_projection_map(path: Path) -> dict[str, str | None]:
    projection = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            projection[parts[0]] = parts[1] if len(parts) > 1 else None
    return projection


def discover_clip_dirs(annotation_root: Path, anon_participant: str) -> list[Path]:
    clip_root = annotation_root / anon_participant / "Insta360" / "DCIM" / "Camera01"
    if not clip_root.exists():
        return []
    return sorted((path for path in clip_root.glob("*") if path.is_dir()), key=lambda path: int(path.name))


def discover_segment_dirs(imu_root: Path, imu_participant: str) -> list[Path]:
    participant_root = imu_root / imu_participant
    if not participant_root.exists():
        return []
    return sorted((path for path in participant_root.glob("*") if path.is_dir()), key=lambda path: int(path.name))


def select_limiting_sensor(
    sensor_stats: list[dict[str, int]],
    key: str,
    reducer,
) -> str:
    limiting_value = reducer(stat[key] for stat in sensor_stats)
    for sensor_id in DOG_SENSOR_IDS:
        for stat in sensor_stats:
            if stat["sensor_id"] == sensor_id and stat[key] == limiting_value:
                return sensor_id
    return ""


def compute_overlap_sec(sensor_stats: list[dict[str, int]], dropped_indexes: tuple[int, ...] = ()) -> float:
    kept = [stat for index, stat in enumerate(sensor_stats) if index not in dropped_indexes]
    if not kept:
        return 0.0

    overlap_start = max(stat["packet_start"] for stat in kept)
    overlap_end = min(stat["packet_end"] for stat in kept)
    if overlap_start > overlap_end:
        return 0.0
    return float(overlap_end - overlap_start + 1) / float(IMU_RATE_HZ)


def compute_drop_overlap_sec(sensor_stats: list[dict[str, int]], num_drop: int) -> float:
    if num_drop <= 0:
        return compute_overlap_sec(sensor_stats)
    if num_drop >= len(sensor_stats):
        return 0.0
    return max(compute_overlap_sec(sensor_stats, dropped) for dropped in combinations(range(len(sensor_stats)), num_drop))


def build_segment_diagnostics(sensor_stats: list[dict[str, int]]) -> dict[str, str | float | int]:
    common_start = max(stat["packet_start"] for stat in sensor_stats)
    common_end = min(stat["packet_end"] for stat in sensor_stats)
    stern_stat = next(stat for stat in sensor_stats if stat["sensor_id"] == STERN_SENSOR_ID)
    return {
        "common_packet_start": common_start,
        "common_packet_end": common_end,
        "stern_duration_sec": stern_stat["duration_sec"],
        "limiting_start_sensor": select_limiting_sensor(sensor_stats, "packet_start", max),
        "limiting_end_sensor": select_limiting_sensor(sensor_stats, "packet_end", min),
        "drop1_overlap_sec": compute_drop_overlap_sec(sensor_stats, 1),
        "drop2_overlap_sec": compute_drop_overlap_sec(sensor_stats, 2),
    }


def load_segment_sensor_arrays(segment_dir: Path) -> tuple[dict[str, np.ndarray], float]:
    sensor_paths = {}
    for pkl_path in sorted(segment_dir.glob("*.pkl")):
        sensor_id = pkl_path.stem.split("_")[-1]
        sensor_paths.setdefault(sensor_id, pkl_path)

    missing_sensors = [sensor_id for sensor_id in DOG_SENSOR_IDS if sensor_id not in sensor_paths]
    if missing_sensors:
        raise FileNotFoundError(",".join(missing_sensors))

    sensor_frames = []
    sensor_stats = []
    for sensor_id in DOG_SENSOR_IDS:
        frame = read_filled_sensor_pickle(sensor_paths[sensor_id])
        frame = frame.set_index("packet_counter", drop=False)
        sensor_frames.append(frame)
        sensor_start = int(frame["packet_counter"].iloc[0])
        sensor_end = int(frame["packet_counter"].iloc[-1])
        sensor_stats.append(
            {
                "sensor_id": sensor_id,
                "packet_start": sensor_start,
                "packet_end": sensor_end,
                "duration_sec": float(sensor_end - sensor_start + 1) / float(IMU_RATE_HZ),
            }
        )

    diagnostics = build_segment_diagnostics(sensor_stats)
    common_start = int(diagnostics["common_packet_start"])
    common_end = int(diagnostics["common_packet_end"])
    if common_start is None or common_end is None or common_start > common_end:
        raise ValueError("empty_common_axis")

    common_packet_counter = np.arange(common_start, common_end + 1, dtype=np.int64)
    num_packets = int(common_packet_counter.shape[0])
    num_sensors = len(DOG_SENSOR_IDS)

    quat = np.zeros((num_packets, num_sensors, 4), dtype=np.float32)
    gyr = np.zeros((num_packets, num_sensors, 3), dtype=np.float32)
    freeacc = np.zeros((num_packets, num_sensors, 3), dtype=np.float32)
    is_interpolated = np.zeros((num_packets, num_sensors), dtype=bool)

    for sensor_index, frame in enumerate(sensor_frames):
        sliced = frame.loc[common_packet_counter]
        quat[:, sensor_index] = sliced.loc[:, ["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=np.float32)
        gyr[:, sensor_index] = sliced.loc[:, ["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=np.float32)
        freeacc[:, sensor_index] = sliced.loc[:, ["freeacc_x", "freeacc_y", "freeacc_z"]].to_numpy(dtype=np.float32)
        is_interpolated[:, sensor_index] = sliced["is_interpolated"].to_numpy(dtype=bool)

    imu_duration_sec = float(num_packets) / float(IMU_RATE_HZ)
    return (
        {
            "packet_counter": common_packet_counter,
            "sensor_ids": np.asarray(DOG_SENSOR_IDS, dtype=object),
            "joint_names": np.asarray(JOINT_ORDER, dtype=object),
            "quat": quat,
            "gyr": gyr,
            "freeacc": freeacc,
            "is_interpolated": is_interpolated,
            "common_packet_start": int(common_packet_counter[0]),
            "common_packet_end": int(common_packet_counter[-1]),
            "stern_duration_sec": diagnostics["stern_duration_sec"],
            "limiting_start_sensor": diagnostics["limiting_start_sensor"],
            "limiting_end_sensor": diagnostics["limiting_end_sensor"],
            "drop1_overlap_sec": diagnostics["drop1_overlap_sec"],
            "drop2_overlap_sec": diagnostics["drop2_overlap_sec"],
        },
        imu_duration_sec,
    )


def save_label_vocab(output_root: Path, label_vocab: list[str]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    vocab_path = output_root / "label_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as handle:
        json.dump(label_vocab, handle, indent=2)


def build_record(
    anon_participant: str,
    imu_participant: str,
    clip_dir: Path,
    segment_dir: Path,
    annotation_root: Path,
    label_vocab: list[str],
    label_to_index: dict[str, int],
) -> tuple[str, dict, dict]:
    labels_1hz, source_json_files = load_clip_labels(clip_dir, label_to_index, annotation_root)
    ann_duration_sec = int(labels_1hz.shape[0])

    try:
        segment_arrays, imu_duration_sec = load_segment_sensor_arrays(segment_dir)
    except FileNotFoundError as exc:
        return "missing_sensor", {}, {"missing_sensors": str(exc)}
    except ValueError:
        return "empty_common_axis", {}, {}

    duration_threshold_sec = max(MIN_DURATION_THRESHOLD_SEC, 0.06 * ann_duration_sec)
    duration_delta_sec = float(imu_duration_sec - ann_duration_sec)
    if abs(duration_delta_sec) > duration_threshold_sec:
        return (
            "duration_mismatch",
            {},
            {
                "ann_duration_sec": ann_duration_sec,
                "imu_duration_sec": imu_duration_sec,
                "duration_delta_sec": duration_delta_sec,
                "duration_threshold_sec": duration_threshold_sec,
                "stern_duration_sec": segment_arrays["stern_duration_sec"],
                "limiting_start_sensor": segment_arrays["limiting_start_sensor"],
                "limiting_end_sensor": segment_arrays["limiting_end_sensor"],
                "drop1_overlap_sec": segment_arrays["drop1_overlap_sec"],
                "drop2_overlap_sec": segment_arrays["drop2_overlap_sec"],
            },
        )

    labels_40hz, second_to_packet = build_packet_labels(labels_1hz, segment_arrays["packet_counter"])
    record = {
        "meta": {
            "anon_participant": anon_participant,
            "imu_participant": imu_participant,
            "clip_id": clip_dir.name,
            "segment_id": segment_dir.name,
            "annotation_duration_sec": ann_duration_sec,
            "imu_duration_sec": imu_duration_sec,
            "duration_delta_sec": duration_delta_sec,
            "duration_threshold_sec": duration_threshold_sec,
            "common_packet_start": segment_arrays["common_packet_start"],
            "common_packet_end": segment_arrays["common_packet_end"],
            "stern_duration_sec": segment_arrays["stern_duration_sec"],
            "limiting_start_sensor": segment_arrays["limiting_start_sensor"],
            "limiting_end_sensor": segment_arrays["limiting_end_sensor"],
            "drop1_overlap_sec": segment_arrays["drop1_overlap_sec"],
            "drop2_overlap_sec": segment_arrays["drop2_overlap_sec"],
            "num_seconds": int(labels_1hz.shape[0]),
            "num_packets": int(segment_arrays["packet_counter"].shape[0]),
            "num_windows": len(source_json_files),
        },
        "packet_counter": segment_arrays["packet_counter"],
        "sensor_ids": segment_arrays["sensor_ids"],
        "joint_names": segment_arrays["joint_names"],
        "quat": segment_arrays["quat"],
        "gyr": segment_arrays["gyr"],
        "freeacc": segment_arrays["freeacc"],
        "is_interpolated": segment_arrays["is_interpolated"],
        "label_vocab": np.asarray(label_vocab, dtype=object),
        "labels_1hz": labels_1hz,
        "labels_40hz": labels_40hz,
        "second_to_packet": second_to_packet,
        "source_json_files": np.asarray(source_json_files, dtype=object),
    }
    return "accepted", record, record["meta"]


def process_alignment(
    imu_root: Path,
    annotation_root: Path,
    projection_txt: Path,
    output_root: Path,
    audit_csv_path: Path,
) -> None:
    if not imu_root.exists():
        raise FileNotFoundError(f"Filled IMU root not found: {imu_root}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"Annotation root not found: {annotation_root}")
    if not projection_txt.exists():
        raise FileNotFoundError(f"Projection file not found: {projection_txt}")

    output_root.mkdir(parents=True, exist_ok=True)
    projection = read_projection_map(projection_txt)
    label_vocab = collect_d_label_vocab(annotation_root)
    label_to_index = {label: index for index, label in enumerate(label_vocab)}
    save_label_vocab(output_root, label_vocab)

    audit_rows = []

    for anon_participant in sorted(projection):
        imu_participant = projection[anon_participant]
        clip_dirs = discover_clip_dirs(annotation_root, anon_participant)
        segment_dirs = discover_segment_dirs(imu_root, imu_participant) if imu_participant else []

        if imu_participant is None or len(clip_dirs) != len(segment_dirs):
            audit_rows.append(
                {
                    "anon_participant": anon_participant,
                    "imu_participant": imu_participant or "",
                    "clip_id": "",
                    "segment_id": "",
                    "status": "count_mismatch",
                    "annotation_duration_sec": "",
                    "imu_duration_sec": "",
                    "duration_delta_sec": "",
                    "duration_threshold_sec": "",
                    "annotation_clip_count": len(clip_dirs),
                    "imu_segment_count": len(segment_dirs),
                    "common_packet_start": "",
                    "common_packet_end": "",
                    "stern_duration_sec": "",
                    "limiting_start_sensor": "",
                    "limiting_end_sensor": "",
                    "drop1_overlap_sec": "",
                    "drop2_overlap_sec": "",
                    "output_path": "",
                    "note": "missing_projection" if imu_participant is None else "clip_segment_count_mismatch",
                }
            )
            continue

        for clip_dir, segment_dir in zip(clip_dirs, segment_dirs):
            status, record, extra = build_record(
                anon_participant=anon_participant,
                imu_participant=imu_participant,
                clip_dir=clip_dir,
                segment_dir=segment_dir,
                annotation_root=annotation_root,
                label_vocab=label_vocab,
                label_to_index=label_to_index,
            )

            output_path = ""
            if status == "accepted":
                participant_output_dir = output_root / anon_participant
                participant_output_dir.mkdir(parents=True, exist_ok=True)
                sample_path = participant_output_dir / f"clip_{clip_dir.name}__segment_{segment_dir.name}.pkl"
                with open(sample_path, "wb") as handle:
                    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
                output_path = str(sample_path)

            audit_rows.append(
                {
                    "anon_participant": anon_participant,
                    "imu_participant": imu_participant,
                    "clip_id": clip_dir.name,
                    "segment_id": segment_dir.name,
                    "status": status,
                    "annotation_duration_sec": extra.get("ann_duration_sec", record.get("meta", {}).get("annotation_duration_sec", "")),
                    "imu_duration_sec": extra.get("imu_duration_sec", record.get("meta", {}).get("imu_duration_sec", "")),
                    "duration_delta_sec": extra.get("duration_delta_sec", record.get("meta", {}).get("duration_delta_sec", "")),
                    "duration_threshold_sec": extra.get(
                        "duration_threshold_sec",
                        record.get("meta", {}).get("duration_threshold_sec", ""),
                    ),
                    "annotation_clip_count": len(clip_dirs),
                    "imu_segment_count": len(segment_dirs),
                    "common_packet_start": record.get("meta", {}).get("common_packet_start", ""),
                    "common_packet_end": record.get("meta", {}).get("common_packet_end", ""),
                    "stern_duration_sec": extra.get("stern_duration_sec", record.get("meta", {}).get("stern_duration_sec", "")),
                    "limiting_start_sensor": extra.get(
                        "limiting_start_sensor",
                        record.get("meta", {}).get("limiting_start_sensor", ""),
                    ),
                    "limiting_end_sensor": extra.get(
                        "limiting_end_sensor",
                        record.get("meta", {}).get("limiting_end_sensor", ""),
                    ),
                    "drop1_overlap_sec": extra.get("drop1_overlap_sec", record.get("meta", {}).get("drop1_overlap_sec", "")),
                    "drop2_overlap_sec": extra.get("drop2_overlap_sec", record.get("meta", {}).get("drop2_overlap_sec", "")),
                    "output_path": output_path,
                    "note": extra.get("missing_sensors", ""),
                }
            )

    with open(audit_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "anon_participant",
                "imu_participant",
                "clip_id",
                "segment_id",
                "status",
                "annotation_duration_sec",
                "imu_duration_sec",
                "duration_delta_sec",
                "duration_threshold_sec",
                "annotation_clip_count",
                "imu_segment_count",
                "common_packet_start",
                "common_packet_end",
                "stern_duration_sec",
                "limiting_start_sensor",
                "limiting_end_sensor",
                "drop1_overlap_sec",
                "drop2_overlap_sec",
                "output_path",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)


def main() -> None:
    args = parse_args()
    audit_csv_path = args.audit_csv or (args.output_root / "alignment_audit.csv")
    process_alignment(
        imu_root=args.imu_root,
        annotation_root=args.annotation_root,
        projection_txt=args.projection_txt,
        output_root=args.output_root,
        audit_csv_path=audit_csv_path,
    )
    print(f"Wrote aligned samples to {args.output_root}")
    print(f"Wrote audit csv to {audit_csv_path}")


if __name__ == "__main__":
    main()
