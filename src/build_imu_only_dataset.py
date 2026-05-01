#!/usr/bin/env python3
"""
20Hz feature export and dataset artifact builder for IMU-only baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from imu_new2_common import DOG_SENSOR_IDS, FLOAT_DTYPE, JOINT_ORDER, read_filled_sensor_pickle, slerp_quaternion


TRAIN_PARTICIPANTS = (
    "Bernice-Amber",
    "Brett-Ursa",
    "Cody-Travis",
    "Emma-Ellie",
    "Jamia-Kelku",
    "Jithvan-Nero",
    "Jonathan-Pemi",
    "Lindsay-Douglas",
    "Meg-Binx",
    "Susan-Prada",
    "Tim-Bellie-Amie",
)
VAL_PARTICIPANTS = ("Cynthia-WolfGang", "Diane-Penny")
TEST_PARTICIPANTS = ("Bill-Coco", "Krista-Romeo")

SEGMENT_MANIFEST_COLUMNS = (
    "participant",
    "segment_id",
    "source_dir",
    "feature_path",
    "has_all_10_sensors",
    "packet_start_40hz",
    "packet_end_40hz",
    "duration_sec_40hz",
    "interp_ratio_40hz",
    "num_frames_20hz",
    "interp_ratio_20hz",
    "split",
    "use_for_raw_baseline",
    "use_for_pseudo_pose",
)

WINDOW_INDEX_COLUMNS = (
    "participant",
    "segment_id",
    "feature_path",
    "start_frame_20hz",
    "end_frame_20hz",
    "packet_start_40hz",
    "packet_end_40hz",
    "interp_ratio",
    "task",
)


def extract_sensor_id_from_path(path: Path) -> str | None:
    stem = path.stem
    for sensor_id in DOG_SENSOR_IDS:
        if sensor_id in stem:
            return sensor_id
    return None


def quaternion_to_rot6d(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    quat = quat / norm
    w, x, y, z = np.moveaxis(quat, -1, 0)
    rotation = np.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))
    return rotation[..., :, :2].reshape(quat.shape[:-1] + (6,)).astype(FLOAT_DTYPE)


def downsample_quaternion_pairs(quat: np.ndarray) -> np.ndarray:
    pair_count = quat.shape[0] // 2
    output = np.zeros((pair_count, quat.shape[1], 4), dtype=FLOAT_DTYPE)
    for pair_index in range(pair_count):
        left = quat[2 * pair_index]
        right = quat[2 * pair_index + 1]
        for sensor_index in range(quat.shape[1]):
            output[pair_index, sensor_index] = slerp_quaternion(left[sensor_index], right[sensor_index], 0.5)
    return output


def determine_split(participant: str) -> str:
    if participant in TRAIN_PARTICIPANTS:
        return "train"
    if participant in VAL_PARTICIPANTS:
        return "val"
    if participant in TEST_PARTICIPANTS:
        return "test"
    return "unused"


def load_filled_dog_segment(segment_dir: Path) -> dict[str, np.ndarray]:
    sensor_frames: dict[str, object] = {}
    for pkl_path in sorted(segment_dir.glob("*.pkl")):
        sensor_id = extract_sensor_id_from_path(pkl_path)
        if sensor_id is None:
            continue
        sensor_frames[sensor_id] = read_filled_sensor_pickle(pkl_path)

    missing_sensor_ids = [sensor_id for sensor_id in DOG_SENSOR_IDS if sensor_id not in sensor_frames]
    if missing_sensor_ids:
        raise ValueError(f"Missing dog sensors in {segment_dir}: {missing_sensor_ids}")

    packet_start = max(int(sensor_frames[sensor_id]["packet_counter"].iloc[0]) for sensor_id in DOG_SENSOR_IDS)
    packet_end = min(int(sensor_frames[sensor_id]["packet_counter"].iloc[-1]) for sensor_id in DOG_SENSOR_IDS)
    if packet_end <= packet_start:
        raise ValueError(f"Empty common packet axis in {segment_dir}: start={packet_start}, end={packet_end}")

    packet_counter = None
    quat = []
    gyr = []
    freeacc = []
    is_interpolated = []
    for sensor_id in DOG_SENSOR_IDS:
        frame = sensor_frames[sensor_id]
        cropped = frame[(frame["packet_counter"] >= packet_start) & (frame["packet_counter"] <= packet_end)].reset_index(drop=True)
        current_packet_counter = cropped["packet_counter"].to_numpy(dtype=np.int64)
        if packet_counter is None:
            packet_counter = current_packet_counter
        elif not np.array_equal(packet_counter, current_packet_counter):
            raise ValueError(f"Packet axis mismatch after cropping in {segment_dir}")
        quat.append(cropped.loc[:, ["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=FLOAT_DTYPE))
        gyr.append(cropped.loc[:, ["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=FLOAT_DTYPE))
        freeacc.append(cropped.loc[:, ["freeacc_x", "freeacc_y", "freeacc_z"]].to_numpy(dtype=FLOAT_DTYPE))
        is_interpolated.append(cropped["is_interpolated"].to_numpy(dtype=bool))

    assert packet_counter is not None
    return {
        "packet_counter": packet_counter,
        "quat": np.stack(quat, axis=1),
        "gyr": np.stack(gyr, axis=1),
        "freeacc": np.stack(freeacc, axis=1),
        "is_interpolated": np.stack(is_interpolated, axis=1),
        "sensor_ids": np.asarray(DOG_SENSOR_IDS),
        "joint_names": np.asarray(JOINT_ORDER),
    }


def downsample_segment_to_20hz(segment_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    packet_counter = segment_data["packet_counter"]
    quat = segment_data["quat"]
    gyr = segment_data["gyr"]
    freeacc = segment_data["freeacc"]
    is_interpolated = segment_data["is_interpolated"]

    frame_count = packet_counter.shape[0] - (packet_counter.shape[0] % 2)
    if frame_count < 2:
        raise ValueError("Need at least two 40Hz frames to downsample to 20Hz")

    packet_counter = packet_counter[:frame_count]
    quat = quat[:frame_count]
    gyr = gyr[:frame_count]
    freeacc = freeacc[:frame_count]
    is_interpolated = is_interpolated[:frame_count]

    downsampled_packet_counter = packet_counter[::2]
    downsampled_quat = downsample_quaternion_pairs(quat)
    downsampled_gyr = ((gyr[::2] + gyr[1::2]) * 0.5).astype(FLOAT_DTYPE)
    downsampled_freeacc = ((freeacc[::2] + freeacc[1::2]) * 0.5).astype(FLOAT_DTYPE)
    downsampled_is_interpolated = np.logical_or(is_interpolated[::2], is_interpolated[1::2])
    downsampled_rot6d = quaternion_to_rot6d(downsampled_quat)

    return {
        "packet_counter": downsampled_packet_counter.astype(np.int64),
        "quat": downsampled_quat.astype(FLOAT_DTYPE),
        "rot6d": downsampled_rot6d.astype(FLOAT_DTYPE),
        "gyr": downsampled_gyr.astype(FLOAT_DTYPE),
        "freeacc": downsampled_freeacc.astype(FLOAT_DTYPE),
        "is_interpolated": downsampled_is_interpolated.astype(bool),
        "sensor_ids": segment_data["sensor_ids"],
        "joint_names": segment_data["joint_names"],
    }


def build_feature_tensor(
    rot6d: np.ndarray,
    gyr: np.ndarray,
    freeacc: np.ndarray,
    is_interpolated: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            rot6d.astype(FLOAT_DTYPE),
            gyr.astype(FLOAT_DTYPE),
            freeacc.astype(FLOAT_DTYPE),
            is_interpolated[..., None].astype(FLOAT_DTYPE),
        ],
        axis=-1,
    )


def build_20hz_feature_file(segment_dir: Path, output_path: Path) -> dict[str, object]:
    segment_data_40hz = load_filled_dog_segment(segment_dir)
    segment_data_20hz = downsample_segment_to_20hz(segment_data_40hz)
    feature = build_feature_tensor(
        rot6d=segment_data_20hz["rot6d"],
        gyr=segment_data_20hz["gyr"],
        freeacc=segment_data_20hz["freeacc"],
        is_interpolated=segment_data_20hz["is_interpolated"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        packet_counter=segment_data_20hz["packet_counter"],
        quat=segment_data_20hz["quat"],
        rot6d=segment_data_20hz["rot6d"],
        gyr=segment_data_20hz["gyr"],
        freeacc=segment_data_20hz["freeacc"],
        is_interpolated=segment_data_20hz["is_interpolated"].astype(np.uint8),
        sensor_ids=segment_data_20hz["sensor_ids"],
        joint_names=segment_data_20hz["joint_names"],
        feature=feature,
        participant=np.asarray(segment_dir.parent.name),
        segment_id=np.asarray(segment_dir.name),
    )
    return {
        "participant": segment_dir.parent.name,
        "segment_id": segment_dir.name,
        "num_frames_20hz": int(feature.shape[0]),
        "output_path": str(output_path),
    }


def generate_window_rows(
    *,
    feature_path: Path,
    participant: str,
    segment_id: str,
    split: str,
    window_frames: int,
    stride_frames: int,
    interp_threshold: float,
) -> list[dict[str, object]]:
    with np.load(feature_path, allow_pickle=False) as payload:
        packet_counter = payload["packet_counter"]
        is_interpolated = payload["is_interpolated"].astype(bool)

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
                "feature_path": str(feature_path),
                "start_frame_20hz": int(start_frame),
                "end_frame_20hz": int(end_frame - 1),
                "packet_start_40hz": int(packet_counter[start_frame]),
                "packet_end_40hz": int(packet_counter[end_frame - 1]),
                "interp_ratio": window_interp_ratio,
                "task": "masked_recon",
            }
        )
    return rows


def sort_segment_dirs(participant_dir: Path) -> list[Path]:
    return sorted([path for path in participant_dir.iterdir() if path.is_dir()], key=lambda path: path.name)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_imu_only_dataset_artifacts(
    *,
    input_root: Path,
    output_root: Path,
    window_frames: int = 240,
    train_stride_frames: int = 20,
    eval_stride_frames: int = 120,
    segment_interp_threshold: float = 0.20,
    train_window_interp_threshold: float = 0.20,
    eval_window_interp_threshold: float = 0.05,
) -> dict[str, object]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    features_root = output_root / "features_20hz"
    manifests_root = output_root / "manifests"
    features_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    segment_rows: list[dict[str, object]] = []
    window_rows_by_split: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    participant_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()], key=lambda path: path.name)
    num_segments_total = 0
    num_segments_with_features = 0
    for participant_dir in participant_dirs:
        participant = participant_dir.name
        split = determine_split(participant)
        for segment_dir in sort_segment_dirs(participant_dir):
            num_segments_total += 1
            feature_path = features_root / participant / f"segment_{segment_dir.name}.npz"
            row: dict[str, object] = {
                "participant": participant,
                "segment_id": segment_dir.name,
                "source_dir": str(segment_dir),
                "feature_path": "",
                "has_all_10_sensors": 0,
                "packet_start_40hz": "",
                "packet_end_40hz": "",
                "duration_sec_40hz": "",
                "interp_ratio_40hz": "",
                "num_frames_20hz": "",
                "interp_ratio_20hz": "",
                "split": split,
                "use_for_raw_baseline": 0,
                "use_for_pseudo_pose": 0,
            }

            try:
                segment_data_40hz = load_filled_dog_segment(segment_dir)
            except Exception:
                segment_rows.append(row)
                continue

            num_segments_with_features += 1
            build_20hz_feature_file(segment_dir=segment_dir, output_path=feature_path)
            with np.load(feature_path, allow_pickle=False) as payload:
                num_frames_20hz = int(payload["feature"].shape[0])
                interp_ratio_20hz = float(payload["is_interpolated"].astype(bool).mean())

            duration_sec_40hz = float(segment_data_40hz["packet_counter"].shape[0]) / 40.0
            interp_ratio_40hz = float(segment_data_40hz["is_interpolated"].mean())
            use_for_raw_baseline = int(
                split in {"train", "val", "test"}
                and num_frames_20hz >= window_frames
                and interp_ratio_20hz <= segment_interp_threshold
            )

            row.update(
                {
                    "feature_path": str(feature_path),
                    "has_all_10_sensors": 1,
                    "packet_start_40hz": int(segment_data_40hz["packet_counter"][0]),
                    "packet_end_40hz": int(segment_data_40hz["packet_counter"][-1]),
                    "duration_sec_40hz": duration_sec_40hz,
                    "interp_ratio_40hz": interp_ratio_40hz,
                    "num_frames_20hz": num_frames_20hz,
                    "interp_ratio_20hz": interp_ratio_20hz,
                    "use_for_raw_baseline": use_for_raw_baseline,
                    "use_for_pseudo_pose": use_for_raw_baseline,
                }
            )
            segment_rows.append(row)

            if use_for_raw_baseline:
                if split == "train":
                    window_rows_by_split["train"].extend(
                        generate_window_rows(
                            feature_path=feature_path,
                            participant=participant,
                            segment_id=segment_dir.name,
                            split=split,
                            window_frames=window_frames,
                            stride_frames=train_stride_frames,
                            interp_threshold=train_window_interp_threshold,
                        )
                    )
                elif split in {"val", "test"}:
                    window_rows_by_split[split].extend(
                        generate_window_rows(
                            feature_path=feature_path,
                            participant=participant,
                            segment_id=segment_dir.name,
                            split=split,
                            window_frames=window_frames,
                            stride_frames=eval_stride_frames,
                            interp_threshold=eval_window_interp_threshold,
                        )
                    )

    write_csv(manifests_root / "segment_manifest.csv", SEGMENT_MANIFEST_COLUMNS, segment_rows)
    write_csv(manifests_root / "window_index_train.csv", WINDOW_INDEX_COLUMNS, window_rows_by_split["train"])
    write_csv(manifests_root / "window_index_val.csv", WINDOW_INDEX_COLUMNS, window_rows_by_split["val"])
    write_csv(manifests_root / "window_index_test.csv", WINDOW_INDEX_COLUMNS, window_rows_by_split["test"])

    split_manifest = {
        "train": list(TRAIN_PARTICIPANTS),
        "val": list(VAL_PARTICIPANTS),
        "test": list(TEST_PARTICIPANTS),
        "unused": [row["participant"] for row in segment_rows if row["split"] == "unused"],
    }
    (manifests_root / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    return {
        "num_segments_total": num_segments_total,
        "num_segments_with_features": num_segments_with_features,
        "num_train_windows": len(window_rows_by_split["train"]),
        "num_val_windows": len(window_rows_by_split["val"]),
        "num_test_windows": len(window_rows_by_split["test"]),
        "output_root": str(output_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 20Hz feature files and dataset artifacts for IMU-only baselines")
    parser.add_argument("--segment-dir", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--eval-stride-frames", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.segment_dir is not None:
        if args.output_path is None:
            raise ValueError("--output-path is required when --segment-dir is used")
        metadata = build_20hz_feature_file(segment_dir=args.segment_dir, output_path=args.output_path)
        print(metadata)
        return
    if args.input_root is None or args.output_root is None:
        raise ValueError("Use either --segment-dir/--output-path or --input-root/--output-root")
    summary = build_imu_only_dataset_artifacts(
        input_root=args.input_root,
        output_root=args.output_root,
        window_frames=args.window_frames,
        train_stride_frames=args.train_stride_frames,
        eval_stride_frames=args.eval_stride_frames,
    )
    print(summary)


if __name__ == "__main__":
    main()
