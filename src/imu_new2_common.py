#!/usr/bin/env python3
"""
Utilities for IMU_New2 packet filling and annotation alignment.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PACKET_COUNTER_MODULUS = 65536
IMU_RATE_HZ = 40
FLOAT_DTYPE = np.float32

RAW_SENSOR_COLUMNS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "freeacc_x",
    "freeacc_y",
    "freeacc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
    "pressure",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)
CANONICAL_SENSOR_COLUMNS = ("packet_counter",) + RAW_SENSOR_COLUMNS + ("is_interpolated",)

SENSOR_TO_JOINT = {
    "00B49A9E": "head",
    "00B49A94": "stern",
    "00B49A9B": "upper_arm_left",
    "00B49702": "upper_arm_right",
    "00B49AA0": "left_hand",
    "00B49A9F": "right_hand",
    "00B49AA1": "upper_leg_left",
    "00B49A98": "upper_leg_right",
    "00B49A9D": "left_foot",
    "00B49A9C": "right_foot",
}
JOINT_ORDER = (
    "stern",
    "head",
    "upper_arm_left",
    "left_hand",
    "upper_arm_right",
    "right_hand",
    "upper_leg_left",
    "left_foot",
    "upper_leg_right",
    "right_foot",
)
JOINT_TO_SENSOR = {joint_name: sensor_id for sensor_id, joint_name in SENSOR_TO_JOINT.items()}
DOG_SENSOR_IDS = tuple(JOINT_TO_SENSOR[joint_name] for joint_name in JOINT_ORDER)


def sort_numeric_paths(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: int(path.name.split("_")[0]) if "_" in path.name else int(path.name))


def normalize_packet_counter(packet_counter_raw: np.ndarray) -> np.ndarray:
    if packet_counter_raw.size == 0:
        return np.zeros((0,), dtype=np.int64)

    continuous = np.zeros(packet_counter_raw.shape[0], dtype=np.int64)
    continuous[0] = int(packet_counter_raw[0])

    saw_value_above_modulus = bool(np.any(packet_counter_raw >= PACKET_COUNTER_MODULUS))

    if saw_value_above_modulus:
        for i in range(1, packet_counter_raw.shape[0]):
            current_value = int(packet_counter_raw[i])
            previous_value = int(continuous[i - 1])

            if current_value >= previous_value:
                continuous[i] = current_value
            else:
                wrapped_candidate = current_value + PACKET_COUNTER_MODULUS
                if wrapped_candidate >= previous_value:
                    continuous[i] = wrapped_candidate
                else:
                    k = (previous_value - current_value + PACKET_COUNTER_MODULUS - 1) // PACKET_COUNTER_MODULUS
                    continuous[i] = current_value + k * PACKET_COUNTER_MODULUS
    else:
        offset = 0
        previous_raw = int(packet_counter_raw[0])
        for i in range(1, packet_counter_raw.shape[0]):
            current_raw = int(packet_counter_raw[i])
            if current_raw < previous_raw:
                offset += PACKET_COUNTER_MODULUS
            continuous[i] = current_raw + offset
            previous_raw = current_raw

    return continuous


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return quat.copy()
    return quat / norm


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(min(dot, 1.0), -1.0)

    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return normalize_quaternion(result).astype(FLOAT_DTYPE)

    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    result = s0 * q0 + s1 * q1
    return normalize_quaternion(result).astype(FLOAT_DTYPE)


def interpolate_linear_vector(
    continuous_packet_counter: np.ndarray,
    values: np.ndarray,
    full_packet_counter: np.ndarray,
) -> np.ndarray:
    output = np.zeros((full_packet_counter.shape[0], values.shape[1]), dtype=FLOAT_DTYPE)
    x = continuous_packet_counter.astype(np.float64)
    x_full = full_packet_counter.astype(np.float64)

    for dim in range(values.shape[1]):
        y = values[:, dim].astype(np.float64)
        valid_mask = np.isfinite(y)
        if not np.any(valid_mask):
            output[:, dim] = 0.0
            continue
        if int(valid_mask.sum()) == 1:
            output[:, dim] = y[valid_mask][0]
            continue
        output[:, dim] = np.interp(x_full, x[valid_mask], y[valid_mask]).astype(FLOAT_DTYPE)

    return output


def interpolate_quaternion(
    continuous_packet_counter: np.ndarray,
    quat_values: np.ndarray,
    full_packet_counter: np.ndarray,
) -> np.ndarray:
    total_length = full_packet_counter.shape[0]
    output = np.zeros((total_length, 4), dtype=FLOAT_DTYPE)
    start_packet = int(full_packet_counter[0])
    index_map = {}
    for i in range(continuous_packet_counter.shape[0]):
        if np.all(np.isfinite(quat_values[i])):
            index_map[int(continuous_packet_counter[i]) - start_packet] = i

    observed_positions = sorted(index_map.keys())
    if not observed_positions:
        return output
    if len(observed_positions) == 1:
        only_quat = normalize_quaternion(quat_values[index_map[observed_positions[0]]]).astype(FLOAT_DTYPE)
        output[:] = only_quat
        return output

    for pos in observed_positions:
        original_index = index_map[pos]
        output[pos] = normalize_quaternion(quat_values[original_index]).astype(FLOAT_DTYPE)

    for pair_index in range(len(observed_positions) - 1):
        left_pos = observed_positions[pair_index]
        right_pos = observed_positions[pair_index + 1]
        left_quat = output[left_pos]
        right_quat = output[right_pos]
        gap = right_pos - left_pos
        if gap <= 1:
            continue
        for step in range(1, gap):
            t = float(step) / float(gap)
            output[left_pos + step] = slerp_quaternion(left_quat, right_quat, t)

    return output


def _split_single_column_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.iloc[:, 0].astype(str)
    rows = []
    for raw_value in values.tolist():
        if "\t" in raw_value:
            parts = raw_value.split("\t")
        else:
            parts = [part.strip() for part in raw_value.split(",")]
        if len(parts) != 18:
            raise ValueError(f"Expected 18 columns after splitting single-column pkl row, got {len(parts)}")
        rows.append(parts)
    return pd.DataFrame(rows)


def read_sensor_pickle(path: Path) -> tuple[pd.DataFrame, str]:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)

    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Unsupported pkl object at {path}: {type(obj)!r}")

    if obj.shape[1] == 18:
        raw_frame = obj.copy()
        schema_type = "18col_numeric"
    elif obj.shape[1] == 1:
        raw_frame = _split_single_column_frame(obj)
        schema_type = "1col_tsv"
    else:
        raise ValueError(f"Unsupported sensor frame shape at {path}: {obj.shape}")

    raw_frame = raw_frame.reset_index(drop=True)
    raw_frame.columns = ["packet_counter"] + list(RAW_SENSOR_COLUMNS)
    raw_frame = raw_frame.apply(pd.to_numeric, errors="coerce")
    raw_frame = raw_frame.dropna(subset=["packet_counter"]).reset_index(drop=True)
    continuous_packet_counter = normalize_packet_counter(raw_frame["packet_counter"].to_numpy(dtype=np.int64))
    raw_frame["packet_counter"] = continuous_packet_counter
    raw_frame = raw_frame.drop_duplicates(subset=["packet_counter"], keep="first")
    raw_frame = raw_frame.sort_values("packet_counter").reset_index(drop=True)
    raw_frame["is_interpolated"] = False

    ordered = raw_frame.loc[:, CANONICAL_SENSOR_COLUMNS].copy()
    ordered["packet_counter"] = ordered["packet_counter"].astype(np.int64)
    ordered["is_interpolated"] = ordered["is_interpolated"].astype(bool)
    for column in RAW_SENSOR_COLUMNS:
        ordered[column] = ordered[column].astype(FLOAT_DTYPE)
    return ordered, schema_type


def fill_sensor_packet_gaps(observed_frame: pd.DataFrame) -> pd.DataFrame:
    if observed_frame.empty:
        return pd.DataFrame(columns=CANONICAL_SENSOR_COLUMNS)

    packet_counter = observed_frame["packet_counter"].to_numpy(dtype=np.int64)
    full_packet_counter = np.arange(packet_counter[0], packet_counter[-1] + 1, dtype=np.int64)
    output_frame = pd.DataFrame({"packet_counter": full_packet_counter})

    observed_positions = packet_counter - full_packet_counter[0]
    is_interpolated = np.ones(full_packet_counter.shape[0], dtype=bool)
    is_interpolated[observed_positions.astype(np.int64)] = False

    linear_columns = [
        "acc_x",
        "acc_y",
        "acc_z",
        "freeacc_x",
        "freeacc_y",
        "freeacc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "mag_x",
        "mag_y",
        "mag_z",
        "pressure",
    ]
    linear_values = observed_frame.loc[:, linear_columns].to_numpy(dtype=FLOAT_DTYPE)
    interpolated_linear = interpolate_linear_vector(packet_counter, linear_values, full_packet_counter)
    for column_index, column_name in enumerate(linear_columns):
        output_frame[column_name] = interpolated_linear[:, column_index]

    quat_values = observed_frame.loc[:, ["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=FLOAT_DTYPE)
    interpolated_quat = interpolate_quaternion(packet_counter, quat_values, full_packet_counter)
    output_frame["quat_w"] = interpolated_quat[:, 0]
    output_frame["quat_x"] = interpolated_quat[:, 1]
    output_frame["quat_y"] = interpolated_quat[:, 2]
    output_frame["quat_z"] = interpolated_quat[:, 3]
    output_frame["is_interpolated"] = is_interpolated

    return output_frame.loc[:, CANONICAL_SENSOR_COLUMNS]


def ensure_canonical_filled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in CANONICAL_SENSOR_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing canonical columns: {missing_columns}")

    canonical = frame.loc[:, CANONICAL_SENSOR_COLUMNS].copy()
    canonical["packet_counter"] = canonical["packet_counter"].astype(np.int64)
    canonical["is_interpolated"] = canonical["is_interpolated"].astype(bool)
    for column in RAW_SENSOR_COLUMNS:
        canonical[column] = canonical[column].astype(FLOAT_DTYPE)
    return canonical


def read_filled_sensor_pickle(path: Path) -> pd.DataFrame:
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"Filled sensor pkl is not a DataFrame at {path}")
    return ensure_canonical_filled_frame(frame)


def collect_d_label_vocab(annotation_root: Path) -> list[str]:
    labels = set()
    for json_path in annotation_root.glob("*/Insta360/DCIM/Camera01/*/*_annotation.json"):
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for frame in data.get("frames", []):
            for label in frame.get("fine_grained_activity", []) or []:
                if isinstance(label, str) and label.startswith("D_"):
                    labels.add(label)
    return sorted(labels)


def load_clip_labels(
    clip_dir: Path,
    label_to_index: dict[str, int],
    annotation_root: Path,
) -> tuple[np.ndarray, list[str]]:
    rows = []
    source_json_files = []
    for json_path in sort_numeric_paths(list(clip_dir.glob("*_annotation.json"))):
        with open(json_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        frames = data.get("frames", [])
        source_json_files.append(str(json_path.relative_to(annotation_root)))
        for frame in frames:
            row = np.zeros((len(label_to_index),), dtype=np.uint8)
            for label in frame.get("fine_grained_activity", []) or []:
                if isinstance(label, str) and label in label_to_index:
                    row[label_to_index[label]] = 1
            rows.append(row)

    if rows:
        labels_1hz = np.stack(rows, axis=0)
    else:
        labels_1hz = np.zeros((0, len(label_to_index)), dtype=np.uint8)
    return labels_1hz, source_json_files


def build_packet_labels(
    labels_1hz: np.ndarray,
    packet_counter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total_seconds = int(labels_1hz.shape[0])
    total_packets = int(packet_counter.shape[0])
    num_labels = int(labels_1hz.shape[1]) if labels_1hz.ndim == 2 else 0

    if total_seconds == 0:
        return (
            np.zeros((total_packets, num_labels), dtype=np.uint8),
            np.zeros((0, 2), dtype=np.int64),
        )

    boundaries = np.floor(np.linspace(0, total_packets, total_seconds + 1)).astype(np.int64)
    boundaries[0] = 0
    boundaries[-1] = total_packets

    labels_40hz = np.zeros((total_packets, num_labels), dtype=np.uint8)
    second_to_packet = np.full((total_seconds, 2), -1, dtype=np.int64)

    for second_index in range(total_seconds):
        start = int(boundaries[second_index])
        end = int(boundaries[second_index + 1])
        if end <= start:
            continue
        labels_40hz[start:end] = labels_1hz[second_index]
        second_to_packet[second_index, 0] = int(packet_counter[start])
        second_to_packet[second_index, 1] = int(packet_counter[end - 1])

    return labels_40hz, second_to_packet
