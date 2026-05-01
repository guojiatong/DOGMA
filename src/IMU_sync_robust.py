#!/usr/bin/env python3
"""
Robust IMU synchronization and denoising pipeline for the DOGMA IMU dataset.

What this script does:
1. Parse alignment.json
2. Read Xsens txt files with automatic comma/tab delimiter detection
3. Detect empty/header-only sensor files
4. Choose a robust reference sensor per segment (prefer stern, fallback to median-length sensor)
5. Apply external alignment trimming at 40 Hz
6. Normalize PacketCounter and compute a common synchronized packet range
7. Split segments into continuous chunks at large PacketCounter gaps
8. Interpolate only short/medium gaps inside each chunk
9. Apply light denoising to linear sensor channels
10. Save per-sensor npz files plus per-segment metadata and a root audit CSV

Usage:
    python IMU_sync_robust.py --alignment_json src/alignment.json --imu_path IMU --output_folder IMU_Sync_Robust
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


PACKET_COUNTER_MODULUS = 65536
IMU_RATE_HZ = 40
NUM_COLUMNS = 18
FLOAT_DTYPE = np.float32
INT_DTYPE = np.int32

EXPECTED_SENSOR_IDS = [
    "00B49A98",
    "00B49AA1",
    "00B49AA0",
    "00B49A9D",
    "00B49A93",
    "00B49A9E",
    "00B49A90",
    "00B49A9F",
    "00B49702",
    "00B49A94",
    "00B49A8F",
    "00B49A9B",
    "00B49A9C",
    "00B49A95",
    "00B49A97",
    "00B49A8E",
    "00B49A8D",
    "00B49A96",
]
PREFERRED_REFERENCE_SENSOR_ID = "00B49A94"

SMALL_GAP_MAX_PACKETS = 5
MAX_INTERPOLATED_GAP_PACKETS = 20
SEVERE_GAP_MIN_PACKETS = 1000

MEDIAN_FILTER_WINDOWS = {
    "acc": 3,
    "freeacc": 3,
    "gyr": 3,
    "mag": 3,
    "pressure": 3,
}
MOVING_AVERAGE_WINDOWS = {
    "acc": 5,
    "freeacc": 5,
    "gyr": 5,
    "mag": 3,
    "pressure": 3,
}


class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, "w", encoding="utf-8")

    def log(self, message: str) -> None:
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def close(self) -> None:
        self.log_file.close()


logger: Logger | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synchronize Xsens IMU txt files with robust gap handling."
    )
    parser.add_argument("--alignment_json", type=Path, required=True, help="Path to alignment.json")
    parser.add_argument("--imu_path", type=Path, required=True, help="Path to IMU folder")
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("IMU_Sync_Robust"),
        help="Output folder. Default: IMU_Sync_Robust beside the IMU folder.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip segments that already contain segment_metadata.json in the output folder.",
    )
    return parser.parse_args()


def parse_alignment_json(alignment_path: Path) -> dict[str, dict[str, float]]:
    with open(alignment_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    alignment_data: dict[str, dict[str, float]] = {}
    if "participants" in data:
        for participant_obj in data["participants"]:
            participant_name = participant_obj["name"]
            alignment_data[participant_name] = {}
            for segment_obj in participant_obj.get("segments", []):
                segment_num = str(segment_obj["segment"])
                alignment_integral = float(segment_obj.get("alignment_integral(seconds)", 0.0))
                alignment_data[participant_name][segment_num] = alignment_integral

    return alignment_data


def parse_data_line(line: str) -> tuple[list[str], str]:
    if "," in line:
        return [part.strip() for part in line.split(",")], "comma"
    return [part.strip() for part in line.split("\t")], "tab"


def sensor_id_from_path(txt_path: Path) -> str:
    return txt_path.stem.split("_")[-1]


def read_txt_sensor_file(txt_path: Path) -> dict[str, object]:
    rows: list[list[str]] = []
    delimiter_counter: Counter[str] = Counter()
    skipped_rows = 0

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or "PacketCounter" in stripped:
                continue

            parts, delimiter = parse_data_line(stripped)
            if len(parts) != NUM_COLUMNS:
                skipped_rows += 1
                continue

            rows.append(parts)
            delimiter_counter[delimiter] += 1

    dominant_delimiter = delimiter_counter.most_common(1)[0][0] if delimiter_counter else "none"

    if not rows:
        return {
            "txt_path": txt_path,
            "sensor_id": sensor_id_from_path(txt_path),
            "delimiter": dominant_delimiter,
            "packet_counter_raw": np.zeros((0,), dtype=np.int64),
            "values": np.zeros((0, NUM_COLUMNS - 1), dtype=FLOAT_DTYPE),
            "continuous_packet_counter": np.zeros((0,), dtype=np.int64),
            "skipped_rows": skipped_rows,
        }

    packet_counter_raw = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    values = np.asarray([[float(x) for x in row[1:]] for row in rows], dtype=FLOAT_DTYPE)
    continuous_packet_counter = normalize_packet_counter(packet_counter_raw)

    return {
        "txt_path": txt_path,
        "sensor_id": sensor_id_from_path(txt_path),
        "delimiter": dominant_delimiter,
        "packet_counter_raw": packet_counter_raw,
        "values": values,
        "continuous_packet_counter": continuous_packet_counter,
        "skipped_rows": skipped_rows,
    }


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
                    k = (
                        previous_value - current_value + PACKET_COUNTER_MODULUS - 1
                    ) // PACKET_COUNTER_MODULUS
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


def get_effective_packet_count(continuous_packet_counter: np.ndarray) -> int:
    if continuous_packet_counter.size == 0:
        return 0
    return int(continuous_packet_counter[-1] - continuous_packet_counter[0] + 1)


def choose_reference_sensor(
    sensor_data_by_file: dict[Path, dict[str, object]],
) -> tuple[Path | None, str]:
    valid_items = []
    for txt_path, sensor_data in sensor_data_by_file.items():
        continuous = sensor_data["continuous_packet_counter"]
        effective_count = get_effective_packet_count(continuous)
        if effective_count <= 0:
            continue
        valid_items.append((txt_path, sensor_data["sensor_id"], effective_count))

    if not valid_items:
        return None, "no valid sensors"

    effective_counts = np.asarray([item[2] for item in valid_items], dtype=np.int64)
    median_effective_count = int(np.median(effective_counts))

    for txt_path, sensor_id, effective_count in valid_items:
        if sensor_id == PREFERRED_REFERENCE_SENSOR_ID:
            if abs(effective_count - median_effective_count) <= MAX_INTERPOLATED_GAP_PACKETS:
                return txt_path, "preferred stern sensor"
            break

    closest_to_median = min(valid_items, key=lambda item: abs(item[2] - median_effective_count))
    txt_path, sensor_id, _ = closest_to_median
    if sensor_id == PREFERRED_REFERENCE_SENSOR_ID:
        return txt_path, "stern sensor fallback despite length mismatch"
    return txt_path, "median effective-length fallback"


def calculate_packet_sync_num(alignment_integral: float, reference_sensor_data: dict[str, object]) -> int:
    continuous = reference_sensor_data["continuous_packet_counter"]
    effective_packet_count = get_effective_packet_count(continuous)
    packet_sync_num = int(effective_packet_count - (alignment_integral - 1.0) * IMU_RATE_HZ)

    assert logger is not None
    logger.log(
        "    Formula: {0} - ({1} - 1) * {2} = {3}".format(
            effective_packet_count,
            alignment_integral,
            IMU_RATE_HZ,
            packet_sync_num,
        )
    )
    logger.log("    Effective packet count: {0}".format(effective_packet_count))
    return packet_sync_num


def calculate_trim_row_count(continuous_packet_counter: np.ndarray, effective_packets_to_trim: int) -> int:
    if effective_packets_to_trim <= 0 or continuous_packet_counter.size == 0:
        return 0

    elapsed_packets = 0
    for row_index in range(1, continuous_packet_counter.shape[0]):
        previous_packet = int(continuous_packet_counter[row_index - 1])
        current_packet = int(continuous_packet_counter[row_index])
        elapsed_packets += current_packet - previous_packet
        if elapsed_packets >= effective_packets_to_trim:
            return row_index

    return int(continuous_packet_counter.shape[0])


def trim_sensor_data(sensor_data: dict[str, object], packet_sync_num: int) -> dict[str, object]:
    packet_counter_raw = sensor_data["packet_counter_raw"]
    values = sensor_data["values"]
    continuous = sensor_data["continuous_packet_counter"]

    if packet_counter_raw.size == 0:
        return dict(sensor_data)

    trim_row_count = calculate_trim_row_count(continuous, packet_sync_num) if packet_sync_num > 0 else 0

    trimmed_packet_counter_raw = packet_counter_raw[trim_row_count:]
    trimmed_values = values[trim_row_count:]
    trimmed_continuous = continuous[trim_row_count:]

    trimmed = dict(sensor_data)
    trimmed["packet_counter_raw"] = trimmed_packet_counter_raw
    trimmed["values"] = trimmed_values
    trimmed["continuous_packet_counter"] = trimmed_continuous
    trimmed["trim_row_count"] = trim_row_count
    return trimmed


def extract_gap_records(continuous_packet_counter: np.ndarray) -> list[dict[str, int]]:
    gap_records: list[dict[str, int]] = []
    if continuous_packet_counter.size <= 1:
        return gap_records

    for row_index in range(continuous_packet_counter.shape[0] - 1):
        left_packet = int(continuous_packet_counter[row_index])
        right_packet = int(continuous_packet_counter[row_index + 1])
        missing_packets = right_packet - left_packet - 1
        if missing_packets <= 0:
            continue
        gap_records.append(
            {
                "row_index": row_index,
                "left_packet": left_packet,
                "right_packet": right_packet,
                "missing_packets": missing_packets,
            }
        )
    return gap_records


def summarize_gap_records(gap_records: list[dict[str, int]]) -> dict[str, int]:
    small = 0
    medium = 0
    large = 0
    severe = 0
    max_gap = 0
    for record in gap_records:
        missing_packets = int(record["missing_packets"])
        max_gap = max(max_gap, missing_packets)
        if missing_packets <= SMALL_GAP_MAX_PACKETS:
            small += 1
        elif missing_packets <= MAX_INTERPOLATED_GAP_PACKETS:
            medium += 1
        else:
            large += 1
            if missing_packets >= SEVERE_GAP_MIN_PACKETS:
                severe += 1
    return {
        "small_gap_count": small,
        "medium_gap_count": medium,
        "large_gap_count": large,
        "severe_gap_count": severe,
        "max_gap_packets": max_gap,
    }


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []

    sorted_intervals = sorted(intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def complement_intervals(start: int, end: int, excluded: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if start > end:
        return []

    chunks: list[tuple[int, int]] = []
    current_start = start
    for excluded_start, excluded_end in excluded:
        if excluded_end < current_start:
            continue
        if excluded_start > end:
            break
        if current_start < excluded_start:
            chunks.append((current_start, excluded_start - 1))
        current_start = max(current_start, excluded_end + 1)
    if current_start <= end:
        chunks.append((current_start, end))
    return chunks


def build_common_chunks(
    trimmed_sensor_data_by_file: dict[Path, dict[str, object]],
) -> tuple[list[tuple[int, int]], int | None, int | None]:
    valid_sensors = [
        sensor_data
        for sensor_data in trimmed_sensor_data_by_file.values()
        if sensor_data["continuous_packet_counter"].size > 0
    ]
    if not valid_sensors:
        return [], None, None

    common_start = max(int(sensor_data["continuous_packet_counter"][0]) for sensor_data in valid_sensors)
    common_end = min(int(sensor_data["continuous_packet_counter"][-1]) for sensor_data in valid_sensors)
    if common_start > common_end:
        return [], common_start, common_end

    excluded_intervals: list[tuple[int, int]] = []
    for sensor_data in valid_sensors:
        for gap_record in extract_gap_records(sensor_data["continuous_packet_counter"]):
            missing_packets = int(gap_record["missing_packets"])
            if missing_packets <= MAX_INTERPOLATED_GAP_PACKETS:
                continue
            excluded_start = max(common_start, int(gap_record["left_packet"]) + 1)
            excluded_end = min(common_end, int(gap_record["right_packet"]) - 1)
            if excluded_start <= excluded_end:
                excluded_intervals.append((excluded_start, excluded_end))

    merged_exclusions = merge_intervals(excluded_intervals)
    return complement_intervals(common_start, common_end, merged_exclusions), common_start, common_end


def build_support_arrays(
    continuous_packet_counter: np.ndarray,
    values: np.ndarray,
    chunk_start: int,
    chunk_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    if continuous_packet_counter.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, values.shape[1]), dtype=FLOAT_DTYPE)

    inside_mask = (continuous_packet_counter >= chunk_start) & (continuous_packet_counter <= chunk_end)
    inside_indices = np.flatnonzero(inside_mask)
    if inside_indices.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, values.shape[1]), dtype=FLOAT_DTYPE)

    support_start = int(inside_indices[0])
    support_end = int(inside_indices[-1])

    if support_start > 0:
        previous_packet = int(continuous_packet_counter[support_start - 1])
        current_packet = int(continuous_packet_counter[support_start])
        if current_packet - previous_packet - 1 <= MAX_INTERPOLATED_GAP_PACKETS:
            support_start -= 1

    if support_end + 1 < continuous_packet_counter.shape[0]:
        current_packet = int(continuous_packet_counter[support_end])
        next_packet = int(continuous_packet_counter[support_end + 1])
        if next_packet - current_packet - 1 <= MAX_INTERPOLATED_GAP_PACKETS:
            support_end += 1

    support_slice = slice(support_start, support_end + 1)
    return continuous_packet_counter[support_slice], values[support_slice]


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
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
    support_packet_counter: np.ndarray,
    values: np.ndarray,
    full_packet_counter: np.ndarray,
    output_dim: int,
) -> np.ndarray:
    output = np.zeros((full_packet_counter.shape[0], output_dim), dtype=FLOAT_DTYPE)
    x = support_packet_counter.astype(np.float64)
    x_full = full_packet_counter.astype(np.float64)

    for dim in range(output_dim):
        y = values[:, dim].astype(np.float64)
        output[:, dim] = np.interp(x_full, x, y).astype(FLOAT_DTYPE)

    return output


def interpolate_quaternion(
    support_packet_counter: np.ndarray,
    quat_values: np.ndarray,
    full_packet_counter: np.ndarray,
) -> np.ndarray:
    output = np.zeros((full_packet_counter.shape[0], 4), dtype=FLOAT_DTYPE)
    positions = support_packet_counter.astype(np.int64) - int(full_packet_counter[0])
    normalized_quats = np.asarray(
        [normalize_quaternion(quat_values[i]) for i in range(quat_values.shape[0])],
        dtype=FLOAT_DTYPE,
    )

    if positions.size == 0:
        return output
    if positions.size == 1:
        only_quat = normalized_quats[0]
        output[:] = only_quat
        return output

    for target_index in range(output.shape[0]):
        if target_index <= positions[0]:
            output[target_index] = normalized_quats[0]
            continue
        if target_index >= positions[-1]:
            output[target_index] = normalized_quats[-1]
            continue

        right_index = int(np.searchsorted(positions, target_index, side="left"))
        if positions[right_index] == target_index:
            output[target_index] = normalized_quats[right_index]
            continue

        left_index = right_index - 1
        left_pos = int(positions[left_index])
        right_pos = int(positions[right_index])
        interpolation_t = float(target_index - left_pos) / float(right_pos - left_pos)
        output[target_index] = slerp_quaternion(
            normalized_quats[left_index],
            normalized_quats[right_index],
            interpolation_t,
        )

    return output.astype(FLOAT_DTYPE)


def enforce_quaternion_continuity(quat: np.ndarray) -> np.ndarray:
    if quat.shape[0] == 0:
        return quat
    output = quat.copy()
    output[0] = normalize_quaternion(output[0])
    for i in range(1, output.shape[0]):
        current = normalize_quaternion(output[i])
        previous = output[i - 1]
        if float(np.dot(previous, current)) < 0.0:
            current = -current
        output[i] = current
    return output.astype(FLOAT_DTYPE)


def median_filter_1d(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.shape[0] <= 1:
        return values.copy()
    radius = window_size // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    filtered = np.empty_like(values)
    for index in range(values.shape[0]):
        filtered[index] = np.median(padded[index : index + window_size])
    return filtered


def median_filter_array(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.shape[0] <= 1:
        return values.copy()
    filtered = np.empty_like(values)
    for dim in range(values.shape[1]):
        filtered[:, dim] = median_filter_1d(values[:, dim], window_size)
    return filtered


def moving_average_array(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.shape[0] <= 1:
        return values.copy()
    radius = window_size // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed = np.empty_like(values, dtype=np.float64)
    for dim in range(values.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return smoothed.astype(values.dtype)


def denoise_linear_signal(values: np.ndarray, signal_name: str) -> np.ndarray:
    median_window = MEDIAN_FILTER_WINDOWS[signal_name]
    average_window = MOVING_AVERAGE_WINDOWS[signal_name]
    filtered = median_filter_array(values, median_window)
    return moving_average_array(filtered, average_window)


def empty_sensor_arrays() -> dict[str, np.ndarray]:
    return {
        "packet_counter": np.zeros((0,), dtype=np.int64),
        "clip_id": np.zeros((0,), dtype=INT_DTYPE),
        "id": np.zeros((0,), dtype=INT_DTYPE),
        "is_interpolated": np.zeros((0,), dtype=bool),
        "is_low_confidence": np.zeros((0,), dtype=bool),
        "acc": np.zeros((0, 3), dtype=FLOAT_DTYPE),
        "freeacc": np.zeros((0, 3), dtype=FLOAT_DTYPE),
        "gyr": np.zeros((0, 3), dtype=FLOAT_DTYPE),
        "mag": np.zeros((0, 3), dtype=FLOAT_DTYPE),
        "pressure": np.zeros((0, 1), dtype=FLOAT_DTYPE),
        "quat": np.zeros((0, 4), dtype=FLOAT_DTYPE),
    }


def build_chunk_arrays(
    sensor_data: dict[str, object],
    chunk_start: int,
    chunk_end: int,
) -> dict[str, np.ndarray] | None:
    continuous = sensor_data["continuous_packet_counter"]
    values = sensor_data["values"]
    support_packet_counter, support_values = build_support_arrays(continuous, values, chunk_start, chunk_end)
    if support_packet_counter.size == 0:
        return None

    full_packet_counter = np.arange(chunk_start, chunk_end + 1, dtype=np.int64)
    inside_mask = (continuous >= chunk_start) & (continuous <= chunk_end)
    observed_packet_counter = continuous[inside_mask]

    observed_lookup = set(int(packet) for packet in observed_packet_counter.tolist())
    is_interpolated = np.asarray(
        [int(packet) not in observed_lookup for packet in full_packet_counter],
        dtype=bool,
    )
    is_low_confidence = np.zeros(full_packet_counter.shape[0], dtype=bool)

    for gap_record in extract_gap_records(observed_packet_counter):
        missing_packets = int(gap_record["missing_packets"])
        if missing_packets <= SMALL_GAP_MAX_PACKETS or missing_packets > MAX_INTERPOLATED_GAP_PACKETS:
            continue
        low_confidence_start = int(gap_record["left_packet"]) + 1
        low_confidence_end = int(gap_record["right_packet"]) - 1
        start_index = low_confidence_start - chunk_start
        end_index = low_confidence_end - chunk_start + 1
        is_low_confidence[start_index:end_index] = True

    acc = interpolate_linear_vector(support_packet_counter, support_values[:, 0:3], full_packet_counter, 3)
    freeacc = interpolate_linear_vector(support_packet_counter, support_values[:, 3:6], full_packet_counter, 3)
    gyr = interpolate_linear_vector(support_packet_counter, support_values[:, 6:9], full_packet_counter, 3)
    mag = interpolate_linear_vector(support_packet_counter, support_values[:, 9:12], full_packet_counter, 3)
    pressure = interpolate_linear_vector(
        support_packet_counter,
        support_values[:, 12:13],
        full_packet_counter,
        1,
    )
    quat = interpolate_quaternion(support_packet_counter, support_values[:, 13:17], full_packet_counter)

    return {
        "packet_counter": full_packet_counter,
        "is_interpolated": is_interpolated,
        "is_low_confidence": is_low_confidence,
        "acc": denoise_linear_signal(acc, "acc"),
        "freeacc": denoise_linear_signal(freeacc, "freeacc"),
        "gyr": denoise_linear_signal(gyr, "gyr"),
        "mag": denoise_linear_signal(mag, "mag"),
        "pressure": denoise_linear_signal(pressure, "pressure"),
        "quat": enforce_quaternion_continuity(quat),
    }


def build_sensor_arrays(
    sensor_data: dict[str, object],
    chunks: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    if sensor_data["continuous_packet_counter"].size == 0 or not chunks:
        return empty_sensor_arrays()

    packet_counter_parts = []
    clip_id_parts = []
    local_id_parts = []
    is_interpolated_parts = []
    is_low_confidence_parts = []
    acc_parts = []
    freeacc_parts = []
    gyr_parts = []
    mag_parts = []
    pressure_parts = []
    quat_parts = []

    for clip_index, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_arrays = build_chunk_arrays(sensor_data, chunk_start, chunk_end)
        if chunk_arrays is None:
            continue

        chunk_length = chunk_arrays["packet_counter"].shape[0]
        packet_counter_parts.append(chunk_arrays["packet_counter"])
        clip_id_parts.append(np.full(chunk_length, clip_index, dtype=INT_DTYPE))
        local_id_parts.append(np.arange(chunk_length, dtype=INT_DTYPE))
        is_interpolated_parts.append(chunk_arrays["is_interpolated"])
        is_low_confidence_parts.append(chunk_arrays["is_low_confidence"])
        acc_parts.append(chunk_arrays["acc"])
        freeacc_parts.append(chunk_arrays["freeacc"])
        gyr_parts.append(chunk_arrays["gyr"])
        mag_parts.append(chunk_arrays["mag"])
        pressure_parts.append(chunk_arrays["pressure"])
        quat_parts.append(chunk_arrays["quat"])

    if not packet_counter_parts:
        return empty_sensor_arrays()

    return {
        "packet_counter": np.concatenate(packet_counter_parts, axis=0),
        "clip_id": np.concatenate(clip_id_parts, axis=0),
        "id": np.concatenate(local_id_parts, axis=0),
        "is_interpolated": np.concatenate(is_interpolated_parts, axis=0),
        "is_low_confidence": np.concatenate(is_low_confidence_parts, axis=0),
        "acc": np.concatenate(acc_parts, axis=0).astype(FLOAT_DTYPE),
        "freeacc": np.concatenate(freeacc_parts, axis=0).astype(FLOAT_DTYPE),
        "gyr": np.concatenate(gyr_parts, axis=0).astype(FLOAT_DTYPE),
        "mag": np.concatenate(mag_parts, axis=0).astype(FLOAT_DTYPE),
        "pressure": np.concatenate(pressure_parts, axis=0).astype(FLOAT_DTYPE),
        "quat": np.concatenate(quat_parts, axis=0).astype(FLOAT_DTYPE),
    }


def save_sensor_npz(
    output_npz_path: Path,
    sensor_name: str,
    sensor_id: str,
    delimiter: str,
    alignment_available: bool,
    segment_quality: str,
    sensor_arrays: dict[str, np.ndarray],
) -> None:
    np.savez_compressed(
        output_npz_path,
        sensor_name=np.asarray(sensor_name),
        sensor_id=np.asarray(sensor_id),
        delimiter=np.asarray(delimiter),
        rate_hz=np.asarray(IMU_RATE_HZ, dtype=INT_DTYPE),
        packet_counter_modulus=np.asarray(PACKET_COUNTER_MODULUS, dtype=INT_DTYPE),
        alignment_available=np.asarray(alignment_available),
        segment_quality=np.asarray(segment_quality),
        packet_counter=sensor_arrays["packet_counter"],
        clip_id=sensor_arrays["clip_id"],
        id=sensor_arrays["id"],
        is_interpolated=sensor_arrays["is_interpolated"],
        is_low_confidence=sensor_arrays["is_low_confidence"],
        acc=sensor_arrays["acc"],
        freeacc=sensor_arrays["freeacc"],
        gyr=sensor_arrays["gyr"],
        mag=sensor_arrays["mag"],
        pressure=sensor_arrays["pressure"],
        quat=sensor_arrays["quat"],
    )


def determine_segment_quality(
    alignment_available: bool,
    missing_sensor_ids: list[str],
    empty_sensor_ids: list[str],
    gap_summary_by_sensor_id: dict[str, dict[str, int]],
    chunks: list[tuple[int, int]],
) -> str:
    if not chunks:
        return "C"

    has_severe_gap = any(summary["severe_gap_count"] > 0 for summary in gap_summary_by_sensor_id.values())
    has_break_gap = any(summary["large_gap_count"] > 0 for summary in gap_summary_by_sensor_id.values())
    has_missing = bool(missing_sensor_ids or empty_sensor_ids)
    has_medium_gap = any(summary["medium_gap_count"] > 0 for summary in gap_summary_by_sensor_id.values())

    if has_severe_gap:
        return "C"
    if has_break_gap or has_missing or not alignment_available or has_medium_gap:
        return "B"
    return "A"


def build_segment_metadata(
    segment_path: Path,
    alignment_available: bool,
    alignment_integral: float | None,
    reference_txt_path: Path | None,
    reference_reason: str,
    packet_sync_num: int,
    sensor_data_by_file: dict[Path, dict[str, object]],
    trimmed_sensor_data_by_file: dict[Path, dict[str, object]],
    chunks: list[tuple[int, int]],
    common_start: int | None,
    common_end: int | None,
) -> dict[str, object]:
    present_sensor_ids = sorted(sensor_data["sensor_id"] for sensor_data in sensor_data_by_file.values())
    valid_sensor_ids = sorted(
        sensor_data["sensor_id"]
        for sensor_data in trimmed_sensor_data_by_file.values()
        if sensor_data["continuous_packet_counter"].size > 0
    )
    empty_sensor_ids = sorted(
        sensor_data["sensor_id"]
        for sensor_data in sensor_data_by_file.values()
        if sensor_data["continuous_packet_counter"].size == 0
    )
    missing_sensor_ids = sorted(set(EXPECTED_SENSOR_IDS) - set(present_sensor_ids))

    sensor_status = {}
    for sensor_id in EXPECTED_SENSOR_IDS:
        if sensor_id in valid_sensor_ids:
            sensor_status[sensor_id] = "valid"
        elif sensor_id in empty_sensor_ids:
            sensor_status[sensor_id] = "empty"
        elif sensor_id in present_sensor_ids:
            sensor_status[sensor_id] = "present_no_valid_rows"
        else:
            sensor_status[sensor_id] = "missing"

    gap_summary_by_sensor_id: dict[str, dict[str, int]] = {}
    for sensor_data in trimmed_sensor_data_by_file.values():
        sensor_id = sensor_data["sensor_id"]
        gap_summary_by_sensor_id[sensor_id] = summarize_gap_records(
            extract_gap_records(sensor_data["continuous_packet_counter"])
        )

    segment_quality = determine_segment_quality(
        alignment_available=alignment_available,
        missing_sensor_ids=missing_sensor_ids,
        empty_sensor_ids=empty_sensor_ids,
        gap_summary_by_sensor_id=gap_summary_by_sensor_id,
        chunks=chunks,
    )

    total_output_packets = int(sum(chunk_end - chunk_start + 1 for chunk_start, chunk_end in chunks))
    return {
        "segment_path": str(segment_path),
        "segment_name": segment_path.name,
        "participant_folder": segment_path.parent.name,
        "alignment_available": alignment_available,
        "alignment_integral_seconds": alignment_integral,
        "reference_sensor_id": sensor_id_from_path(reference_txt_path) if reference_txt_path is not None else None,
        "reference_sensor_file": reference_txt_path.name if reference_txt_path is not None else None,
        "reference_sensor_reason": reference_reason,
        "packet_sync_num": int(packet_sync_num),
        "present_sensor_ids": present_sensor_ids,
        "valid_sensor_ids": valid_sensor_ids,
        "empty_sensor_ids": empty_sensor_ids,
        "missing_sensor_ids": missing_sensor_ids,
        "sensor_status": sensor_status,
        "common_packet_start": common_start,
        "common_packet_end": common_end,
        "num_chunks": len(chunks),
        "chunks": [
            {"clip_id": clip_id, "start_packet": int(chunk_start), "end_packet": int(chunk_end)}
            for clip_id, (chunk_start, chunk_end) in enumerate(chunks)
        ],
        "total_output_packets": total_output_packets,
        "gap_summary_by_sensor_id": gap_summary_by_sensor_id,
        "segment_quality": segment_quality,
    }


def write_segment_metadata(output_segment_path: Path, metadata: dict[str, object]) -> None:
    metadata_path = output_segment_path / "segment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)


def read_segment_metadata(metadata_path: Path) -> dict[str, object]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def process_segment(
    segment_path: Path,
    alignment_integral: float | None,
    output_segment_path: Path,
) -> dict[str, object]:
    txt_files = sorted(segment_path.glob("*.txt"))
    output_segment_path.mkdir(parents=True, exist_ok=True)

    mtb_files = sorted(segment_path.glob("*.mtb"))
    for mtb_file in mtb_files:
        shutil.copy2(mtb_file, output_segment_path / mtb_file.name)

    assert logger is not None
    logger.log("  Step 1: Reading txt sensor files...")
    sensor_data_by_file: dict[Path, dict[str, object]] = {}
    for txt_file in txt_files:
        sensor_data = read_txt_sensor_file(txt_file)
        sensor_data_by_file[txt_file] = sensor_data
        logger.log(
            "    Loaded: {0} ({1} rows, delimiter={2})".format(
                txt_file.name,
                sensor_data["packet_counter_raw"].shape[0],
                sensor_data["delimiter"],
            )
        )

    reference_txt_path, reference_reason = choose_reference_sensor(sensor_data_by_file)
    if reference_txt_path is None:
        logger.log("  No valid sensor rows found in segment; writing empty metadata only.")
        metadata = build_segment_metadata(
            segment_path=segment_path,
            alignment_available=alignment_integral is not None,
            alignment_integral=alignment_integral,
            reference_txt_path=None,
            reference_reason=reference_reason,
            packet_sync_num=0,
            sensor_data_by_file=sensor_data_by_file,
            trimmed_sensor_data_by_file=sensor_data_by_file,
            chunks=[],
            common_start=None,
            common_end=None,
        )
        write_segment_metadata(output_segment_path, metadata)
        return metadata

    packet_sync_num = 0
    alignment_available = alignment_integral is not None
    if alignment_available:
        logger.log(
            "  Step 2: Calculating packet_sync_num from {0} ({1})...".format(
                reference_txt_path.name,
                reference_reason,
            )
        )
        packet_sync_num = calculate_packet_sync_num(
            alignment_integral=float(alignment_integral),
            reference_sensor_data=sensor_data_by_file[reference_txt_path],
        )
    else:
        logger.log("  Step 2: No alignment entry found; keeping all packets and marking external_alignment=false.")

    logger.log("  Step 3: Trimming sensor streams...")
    trimmed_sensor_data_by_file = {
        txt_path: trim_sensor_data(sensor_data, packet_sync_num)
        for txt_path, sensor_data in sensor_data_by_file.items()
    }

    logger.log("  Step 4: Building synchronized chunks...")
    chunks, common_start, common_end = build_common_chunks(trimmed_sensor_data_by_file)
    logger.log(
        "    Common packet range: start={0}, end={1}, num_chunks={2}".format(
            common_start,
            common_end,
            len(chunks),
        )
    )

    metadata = build_segment_metadata(
        segment_path=segment_path,
        alignment_available=alignment_available,
        alignment_integral=alignment_integral,
        reference_txt_path=reference_txt_path,
        reference_reason=reference_reason,
        packet_sync_num=packet_sync_num,
        sensor_data_by_file=sensor_data_by_file,
        trimmed_sensor_data_by_file=trimmed_sensor_data_by_file,
        chunks=chunks,
        common_start=common_start,
        common_end=common_end,
    )
    segment_quality = str(metadata["segment_quality"])

    logger.log("  Step 5: Interpolating, denoising, and saving npz files...")
    for txt_file, sensor_data in trimmed_sensor_data_by_file.items():
        sensor_arrays = build_sensor_arrays(sensor_data, chunks)
        output_npz_path = output_segment_path / "{0}.npz".format(txt_file.stem)
        save_sensor_npz(
            output_npz_path=output_npz_path,
            sensor_name=txt_file.stem,
            sensor_id=str(sensor_data["sensor_id"]),
            delimiter=str(sensor_data["delimiter"]),
            alignment_available=alignment_available,
            segment_quality=segment_quality,
            sensor_arrays=sensor_arrays,
        )
        logger.log(
            "    Saved: {0} ({1} frames, {2} interpolated, {3} low-confidence)".format(
                output_npz_path.name,
                sensor_arrays["packet_counter"].shape[0],
                int(sensor_arrays["is_interpolated"].sum()),
                int(sensor_arrays["is_low_confidence"].sum()),
            )
        )

    write_segment_metadata(output_segment_path, metadata)
    logger.log("    Wrote segment metadata: {0}".format((output_segment_path / "segment_metadata.json").name))
    return metadata


def write_audit_csv(output_path: Path, metadata_rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "participant_folder",
        "segment_name",
        "segment_quality",
        "alignment_available",
        "alignment_integral_seconds",
        "reference_sensor_id",
        "reference_sensor_reason",
        "packet_sync_num",
        "present_sensor_count",
        "valid_sensor_count",
        "empty_sensor_count",
        "missing_sensor_count",
        "common_packet_start",
        "common_packet_end",
        "num_chunks",
        "total_output_packets",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metadata in metadata_rows:
            writer.writerow(
                {
                    "participant_folder": metadata["participant_folder"],
                    "segment_name": metadata["segment_name"],
                    "segment_quality": metadata["segment_quality"],
                    "alignment_available": metadata["alignment_available"],
                    "alignment_integral_seconds": metadata["alignment_integral_seconds"],
                    "reference_sensor_id": metadata["reference_sensor_id"],
                    "reference_sensor_reason": metadata["reference_sensor_reason"],
                    "packet_sync_num": metadata["packet_sync_num"],
                    "present_sensor_count": len(metadata["present_sensor_ids"]),
                    "valid_sensor_count": len(metadata["valid_sensor_ids"]),
                    "empty_sensor_count": len(metadata["empty_sensor_ids"]),
                    "missing_sensor_count": len(metadata["missing_sensor_ids"]),
                    "common_packet_start": metadata["common_packet_start"],
                    "common_packet_end": metadata["common_packet_end"],
                    "num_chunks": metadata["num_chunks"],
                    "total_output_packets": metadata["total_output_packets"],
                }
            )


def main() -> None:
    global logger

    args = parse_args()
    alignment_json_path = args.alignment_json
    imu_path = args.imu_path

    if not alignment_json_path.exists():
        raise FileNotFoundError("alignment.json not found at {0}".format(alignment_json_path))
    if not imu_path.exists():
        raise FileNotFoundError("IMU folder not found at {0}".format(imu_path))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = imu_path.parent / "IMU_sync_robust_log_{0}.txt".format(timestamp)
    logger = Logger(log_path)

    if args.output_folder.is_absolute():
        output_imu_path = args.output_folder
    else:
        output_imu_path = imu_path.parent / args.output_folder
    output_imu_path.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 60)
    logger.log("Robust IMU Data Synchronization Script")
    logger.log("Started at: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    logger.log("=" * 60)
    logger.log("Alignment JSON: {0}".format(alignment_json_path))
    logger.log("IMU folder: {0}".format(imu_path))
    logger.log("Output folder: {0}".format(output_imu_path))
    logger.log("Log file: {0}".format(log_path))
    logger.log("")

    alignment_data = parse_alignment_json(alignment_json_path)
    logger.log("Parsed alignment.json for {0} participants".format(len(alignment_data)))
    logger.log("")

    participants_folder = imu_path / "Participants"
    if not participants_folder.exists():
        logger.close()
        raise FileNotFoundError("Participants folder not found at {0}".format(participants_folder))

    metadata_rows: list[dict[str, object]] = []

    for participant_folder in sorted(participants_folder.glob("*")):
        if not participant_folder.is_dir():
            continue

        folder_name = participant_folder.name
        logger.log("\nProcessing folder: {0}".format(folder_name))

        if "-" in folder_name:
            participant_name = folder_name.split("-")[0].lower()
        else:
            logger.log("  Skipping special-case folder (no hyphen): {0}".format(folder_name))
            continue

        matching_key = None
        for key in alignment_data.keys():
            if key.lower() == participant_name:
                matching_key = key
                break

        for segment_folder in sorted(participant_folder.glob("*")):
            if not segment_folder.is_dir():
                continue

            segment_num = segment_folder.name
            alignment_integral = None
            if matching_key is not None:
                alignment_integral = alignment_data[matching_key].get(str(segment_num))

            logger.log("  Processing segment: {0}".format(segment_num))
            if alignment_integral is None:
                logger.log("    Alignment not found for this segment; falling back to packet-only synchronization.")

            output_participant_path = output_imu_path / folder_name
            output_segment_path = output_participant_path / segment_num
            metadata_path = output_segment_path / "segment_metadata.json"

            if args.skip_existing and metadata_path.exists():
                logger.log("    Skipping existing segment output: {0}".format(metadata_path))
                metadata_rows.append(read_segment_metadata(metadata_path))
                continue

            metadata = process_segment(segment_folder, alignment_integral, output_segment_path)
            metadata_rows.append(metadata)

    audit_csv_path = output_imu_path / "sync_audit.csv"
    write_audit_csv(audit_csv_path, metadata_rows)
    logger.log("")
    logger.log("Wrote audit CSV: {0}".format(audit_csv_path))
    logger.log("=" * 60)
    logger.log("Processing complete")
    logger.log("Finished at: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    logger.log("=" * 60)
    logger.close()


if __name__ == "__main__":
    main()
