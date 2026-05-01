#!/usr/bin/env python3
"""
Self-contained dataloader for DOGMA raw-IMU latent diffusion.

Canonical task:
- condition: past 6s raw IMU feature at 20Hz -> [120, 130]
- target: 8s combined raw IMU feature window -> [160, 130]

Expected shared data bundle:
- features_20hz/
- manifests_8s160f_remote/vae_window_index_{train,val,test}.csv

This version still reads the same VAE window manifest, but only uses
`feature_path` for the actual training target.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


NUM_SENSORS = 10
RAW_IMU_SENSOR_DIM = 13
RAW_IMU_INPUT_DIM = NUM_SENSORS * RAW_IMU_SENSOR_DIM
CONDITION_DIM = RAW_IMU_INPUT_DIM


@dataclass(frozen=True)
class FeatureWindowRecord:
    participant: str
    segment_id: str
    pose_path: Path
    feature_path: Path
    start_frame: int
    valid_frames: int
    packet_start_40hz: int
    packet_end_40hz: int
    interp_ratio: float
    task: str


def _normalize_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip()


def _parse_int(value: Any, default: int = 0) -> int:
    text = _normalize_text(value)
    if not text or text.lower() == "nan":
        return default
    return int(float(text))


def _parse_float(value: Any, default: float = 0.0) -> float:
    text = _normalize_text(value)
    if not text or text.lower() == "nan":
        return default
    return float(text)


def _normalize_prefix_map(
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    if path_prefix_map is None:
        return ()
    if isinstance(path_prefix_map, Mapping):
        items = path_prefix_map.items()
    else:
        items = path_prefix_map
    normalized: list[tuple[str, str]] = []
    for source_prefix, target_prefix in items:
        source_text = str(source_prefix).rstrip("/")
        target_text = str(target_prefix).rstrip("/")
        if source_text and target_text:
            normalized.append((source_text, target_text))
    return tuple(normalized)


def _apply_prefix_map(path: Path, prefix_map: tuple[tuple[str, str], ...]) -> Path:
    path_text = str(path)
    for source_prefix, target_prefix in prefix_map:
        if path_text == source_prefix or path_text.startswith(source_prefix + "/"):
            return Path(target_prefix + path_text[len(source_prefix) :])
    return path


def _build_feature_path(
    *,
    data_root: Path,
    participant: str,
    segment_id: str,
) -> Path:
    return Path(data_root) / "features_20hz" / participant / f"segment_{segment_id}.npz"


def resolve_feature_path(
    original_path: str | Path,
    *,
    participant: str,
    segment_id: str,
    data_root: str | Path | None = None,
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> Path:
    path = Path(original_path)
    prefix_map = _normalize_prefix_map(path_prefix_map)
    if data_root is not None:
        return _build_feature_path(
            data_root=Path(data_root),
            participant=participant,
            segment_id=segment_id,
        )
    return _apply_prefix_map(path, prefix_map)


def resolve_optional_pose_path(
    original_path: str | Path,
    *,
    participant: str,
    segment_id: str,
    data_root: str | Path | None = None,
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> Path:
    path = Path(original_path)
    prefix_map = _normalize_prefix_map(path_prefix_map)
    if data_root is not None:
        return Path(data_root) / "pseudo_pose_20hz" / participant / f"segment_{segment_id}.npz"
    return _apply_prefix_map(path, prefix_map)


def read_feature_window_records_from_csv(
    window_index_csv: str | Path,
    *,
    data_root: str | Path | None = None,
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> list[FeatureWindowRecord]:
    records: list[FeatureWindowRecord] = []
    with Path(window_index_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            participant = _normalize_text(row.get("participant", ""))
            segment_id = _normalize_text(row.get("segment_id", ""))
            start_frame = _parse_int(row.get("start_frame_20hz"))
            end_frame = _parse_int(row.get("end_frame_20hz"))
            feature_path = resolve_feature_path(
                row["feature_path"],
                participant=participant,
                segment_id=segment_id,
                data_root=data_root,
                path_prefix_map=path_prefix_map,
            )
            pose_path = resolve_optional_pose_path(
                row.get("pose_path", ""),
                participant=participant,
                segment_id=segment_id,
                data_root=data_root,
                path_prefix_map=path_prefix_map,
            )
            records.append(
                FeatureWindowRecord(
                    participant=participant,
                    segment_id=segment_id,
                    pose_path=pose_path,
                    feature_path=feature_path,
                    start_frame=start_frame,
                    valid_frames=end_frame - start_frame + 1,
                    packet_start_40hz=_parse_int(row.get("packet_start_40hz")),
                    packet_end_40hz=_parse_int(row.get("packet_end_40hz")),
                    interp_ratio=_parse_float(row.get("interp_ratio")),
                    task=_normalize_text(row.get("task", "")),
                )
            )
    if not records:
        raise ValueError(f"No windows found in {window_index_csv}")
    return records


def select_feature_window_records(
    records: Sequence[FeatureWindowRecord],
    *,
    max_windows: int = 0,
    shuffle: bool = False,
    subset_seed: int = 0,
) -> list[FeatureWindowRecord]:
    selected = list(records)
    if shuffle and len(selected) > 1:
        rng = np.random.default_rng(subset_seed)
        order = rng.permutation(len(selected)).tolist()
        selected = [selected[int(index)] for index in order]
    if max_windows > 0:
        selected = selected[:max_windows]
    return selected


def flatten_feature_window(feature_window: np.ndarray) -> np.ndarray:
    feature_window = np.asarray(feature_window, dtype=np.float32)
    if feature_window.ndim != 3 or feature_window.shape[1:] != (NUM_SENSORS, RAW_IMU_SENSOR_DIM):
        raise ValueError(
            f"feature_window must have shape [T,{NUM_SENSORS},{RAW_IMU_SENSOR_DIM}], got {feature_window.shape}"
        )
    return feature_window.reshape(feature_window.shape[0], RAW_IMU_INPUT_DIM).astype(np.float32)


class RawImuLatentDiffusionDataset(Dataset):
    def __init__(
        self,
        *,
        window_records: Sequence[FeatureWindowRecord],
        past_frames: int = 120,
        future_window_frames: int = 40,
        validate_paths: bool = False,
    ) -> None:
        self.past_frames = int(past_frames)
        self.future_window_frames = int(future_window_frames)
        self.window_records = [
            record
            for record in window_records
            if record.start_frame >= self.past_frames and record.valid_frames >= self.future_window_frames
        ]
        if not self.window_records:
            raise ValueError("No raw IMU latent diffusion windows available")
        self._feature_cache: dict[Path, dict[str, np.ndarray]] = {}
        if validate_paths:
            self._validate_unique_paths_exist()

    @classmethod
    def from_window_index_csv(
        cls,
        *,
        window_index_csv: str | Path,
        past_frames: int = 120,
        future_window_frames: int = 40,
        max_windows: int = 0,
        shuffle: bool = False,
        subset_seed: int = 0,
        data_root: str | Path | None = None,
        path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        validate_paths: bool = False,
    ) -> "RawImuLatentDiffusionDataset":
        records = read_feature_window_records_from_csv(
            window_index_csv,
            data_root=data_root,
            path_prefix_map=path_prefix_map,
        )
        records = [
            record
            for record in records
            if record.start_frame >= int(past_frames) and record.valid_frames >= int(future_window_frames)
        ]
        records = select_feature_window_records(
            records,
            max_windows=max_windows,
            shuffle=shuffle,
            subset_seed=subset_seed,
        )
        return cls(
            window_records=records,
            past_frames=past_frames,
            future_window_frames=future_window_frames,
            validate_paths=validate_paths,
        )

    def _validate_unique_paths_exist(self) -> None:
        unique_paths = {record.feature_path for record in self.window_records}
        missing_paths = [path for path in sorted(unique_paths) if not path.exists()]
        if missing_paths:
            preview = ", ".join(str(path) for path in missing_paths[:5])
            raise FileNotFoundError(f"Resolved dataset paths do not exist. First missing paths: {preview}")

    def _load_feature_payload(self, feature_path: Path) -> dict[str, np.ndarray]:
        if feature_path not in self._feature_cache:
            with np.load(feature_path, allow_pickle=False) as payload:
                self._feature_cache[feature_path] = {key: payload[key] for key in payload.files}
        return self._feature_cache[feature_path]

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.window_records[index]
        feature_payload = self._load_feature_payload(record.feature_path)
        feature = np.asarray(feature_payload["feature"], dtype=np.float32)
        if feature.ndim != 3 or feature.shape[1:] != (NUM_SENSORS, RAW_IMU_SENSOR_DIM):
            raise ValueError(
                f"feature array must have shape [T,{NUM_SENSORS},{RAW_IMU_SENSOR_DIM}], got {feature.shape}"
            )
        packet_counter = np.asarray(feature_payload["packet_counter"], dtype=np.int64)

        past_start = int(record.start_frame - self.past_frames)
        past_end = int(record.start_frame)
        future_end = int(record.start_frame + self.future_window_frames)
        combined_start = int(past_start)
        combined_end = int(future_end)

        condition = flatten_feature_window(feature[past_start:past_end]).astype(np.float32)
        target_feature = flatten_feature_window(feature[combined_start:combined_end]).astype(np.float32)

        participant = (
            _normalize_text(feature_payload["participant"])
            if "participant" in feature_payload
            else record.participant
        )
        segment_id = (
            _normalize_text(feature_payload["segment_id"])
            if "segment_id" in feature_payload
            else record.segment_id
        )
        return {
            "condition": torch.from_numpy(condition),
            "target_feature": torch.from_numpy(target_feature),
            "meta": {
                "participant": participant,
                "segment_id": segment_id,
                "pose_path": str(record.pose_path),
                "feature_path": str(record.feature_path),
                "start_frame_20hz": int(record.start_frame),
                "past_start_frame_20hz": int(past_start),
                "past_end_frame_20hz": int(past_end - 1),
                "future_end_frame_20hz": int(future_end - 1),
                "packet_start_20hz": int(packet_counter[combined_start]),
                "packet_end_20hz": int(packet_counter[combined_end - 1]),
                "interp_ratio": float(record.interp_ratio),
                "task": record.task,
            },
        }


def compute_condition_normalization_stats(
    dataset: RawImuLatentDiffusionDataset,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((CONDITION_DIM,), dtype=np.float64)
    sums_sq = np.zeros((CONDITION_DIM,), dtype=np.float64)
    count = 0
    for index in range(len(dataset)):
        condition = dataset[index]["condition"].numpy().astype(np.float64)
        sums += condition.sum(axis=0)
        sums_sq += np.square(condition).sum(axis=0)
        count += condition.shape[0]
    if count <= 0:
        raise ValueError("No raw IMU diffusion condition frames available for normalization")
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def build_raw_imu_diffusion_dataloader(
    *,
    window_index_csv: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    past_frames: int = 120,
    future_window_frames: int = 40,
    max_windows: int = 0,
    subset_seed: int = 0,
    data_root: str | Path | None = None,
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
    validate_paths: bool = False,
) -> tuple[RawImuLatentDiffusionDataset, DataLoader]:
    dataset = RawImuLatentDiffusionDataset.from_window_index_csv(
        window_index_csv=window_index_csv,
        past_frames=past_frames,
        future_window_frames=future_window_frames,
        max_windows=max_windows,
        shuffle=shuffle,
        subset_seed=subset_seed,
        data_root=data_root,
        path_prefix_map=path_prefix_map,
        validate_paths=validate_paths,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
    )
    return dataset, dataloader


__all__ = [
    "CONDITION_DIM",
    "FeatureWindowRecord",
    "NUM_SENSORS",
    "RAW_IMU_INPUT_DIM",
    "RAW_IMU_SENSOR_DIM",
    "RawImuLatentDiffusionDataset",
    "build_raw_imu_diffusion_dataloader",
    "compute_condition_normalization_stats",
    "flatten_feature_window",
    "read_feature_window_records_from_csv",
    "resolve_feature_path",
    "select_feature_window_records",
]
