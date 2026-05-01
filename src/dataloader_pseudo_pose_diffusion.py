#!/usr/bin/env python3
"""
Self-contained dataloader for DOGMA pseudo-pose latent diffusion.

Canonical task:
- condition: past 6s raw IMU feature at 20Hz -> [120, 130]
- target: 8s combined pseudo-pose window -> [160, 36]

Expected shared data bundle:
- features_20hz/
- pseudo_pose_20hz/
- manifests_8s160f_remote/vae_window_index_{train,val,test}.csv
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
IMU_FEATURE_DIM = 13
CONDITION_DIM = NUM_SENSORS * IMU_FEATURE_DIM
POSE_POSITION_DIM = NUM_SENSORS * 3
ROOT_HEADING_DIM = 6
POSE_DIM = POSE_POSITION_DIM + ROOT_HEADING_DIM


@dataclass(frozen=True)
class PoseWindowRecord:
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


def _build_segment_path(
    *,
    data_root: Path,
    subdir: str,
    participant: str,
    segment_id: str,
) -> Path:
    return Path(data_root) / subdir / participant / f"segment_{segment_id}.npz"


def resolve_manifest_path(
    original_path: str | Path,
    *,
    participant: str,
    segment_id: str,
    kind: str,
    data_root: str | Path | None = None,
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> Path:
    path = Path(original_path)
    prefix_map = _normalize_prefix_map(path_prefix_map)
    if data_root is not None:
        subdir = "features_20hz" if kind == "feature" else "pseudo_pose_20hz"
        return _build_segment_path(
            data_root=Path(data_root),
            subdir=subdir,
            participant=participant,
            segment_id=segment_id,
        )
    return _apply_prefix_map(path, prefix_map)


def read_pose_window_records_from_csv(
    window_index_csv: str | Path,
    *,
    data_root: str | Path | None = None,
    path_prefix_map: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
) -> list[PoseWindowRecord]:
    records: list[PoseWindowRecord] = []
    with Path(window_index_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            participant = _normalize_text(row.get("participant", ""))
            segment_id = _normalize_text(row.get("segment_id", ""))
            start_frame = _parse_int(row.get("start_frame_20hz"))
            end_frame = _parse_int(row.get("end_frame_20hz"))
            feature_path = resolve_manifest_path(
                row["feature_path"],
                participant=participant,
                segment_id=segment_id,
                kind="feature",
                data_root=data_root,
                path_prefix_map=path_prefix_map,
            )
            pose_path = resolve_manifest_path(
                row["pose_path"],
                participant=participant,
                segment_id=segment_id,
                kind="pose",
                data_root=data_root,
                path_prefix_map=path_prefix_map,
            )
            records.append(
                PoseWindowRecord(
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


def select_pose_window_records(
    records: Sequence[PoseWindowRecord],
    *,
    max_windows: int = 0,
    shuffle: bool = False,
    subset_seed: int = 0,
) -> list[PoseWindowRecord]:
    selected = list(records)
    if shuffle and len(selected) > 1:
        rng = np.random.default_rng(subset_seed)
        order = rng.permutation(len(selected)).tolist()
        selected = [selected[int(index)] for index in order]
    if max_windows > 0:
        selected = selected[:max_windows]
    return selected


def flatten_pose_window(relative_positions: np.ndarray, root_heading_6d: np.ndarray) -> np.ndarray:
    relative_positions = np.asarray(relative_positions, dtype=np.float32)
    root_heading_6d = np.asarray(root_heading_6d, dtype=np.float32)
    if relative_positions.ndim != 3 or relative_positions.shape[1:] != (NUM_SENSORS, 3):
        raise ValueError(
            f"relative_positions must have shape [T,{NUM_SENSORS},3], got {relative_positions.shape}"
        )
    if root_heading_6d.ndim != 2 or root_heading_6d.shape[1] != ROOT_HEADING_DIM:
        raise ValueError(f"root_heading_6d must have shape [T,6], got {root_heading_6d.shape}")
    if relative_positions.shape[0] != root_heading_6d.shape[0]:
        raise ValueError("relative_positions and root_heading_6d must have the same frame count")
    frame_count = int(relative_positions.shape[0])
    return np.concatenate(
        [
            relative_positions.reshape(frame_count, -1),
            root_heading_6d,
        ],
        axis=1,
    ).astype(np.float32)


def root_heading_6d_to_angles(root_heading_6d: np.ndarray) -> np.ndarray:
    root_heading_6d = np.asarray(root_heading_6d, dtype=np.float64)
    if root_heading_6d.ndim != 2 or root_heading_6d.shape[1] != ROOT_HEADING_DIM:
        raise ValueError(f"root_heading_6d must have shape [T,6], got {root_heading_6d.shape}")
    if root_heading_6d.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    forward_xy = root_heading_6d[:, :2]
    norms = np.linalg.norm(forward_xy, axis=1, keepdims=True)
    safe_forward_xy = np.divide(
        forward_xy,
        np.maximum(norms, 1e-8),
        out=np.zeros_like(forward_xy),
        where=norms > 1e-8,
    )
    if np.linalg.norm(safe_forward_xy[0]) <= 1e-8:
        safe_forward_xy[0] = np.asarray([1.0, 0.0], dtype=np.float64)
    for frame_index in range(1, safe_forward_xy.shape[0]):
        if np.linalg.norm(safe_forward_xy[frame_index]) <= 1e-8:
            safe_forward_xy[frame_index] = safe_forward_xy[frame_index - 1]
    return np.unwrap(np.arctan2(safe_forward_xy[:, 1], safe_forward_xy[:, 0]))


def angles_to_root_heading_6d(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float64)
    cosine = np.cos(angles)
    sine = np.sin(angles)
    zeros = np.zeros_like(cosine)
    return np.stack([cosine, sine, zeros, -sine, cosine, zeros], axis=1).astype(np.float32)


def rebase_root_heading_6d(
    root_heading_6d: np.ndarray,
    *,
    reference_root_heading_6d: np.ndarray | None = None,
) -> np.ndarray:
    heading_angles = root_heading_6d_to_angles(root_heading_6d)
    if heading_angles.shape[0] == 0:
        return np.zeros((0, ROOT_HEADING_DIM), dtype=np.float32)

    if reference_root_heading_6d is None:
        reference_angle = float(heading_angles[0])
    else:
        reference_angles = root_heading_6d_to_angles(np.asarray(reference_root_heading_6d, dtype=np.float64))
        if reference_angles.shape[0] == 0:
            raise ValueError("reference_root_heading_6d must contain at least one frame")
        reference_angle = float(reference_angles[-1])

    heading_delta = heading_angles - reference_angle
    return angles_to_root_heading_6d(heading_delta)


class PseudoPoseLatentDiffusionDataset(Dataset):
    def __init__(
        self,
        *,
        window_records: Sequence[PoseWindowRecord],
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
            raise ValueError("No pseudo-pose latent diffusion windows available")
        self._feature_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._pose_cache: dict[Path, dict[str, np.ndarray]] = {}
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
    ) -> "PseudoPoseLatentDiffusionDataset":
        records = read_pose_window_records_from_csv(
            window_index_csv,
            data_root=data_root,
            path_prefix_map=path_prefix_map,
        )
        records = [
            record
            for record in records
            if record.start_frame >= int(past_frames) and record.valid_frames >= int(future_window_frames)
        ]
        records = select_pose_window_records(
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
        unique_paths.update(record.pose_path for record in self.window_records)
        missing_paths = [path for path in sorted(unique_paths) if not path.exists()]
        if missing_paths:
            preview = ", ".join(str(path) for path in missing_paths[:5])
            raise FileNotFoundError(f"Resolved dataset paths do not exist. First missing paths: {preview}")

    def _load_npz_payload(
        self,
        path: Path,
        cache: dict[Path, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        if path not in cache:
            with np.load(path, allow_pickle=False) as payload:
                cache[path] = {key: payload[key] for key in payload.files}
        return cache[path]

    def _load_feature_payload(self, feature_path: Path) -> dict[str, np.ndarray]:
        return self._load_npz_payload(feature_path, self._feature_cache)

    def _load_pose_payload(self, pose_path: Path) -> dict[str, np.ndarray]:
        return self._load_npz_payload(pose_path, self._pose_cache)

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.window_records[index]
        feature_payload = self._load_feature_payload(record.feature_path)
        pose_payload = self._load_pose_payload(record.pose_path)

        past_start = int(record.start_frame - self.past_frames)
        past_end = int(record.start_frame)
        future_end = int(record.start_frame + self.future_window_frames)
        combined_start = int(past_start)
        combined_end = int(future_end)

        feature_array = np.asarray(feature_payload["feature"], dtype=np.float32)
        if feature_array.ndim != 3 or feature_array.shape[1:] != (NUM_SENSORS, IMU_FEATURE_DIM):
            raise ValueError(
                f"feature array must have shape [T,{NUM_SENSORS},{IMU_FEATURE_DIM}], got {feature_array.shape}"
            )
        condition = feature_array[past_start:past_end].reshape(self.past_frames, CONDITION_DIM).astype(np.float32)
        relative_positions = np.asarray(
            pose_payload["relative_positions"][combined_start:combined_end],
            dtype=np.float32,
        )
        root_heading_6d = rebase_root_heading_6d(
            np.asarray(
                pose_payload["root_heading_6d"][combined_start:combined_end],
                dtype=np.float32,
            )
        )
        target_pose = flatten_pose_window(relative_positions, root_heading_6d).astype(np.float32)
        packet_counter = np.asarray(pose_payload["packet_counter"], dtype=np.int64)
        future_packets = packet_counter[record.start_frame:future_end]

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
            "target_pose": torch.from_numpy(target_pose),
            "meta": {
                "participant": participant,
                "segment_id": segment_id,
                "pose_path": str(record.pose_path),
                "feature_path": str(record.feature_path),
                "start_frame_20hz": int(record.start_frame),
                "past_start_frame_20hz": int(past_start),
                "past_end_frame_20hz": int(past_end - 1),
                "future_end_frame_20hz": int(future_end - 1),
                "packet_start_20hz": int(future_packets[0]) if future_packets.size else -1,
                "packet_end_20hz": int(future_packets[-1]) if future_packets.size else -1,
                "interp_ratio": float(record.interp_ratio),
                "task": record.task,
            },
        }


def compute_condition_normalization_stats(
    dataset: PseudoPoseLatentDiffusionDataset,
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
        raise ValueError("No pseudo-pose diffusion condition frames available for normalization")
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def build_pseudo_pose_diffusion_dataloader(
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
) -> tuple[PseudoPoseLatentDiffusionDataset, DataLoader]:
    dataset = PseudoPoseLatentDiffusionDataset.from_window_index_csv(
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
    "IMU_FEATURE_DIM",
    "NUM_SENSORS",
    "POSE_DIM",
    "POSE_POSITION_DIM",
    "ROOT_HEADING_DIM",
    "PseudoPoseLatentDiffusionDataset",
    "PoseWindowRecord",
    "angles_to_root_heading_6d",
    "build_pseudo_pose_diffusion_dataloader",
    "compute_condition_normalization_stats",
    "flatten_pose_window",
    "read_pose_window_records_from_csv",
    "rebase_root_heading_6d",
    "resolve_manifest_path",
    "root_heading_6d_to_angles",
    "select_pose_window_records",
]
