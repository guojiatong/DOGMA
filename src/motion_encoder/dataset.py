from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
DIT_DIR = REPO_ROOT / "DiT"
for path in (REPO_ROOT, DIT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from DiT.motion_config import CLIP_FRAMES, COND_FRAMES, FUTURE_FRAMES
from DiT.motion_splits import resolve_user_splits


@dataclass(frozen=True)
class MotionClipSample:
    sequence: torch.Tensor
    length: int
    meta: dict[str, Any]


def flatten_motion_features(relative_positions: np.ndarray, root_heading_6d: np.ndarray) -> np.ndarray:
    if relative_positions.shape[0] != root_heading_6d.shape[0]:
        raise ValueError("relative_positions and root_heading_6d must share the same frame axis")
    rel_flat = relative_positions.reshape(relative_positions.shape[0], -1)
    return np.concatenate([rel_flat, root_heading_6d], axis=-1).astype(np.float32)


def _normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((sequence - mean) / std).astype(np.float32)


def augment_motion_sequence(
    sequence: np.ndarray,
    *,
    jitter_std: float = 0.01,
    frame_dropout_prob: float = 0.05,
    scale_jitter: float = 0.05,
    rng: np.random.Generator,
) -> np.ndarray:
    augmented = np.array(sequence, copy=True)
    if jitter_std > 0:
        augmented += rng.normal(0.0, jitter_std, size=augmented.shape).astype(np.float32)
    if frame_dropout_prob > 0 and augmented.shape[0] > 2:
        keep_mask = rng.random(augmented.shape[0]) > frame_dropout_prob
        keep_mask[0] = True
        keep_mask[-1] = True
        if not np.all(keep_mask):
            dropped = augmented[keep_mask]
            source_x = np.linspace(0.0, 1.0, num=dropped.shape[0], dtype=np.float32)
            target_x = np.linspace(0.0, 1.0, num=augmented.shape[0], dtype=np.float32)
            for feature_idx in range(augmented.shape[1]):
                augmented[:, feature_idx] = np.interp(target_x, source_x, dropped[:, feature_idx]).astype(np.float32)
    if scale_jitter > 0:
        scale = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
        augmented *= np.float32(scale)
    return augmented.astype(np.float32)


class MotionClipDataset(Dataset):
    def __init__(
        self,
        pose_root: Path,
        split: str = "train",
        clip_mode: str = "full_clip",
        include_heading: bool = True,
        normalize_per_clip: bool = True,
        max_clips: int | None = None,
        augment: bool = False,
        jitter_std: float = 0.01,
        frame_dropout_prob: float = 0.05,
        scale_jitter: float = 0.05,
        seed: int = 0,
    ):
        self.pose_root = Path(pose_root)
        self.clip_mode = str(clip_mode)
        self.include_heading = include_heading
        self.normalize_per_clip = normalize_per_clip
        self.max_clips = max_clips
        self.augment = augment
        self.jitter_std = float(jitter_std)
        self.frame_dropout_prob = float(frame_dropout_prob)
        self.scale_jitter = float(scale_jitter)
        self.rng = np.random.default_rng(seed)

        train_users, test_users = resolve_user_splits(self.pose_root)
        if split == "train":
            self.users = train_users
        elif split == "test":
            self.users = test_users
        else:
            raise ValueError(f"Unsupported split: {split}")

        self.samples: list[dict[str, Any]] = []
        self._build_index()

    def _resolve_window_params(self) -> tuple[int, int, int]:
        if self.clip_mode == "full_clip":
            return CLIP_FRAMES, CLIP_FRAMES, 0
        if self.clip_mode == "future_only":
            return FUTURE_FRAMES, FUTURE_FRAMES, 0
        if self.clip_mode == "dit_conditional_target":
            return FUTURE_FRAMES, FUTURE_FRAMES, COND_FRAMES
        raise ValueError(f"Unsupported clip_mode: {self.clip_mode}")

    def _build_index(self) -> None:
        window_frames, stride, offset = self._resolve_window_params()
        for user in self.users:
            user_dir = self.pose_root / user
            for pose_path in sorted(user_dir.glob("segment_*.npz")):
                with np.load(pose_path, allow_pickle=False) as payload:
                    rel = payload["relative_positions"].astype(np.float32)
                    heading = payload["root_heading_6d"].astype(np.float32)
                    packets = payload["packet_counter"].astype(np.int64)

                total_frames = rel.shape[0]
                if total_frames < window_frames + offset:
                    continue
                for base_start in range(0, total_frames - (window_frames + offset) + 1, stride):
                    start = base_start + offset
                    end = start + window_frames
                    self.samples.append(
                        {
                            "pose_path": pose_path,
                            "user": user,
                            "segment_id": pose_path.stem.removeprefix("segment_"),
                            "start": start,
                            "end": end,
                            "packet_start": int(packets[start]),
                            "packet_end": int(packets[end - 1]),
                        }
                    )
                    if self.max_clips is not None and len(self.samples) >= self.max_clips:
                        return

    def __len__(self) -> int:
        return len(self.samples)

    def _load_window(self, sample: dict[str, Any]) -> np.ndarray:
        with np.load(sample["pose_path"], allow_pickle=False) as payload:
            rel = payload["relative_positions"][sample["start"] : sample["end"]].astype(np.float32)
            if self.include_heading:
                heading = payload["root_heading_6d"][sample["start"] : sample["end"]].astype(np.float32)
                sequence = flatten_motion_features(rel, heading)
            else:
                sequence = rel.reshape(rel.shape[0], -1).astype(np.float32)
        if self.normalize_per_clip:
            sequence = _normalize_sequence(sequence)
        return sequence

    def __getitem__(self, index: int) -> MotionClipSample:
        sample = self.samples[index]
        sequence = self._load_window(sample)
        if self.augment:
            sequence = augment_motion_sequence(
                sequence,
                jitter_std=self.jitter_std,
                frame_dropout_prob=self.frame_dropout_prob,
                scale_jitter=self.scale_jitter,
                rng=self.rng,
            )
        return MotionClipSample(
            sequence=torch.from_numpy(sequence),
            length=int(sequence.shape[0]),
            meta={
                "user": sample["user"],
                "segment_id": sample["segment_id"],
                "pose_path": str(sample["pose_path"]),
                "start_frame": int(sample["start"]),
                "end_frame": int(sample["end"]),
                "packet_start": int(sample["packet_start"]),
                "packet_end": int(sample["packet_end"]),
            },
        )


def motion_collate(batch: list[MotionClipSample]) -> dict[str, Any]:
    lengths = torch.as_tensor([item.length for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = int(batch[0].sequence.shape[-1])
    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for index, item in enumerate(batch):
        padded[index, : item.length] = item.sequence
    meta = [item.meta for item in batch]
    return {
        "sequence": padded,
        "lengths": lengths,
        "meta": meta,
    }
