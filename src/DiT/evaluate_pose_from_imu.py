#!/usr/bin/env python3
"""
Evaluate raw-IMU predictions after converting them back to pseudo-pose space.

Input is a raw-IMU evaluation npz, typically:
  feature_predictions.npz with target_feature, pred_feature, meta_json

The conversion path intentionally reuses the same pseudo-pose and translation
helpers used by the dataset export scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
VAE_DIR = CURRENT_DIR.parents[0]
REPO_ROOT = CURRENT_DIR.parents[1]
FROM_JIATONG_DIR = REPO_ROOT / "from_jiatong"
for path in (CURRENT_DIR, VAE_DIR, REPO_ROOT, FROM_JIATONG_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from evaluation_extras import (  # noqa: E402
    DEFAULT_MOTION_ENCODER_CHECKPOINT,
    compute_trimmed_metric_means,
    evaluate_pose_distribution_metrics,
)
from evaluation_metrics import build_metrics_summary, compute_future_metrics, future_slice  # noqa: E402
from imu2translation import estimate_root_translation  # noqa: E402
from raw_imu_ablation import (  # noqa: E402
    DEFAULT_GYR_MOTION_SCALE,
    RAW_IMU_FREEACC_DIM,
    RAW_IMU_GYR_DIM,
    RAW_IMU_INPUT_DIM,
    RAW_IMU_ROT6D_DIM,
    RAW_IMU_SENSOR_COUNT,
    RAW_IMU_SENSOR_DIM,
    predicted_rot6d_to_quaternion,
    unflatten_feature_window,
)
from train_imu_masked_recon import write_json  # noqa: E402
from visualize_dog_skeleton import (  # noqa: E402
    JOINT_ORDER,
    build_heading_rotation_matrices,
    build_relative_positions,
    rotation_matrices_to_rot6d,
)

POSE_POSITION_DIM = 30
ROOT_HEADING_DIM = 6
ROOT_TRANSLATION_DIM = 3
ROOT_JOINT_NAME = "stern"
DEFAULT_RATE_HZ = 20.0
SUMMARY_METADATA_KEYS = (
    "checkpoint_path",
    "checkpoint_epoch",
    "window_index_csv",
    "device_resolved",
    "position_smoothing_kernel",
    "sampling_seed",
)


def normalize_feature_array(feature: np.ndarray, *, key: str) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 3:
        raise ValueError(f"{key} must have shape [N,T,{RAW_IMU_INPUT_DIM}], got {feature.shape}")
    if feature.shape[-1] != RAW_IMU_INPUT_DIM:
        raise ValueError(f"{key} last dim must be {RAW_IMU_INPUT_DIM}, got {feature.shape[-1]}")
    return feature


def normalize_sample_feature_array(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32)
    if feature.ndim != 4:
        raise ValueError(f"pred sample feature must have shape [N,K,T,D] or [K,N,T,D], got {feature.shape}")
    if feature.shape[-1] != RAW_IMU_INPUT_DIM:
        raise ValueError(f"pred sample feature last dim must be {RAW_IMU_INPUT_DIM}, got {feature.shape[-1]}")
    return feature


def resolve_prediction_path(path: Path) -> Path:
    path = Path(path)
    if path.is_dir():
        candidate = path / "feature_predictions.npz"
        if not candidate.exists():
            raise FileNotFoundError(f"{candidate} not found")
        return candidate
    return path


def load_meta(payload: np.lib.npyio.NpzFile, count: int) -> list[dict[str, Any]]:
    if "meta_json" not in payload.files:
        return [{} for _ in range(count)]
    raw = payload["meta_json"]
    text = str(raw.item() if raw.shape == () else raw.tolist())
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("meta_json must decode to a list")
    if len(parsed) != count:
        raise ValueError(f"meta_json length {len(parsed)} does not match prediction count {count}")
    return [dict(item) if isinstance(item, dict) else {"value": item} for item in parsed]


def feature_to_sensor_payload(
    flat_feature_window: np.ndarray,
    *,
    reference_quat: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    feature_window = unflatten_feature_window(flat_feature_window).astype(np.float64)
    quat = predicted_rot6d_to_quaternion(
        feature_window[:, :, :RAW_IMU_ROT6D_DIM],
        reference_quat=reference_quat,
    )
    gyr_start = RAW_IMU_ROT6D_DIM
    gyr_end = gyr_start + RAW_IMU_GYR_DIM
    freeacc_start = gyr_end
    freeacc_end = freeacc_start + RAW_IMU_FREEACC_DIM
    return {
        "quat": quat.astype(np.float64),
        "gyr": feature_window[:, :, gyr_start:gyr_end].astype(np.float64),
        "freeacc": feature_window[:, :, freeacc_start:freeacc_end].astype(np.float64),
        "is_interpolated": (feature_window[:, :, RAW_IMU_SENSOR_DIM - 1] > 0.5),
    }


def build_sensor_data_by_joint(sensor_payload: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    sensor_count = sensor_payload["quat"].shape[1]
    if sensor_count != RAW_IMU_SENSOR_COUNT or sensor_count != len(JOINT_ORDER):
        raise ValueError(f"Expected {len(JOINT_ORDER)} sensors, got {sensor_count}")
    return {
        joint_name: {
            "quat": sensor_payload["quat"][:, joint_index],
            "gyr": sensor_payload["gyr"][:, joint_index],
            "freeacc": sensor_payload["freeacc"][:, joint_index],
            "is_interpolated": sensor_payload["is_interpolated"][:, joint_index],
        }
        for joint_index, joint_name in enumerate(JOINT_ORDER)
    }


def feature_window_to_pose(
    flat_feature_window: np.ndarray,
    *,
    include_root_translation: bool,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
    rate_hz: float,
    reference_quat: np.ndarray | None = None,
) -> np.ndarray:
    sensor_payload = feature_to_sensor_payload(flat_feature_window, reference_quat=reference_quat)
    sensor_data_by_joint = build_sensor_data_by_joint(sensor_payload)
    relative_positions = build_relative_positions(
        sensor_data_by_joint=sensor_data_by_joint,
        neutral_pose_mode=neutral_pose_mode,
        gyr_motion_scale=gyr_motion_scale,
    ).astype(np.float32)
    root_heading_6d = rotation_matrices_to_rot6d(
        build_heading_rotation_matrices(sensor_data_by_joint[ROOT_JOINT_NAME]["quat"])
    ).astype(np.float32)
    parts = [
        relative_positions.reshape(relative_positions.shape[0], POSE_POSITION_DIM),
        root_heading_6d,
    ]
    if include_root_translation:
        root_index = JOINT_ORDER.index(ROOT_JOINT_NAME)
        root_translation = estimate_root_translation(
            freeacc_root=sensor_payload["freeacc"][:, root_index, :].astype(np.float64),
            gyr_root=sensor_payload["gyr"][:, root_index, :].astype(np.float64),
            quat_root=sensor_payload["quat"][:, root_index, :].astype(np.float64),
            is_interpolated_root=sensor_payload["is_interpolated"][:, root_index].astype(bool),
            rate_hz=float(rate_hz),
        ).astype(np.float32)
        parts.append(root_translation)
    return np.concatenate(parts, axis=-1).astype(np.float32)


def choose_future_slice(frame_count: int, *, past_frames: int, future_window_frames: int) -> slice:
    if frame_count == int(future_window_frames):
        return slice(0, frame_count)
    return future_slice(past_frames, future_window_frames, frame_count)


def get_sample_candidates(payload: np.lib.npyio.NpzFile, count: int) -> np.ndarray | None:
    for key in ("pred_feature_samples", "sampled_pred_feature", "sampled_feature", "pred_samples"):
        if key in payload.files:
            samples = normalize_sample_feature_array(payload[key])
            if samples.shape[0] == count:
                return samples
            if samples.shape[1] == count:
                return np.swapaxes(samples, 0, 1)
            raise ValueError(f"{key} must have sample count axis compatible with N={count}, got {samples.shape}")
    return None


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return dict(loaded) if isinstance(loaded, dict) else None


def _resolve_checkpoint_artifacts(
    checkpoint_path: str | Path | None,
    *,
    prediction_npz: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    if checkpoint_path is None:
        return None, None
    checkpoint_candidate = Path(str(checkpoint_path))
    if not checkpoint_candidate.is_absolute():
        repo_relative = (REPO_ROOT / checkpoint_candidate).resolve()
        raw_imu_relative = (CURRENT_DIR / checkpoint_candidate).resolve()
        if repo_relative.exists():
            checkpoint_candidate = repo_relative
        elif raw_imu_relative.exists():
            checkpoint_candidate = raw_imu_relative
        else:
            checkpoint_candidate = raw_imu_relative
    args_path = checkpoint_candidate.parent / "args.json"
    return checkpoint_candidate, _load_json_if_exists(args_path)


def load_summary_metadata(prediction_npz: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    upstream_summary = _load_json_if_exists(prediction_npz.parent / "metrics.json")
    if upstream_summary is not None:
        for key in SUMMARY_METADATA_KEYS:
            if key in upstream_summary:
                summary[key] = upstream_summary[key]
        checkpoint_path = upstream_summary.get("checkpoint_path")
        checkpoint_candidate, checkpoint_args = _resolve_checkpoint_artifacts(
            checkpoint_path,
            prediction_npz=prediction_npz,
        )
        if checkpoint_candidate is not None and "checkpoint_path" not in summary:
            summary["checkpoint_path"] = str(checkpoint_candidate)
        if checkpoint_args is not None:
            smoothing_kernel = checkpoint_args.get("sample_position_smoothing_kernel")
            if smoothing_kernel is not None and "position_smoothing_kernel" not in summary:
                summary["position_smoothing_kernel"] = str(smoothing_kernel)
    return summary


def remove_stale_imu_metric_artifacts(output_dir: Path) -> None:
    for stale_name in (
        "window_metrics.csv",
        "participant_metrics.csv",
        "summary.json",
        "pose_predictions_from_imu.npz",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


def evaluate_pose_from_imu_predictions(
    *,
    prediction_npz: Path,
    output_dir: Path,
    include_root_translation: bool,
    past_frames: int,
    future_window_frames: int,
    sample_count: int,
    neutral_pose_mode: str,
    gyr_motion_scale: float,
    rate_hz: float,
    motion_encoder_checkpoint: Path = DEFAULT_MOTION_ENCODER_CHECKPOINT,
) -> dict[str, Any]:
    prediction_npz = resolve_prediction_path(prediction_npz)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale_imu_metric_artifacts(output_dir)
    summary_metadata = load_summary_metadata(prediction_npz)

    with np.load(prediction_npz, allow_pickle=False) as payload:
        target_feature = normalize_feature_array(payload["target_feature"], key="target_feature")
        pred_feature = normalize_feature_array(payload["pred_feature"], key="pred_feature")
        if target_feature.shape != pred_feature.shape:
            raise ValueError(f"target_feature/pred_feature shape mismatch: {target_feature.shape} vs {pred_feature.shape}")
        meta = load_meta(payload, target_feature.shape[0])
        pred_samples = get_sample_candidates(payload, target_feature.shape[0])

    target_pose_exports: list[np.ndarray] = []
    pred_pose_exports: list[np.ndarray] = []
    metric_rows: list[dict[str, float | None]] = []
    sampled_pose_candidate_exports: list[list[np.ndarray]] = []
    fs = choose_future_slice(
        target_feature.shape[1],
        past_frames=past_frames,
        future_window_frames=future_window_frames,
    )

    for index in tqdm(range(target_feature.shape[0]), desc="convert/eval"):
        target_pose = feature_window_to_pose(
            target_feature[index],
            include_root_translation=include_root_translation,
            neutral_pose_mode=neutral_pose_mode,
            gyr_motion_scale=gyr_motion_scale,
            rate_hz=rate_hz,
        )
        target_sensor_payload = feature_to_sensor_payload(target_feature[index])
        pred_pose = feature_window_to_pose(
            pred_feature[index],
            include_root_translation=include_root_translation,
            neutral_pose_mode=neutral_pose_mode,
            gyr_motion_scale=gyr_motion_scale,
            rate_hz=rate_hz,
            reference_quat=target_sensor_payload["quat"][0],
        )
        target_future_np = target_pose[fs][None].astype(np.float32)
        pred_future_np = pred_pose[fs][None].astype(np.float32)
        target_future = torch.from_numpy(target_future_np)
        pred_future = torch.from_numpy(pred_future_np)
        row = compute_future_metrics(prediction=pred_future, target=target_future)

        candidate_futures: list[np.ndarray] = []
        if pred_samples is not None:
            for sample_index in range(min(int(sample_count), pred_samples.shape[1])):
                sample_pose = feature_window_to_pose(
                    pred_samples[index, sample_index],
                    include_root_translation=include_root_translation,
                    neutral_pose_mode=neutral_pose_mode,
                    gyr_motion_scale=gyr_motion_scale,
                    rate_hz=rate_hz,
                    reference_quat=target_sensor_payload["quat"][0],
                )
                candidate_futures.append(sample_pose[fs].astype(np.float32))
        else:
            candidate_futures.append(pred_future_np[0].astype(np.float32))
        metric_rows.append(row)
        target_pose_exports.append(target_future_np[0].astype(np.float32))
        pred_pose_exports.append(pred_future_np[0].astype(np.float32))
        sampled_pose_candidate_exports.append(candidate_futures)

    target_pose_array = np.stack(target_pose_exports, axis=0).astype(np.float32)
    pred_pose_array = np.stack(pred_pose_exports, axis=0).astype(np.float32)
    per_window_extras, global_distribution_metrics = evaluate_pose_distribution_metrics(
        target_pose=target_pose_array,
        pred_pose=pred_pose_array,
        sampled_pose_candidates=sampled_pose_candidate_exports,
        motion_encoder_checkpoint=motion_encoder_checkpoint,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    for row, extra_row in zip(metric_rows, per_window_extras):
        row.update(extra_row)

    np.savez_compressed(
        output_dir / "pose_predictions.npz",
        target_pose=target_pose_array,
        pred_pose=pred_pose_array,
        pred_pose_samples=np.stack(
            [np.stack(candidates, axis=0).astype(np.float32) for candidates in sampled_pose_candidate_exports],
            axis=0,
        ).astype(np.float32),
        meta_json=np.asarray(json.dumps(meta)),
    )

    summary = build_metrics_summary(
        rows=metric_rows,
        sample_count=sample_count,
        future_frames=int(pred_pose_exports[0].shape[0]) if pred_pose_exports else int(future_window_frames),
        include_root_translation=include_root_translation,
        extra={
            **summary_metadata,
            "window_count": len(metric_rows),
            "motion_distribution_metrics": global_distribution_metrics,
            "translation_metrics_p95_trimmed": compute_trimmed_metric_means(metric_rows),
        },
    )
    write_json(output_dir / "metrics.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw-IMU predictions to pose space and evaluate pose metrics")
    parser.add_argument("--prediction-npz", type=Path, required=True, help="feature_predictions.npz or its parent directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-root-translation", action="store_true")
    parser.add_argument("--past-frames", type=int, default=120)
    parser.add_argument("--future-window-frames", type=int, default=40)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--neutral-pose-mode", type=str, default="sequence-median", choices=("first-frame", "sequence-median"))
    parser.add_argument("--gyr-motion-scale", type=float, default=DEFAULT_GYR_MOTION_SCALE)
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument("--motion-encoder-checkpoint", type=Path, default=DEFAULT_MOTION_ENCODER_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_pose_from_imu_predictions(
        prediction_npz=args.prediction_npz,
        output_dir=args.output_dir,
        include_root_translation=args.include_root_translation,
        past_frames=args.past_frames,
        future_window_frames=args.future_window_frames,
        sample_count=args.sample_count,
        neutral_pose_mode=args.neutral_pose_mode,
        gyr_motion_scale=args.gyr_motion_scale,
        rate_hz=args.rate_hz,
        motion_encoder_checkpoint=args.motion_encoder_checkpoint,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
