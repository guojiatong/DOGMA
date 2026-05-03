from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imu_new2_common import JOINT_ORDER
from train_temporal_vae import POSE_POSITION_DIM, ROOT_HEADING_DIM, ROOT_TRANSLATION_DIM, root_heading_6d_to_angles

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_ENCODER_DIR = REPO_ROOT / "motion_encoder"
for path in (REPO_ROOT, MOTION_ENCODER_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from motion_encoder.model import MotionConvEncoder, MotionEncoderBiGRUCo, MotionEncoderConfig  # noqa: E402
from motion_encoder.fid import compute_frechet_distance  # noqa: E402

DEFAULT_MOTION_ENCODER_CHECKPOINT = (
    REPO_ROOT / "motion_encoder" / "logs_dir" / "w_trans" / "train" / "encoder_20260427_191506" / "best.pt"
)
ROM_JOINT_NAMES = tuple(name for name in JOINT_ORDER if name != "stern")
ROM_AXIS_NAMES = ("yaw", "pitch")


def _build_encoder(config: MotionEncoderConfig, encoder_type: str) -> torch.nn.Module:
    if encoder_type == "gru":
        return MotionEncoderBiGRUCo(config)
    if encoder_type == "conv":
        return MotionConvEncoder(config)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


def _rebase_root_translation(root_translation: np.ndarray) -> np.ndarray:
    root_translation = np.asarray(root_translation, dtype=np.float32)
    if root_translation.ndim != 2 or root_translation.shape[1] != ROOT_TRANSLATION_DIM:
        raise ValueError(f"root_translation must have shape [T,{ROOT_TRANSLATION_DIM}], got {root_translation.shape}")
    if root_translation.shape[0] == 0:
        return np.zeros((0, ROOT_TRANSLATION_DIM), dtype=np.float32)
    return (root_translation - root_translation[0:1]).astype(np.float32)


def _normalize_per_clip(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=0, keepdims=True)
    std = np.maximum(sequence.std(axis=0, keepdims=True), 1e-6)
    return ((sequence - mean) / std).astype(np.float32)


def pose_to_motion_encoder_sequence(
    pose_sequence: np.ndarray,
    *,
    include_root_translation: bool,
    normalize_per_clip: bool = True,
) -> np.ndarray:
    pose_sequence = np.asarray(pose_sequence, dtype=np.float32)
    if pose_sequence.ndim != 2 or pose_sequence.shape[1] < POSE_POSITION_DIM + ROOT_HEADING_DIM:
        raise ValueError(f"pose_sequence must have shape [T,>=36], got {pose_sequence.shape}")
    parts = [
        pose_sequence[:, :POSE_POSITION_DIM],
        pose_sequence[:, POSE_POSITION_DIM : POSE_POSITION_DIM + ROOT_HEADING_DIM],
    ]
    if include_root_translation:
        start = POSE_POSITION_DIM + ROOT_HEADING_DIM
        end = start + ROOT_TRANSLATION_DIM
        if pose_sequence.shape[1] < end:
            raise ValueError("Pose sequence is missing root translation channels required by the motion encoder")
        parts.append(_rebase_root_translation(pose_sequence[:, start:end]))
    sequence = np.concatenate(parts, axis=-1).astype(np.float32)
    if normalize_per_clip:
        sequence = _normalize_per_clip(sequence)
    return sequence


def load_motion_encoder_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = dict(checkpoint["args"])
    include_root_translation = bool(checkpoint_args.get("include_root_translation", False))
    input_size = POSE_POSITION_DIM + ROOT_HEADING_DIM + (ROOT_TRANSLATION_DIM if include_root_translation else 0)
    config = MotionEncoderConfig(
        input_size=input_size,
        hidden_size=int(checkpoint_args["hidden_size"]),
        embedding_size=int(checkpoint_args["embedding_size"]),
        projection_size=int(checkpoint_args["projection_size"]),
    )
    model = _build_encoder(config, str(checkpoint_args["encoder_type"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint_args, include_root_translation


@torch.no_grad()
def encode_pose_sequences(
    model: torch.nn.Module,
    pose_sequences: np.ndarray,
    *,
    device: torch.device,
    include_root_translation: bool,
    batch_size: int = 128,
) -> np.ndarray:
    pose_sequences = np.asarray(pose_sequences, dtype=np.float32)
    if pose_sequences.ndim != 3:
        raise ValueError(f"pose_sequences must have shape [N,T,D], got {pose_sequences.shape}")
    features = np.stack(
        [
            pose_to_motion_encoder_sequence(
                pose_sequences[index],
                include_root_translation=include_root_translation,
                normalize_per_clip=True,
            )
            for index in range(pose_sequences.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    lengths = torch.full((features.shape[0],), features.shape[1], dtype=torch.long, device=device)
    outputs: list[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        end = min(start + batch_size, features.shape[0])
        batch = torch.from_numpy(features[start:end]).to(device=device, dtype=torch.float32)
        embedding, _ = model(batch, lengths[start:end])
        outputs.append(embedding.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, 1), dtype=np.float64)


def _fit_gaussian(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must have shape [N,D], got {features.shape}")
    mean = np.mean(features, axis=0)
    if features.shape[0] <= 1:
        cov = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    else:
        cov = np.cov(features, rowvar=False).astype(np.float64)
        if cov.ndim == 0:
            cov = np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return mean, cov


def compute_fid_from_embeddings(real_embeddings: np.ndarray, pred_embeddings: np.ndarray) -> float:
    mu_real, cov_real = _fit_gaussian(real_embeddings)
    mu_pred, cov_pred = _fit_gaussian(pred_embeddings)
    return float(compute_frechet_distance(mu_real, cov_real, mu_pred, cov_pred))


def compute_embedding_diversity_at_k(embeddings: np.ndarray) -> float:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    sample_count = int(embeddings.shape[0])
    if sample_count <= 1:
        return 0.0
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    pairwise = np.linalg.norm(diffs, axis=-1)
    upper = pairwise[np.triu_indices(sample_count, k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def _future_pose_components(future_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(future_pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < POSE_POSITION_DIM + ROOT_HEADING_DIM:
        raise ValueError(f"future_pose must have shape [T,>=36], got {pose.shape}")
    positions = pose[:, :POSE_POSITION_DIM].reshape(pose.shape[0], len(JOINT_ORDER), 3)
    root_heading_6d = pose[:, POSE_POSITION_DIM : POSE_POSITION_DIM + ROOT_HEADING_DIM]
    return positions, root_heading_6d


def _compute_p95_joint_jerk(positions: np.ndarray) -> float:
    if positions.shape[0] < 3:
        return 0.0
    jerk = positions[2:] - 2.0 * positions[1:-1] + positions[:-2]
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    return float(np.percentile(jerk_norm, 95.0))


def _compute_p95_heading_delta(root_heading_6d: np.ndarray) -> float:
    if root_heading_6d.shape[0] < 2:
        return 0.0
    heading_angles = root_heading_6d_to_angles(root_heading_6d.astype(np.float32))
    heading_delta = np.abs(np.diff(heading_angles))
    return float(np.percentile(heading_delta, 95.0))


def _compute_joint_rom(positions: np.ndarray) -> np.ndarray:
    if positions.shape[0] == 0:
        return np.zeros((len(ROM_JOINT_NAMES), len(ROM_AXIS_NAMES)), dtype=np.float64)
    joint_vectors = positions[:, 1:, :]
    xy_norm = np.linalg.norm(joint_vectors[..., :2], axis=-1)
    yaw = np.unwrap(np.arctan2(joint_vectors[..., 1], joint_vectors[..., 0]), axis=0)
    pitch = np.unwrap(np.arctan2(joint_vectors[..., 2], np.maximum(xy_norm, 1e-8)), axis=0)
    rom_yaw = np.max(yaw, axis=0) - np.min(yaw, axis=0)
    rom_pitch = np.max(pitch, axis=0) - np.min(pitch, axis=0)
    return np.stack([rom_yaw, rom_pitch], axis=-1)


def compute_window_plausibility_stats(future_pose: np.ndarray) -> dict[str, Any]:
    positions, root_heading_6d = _future_pose_components(future_pose)
    return {
        "p95_joint_jerk": _compute_p95_joint_jerk(positions),
        "p95_heading_delta": _compute_p95_heading_delta(root_heading_6d),
        "rom_joint_axis": _compute_joint_rom(positions),
    }


def _compute_scalar_band(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def compute_real_bands(real_stats: list[dict[str, Any]]) -> dict[str, Any]:
    jerk = np.asarray([row["p95_joint_jerk"] for row in real_stats], dtype=np.float64)
    heading = np.asarray([row["p95_heading_delta"] for row in real_stats], dtype=np.float64)
    rom = np.stack([row["rom_joint_axis"] for row in real_stats], axis=0).astype(np.float64)
    return {
        "jerk_band": _compute_scalar_band(jerk),
        "heading_delta_band": _compute_scalar_band(heading),
        "rom_low": np.quantile(rom, 0.05, axis=0),
        "rom_high": np.quantile(rom, 0.95, axis=0),
    }


def _outside_band(value: float, band: tuple[float, float]) -> int:
    return int(value < band[0] or value > band[1])


def compute_rom_band_violation_fraction(
    rom_joint_axis: np.ndarray,
    *,
    rom_low: np.ndarray,
    rom_high: np.ndarray,
) -> float:
    outside = (rom_joint_axis < rom_low) | (rom_joint_axis > rom_high)
    return float(np.mean(outside.astype(np.float64)))


def compute_trimmed_metric_means(
    rows: list[dict[str, float | None]],
    *,
    keys: tuple[str, ...] = ("ade", "fde"),
    upper_quantile: float = 0.95,
) -> dict[str, float | None]:
    trimmed: dict[str, float | None] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if row.get(key) is not None], dtype=np.float64)
        if values.size == 0:
            trimmed[key] = None
            continue
        if values.size == 1:
            trimmed[key] = float(values[0])
            continue
        cutoff = float(np.quantile(values, upper_quantile))
        kept = values[values <= cutoff]
        trimmed[key] = float(np.mean(kept)) if kept.size else float(np.mean(values))
    return trimmed


def evaluate_pose_distribution_metrics(
    *,
    target_pose: np.ndarray,
    pred_pose: np.ndarray,
    sampled_pose_candidates: list[list[np.ndarray]],
    motion_encoder_checkpoint: Path,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    if target_pose.shape != pred_pose.shape:
        raise ValueError(f"target_pose/pred_pose shape mismatch: {target_pose.shape} vs {pred_pose.shape}")
    if len(sampled_pose_candidates) != target_pose.shape[0]:
        raise ValueError(
            "sampled_pose_candidates must have one candidate list per window; "
            f"got {len(sampled_pose_candidates)} for {target_pose.shape[0]} windows"
        )

    encoder, encoder_args, include_root_translation = load_motion_encoder_checkpoint(
        motion_encoder_checkpoint,
        device=device,
    )
    real_embeddings = encode_pose_sequences(
        encoder,
        target_pose,
        device=device,
        include_root_translation=include_root_translation,
    )
    pred_embeddings = encode_pose_sequences(
        encoder,
        pred_pose,
        device=device,
        include_root_translation=include_root_translation,
    )
    fid = compute_fid_from_embeddings(real_embeddings, pred_embeddings)

    real_stats = [compute_window_plausibility_stats(target_pose[index]) for index in range(target_pose.shape[0])]
    real_bands = compute_real_bands(real_stats)

    per_window_rows: list[dict[str, float]] = []
    diversity_values: list[float] = []
    jerk_violations: list[float] = []
    rom_violations: list[float] = []
    heading_violations: list[float] = []
    for index in range(pred_pose.shape[0]):
        pred_stats = compute_window_plausibility_stats(pred_pose[index])
        jerk_violation = _outside_band(pred_stats["p95_joint_jerk"], real_bands["jerk_band"])
        heading_violation = _outside_band(pred_stats["p95_heading_delta"], real_bands["heading_delta_band"])
        rom_violation = compute_rom_band_violation_fraction(
            pred_stats["rom_joint_axis"],
            rom_low=real_bands["rom_low"],
            rom_high=real_bands["rom_high"],
        )
        candidates = sampled_pose_candidates[index]
        if not candidates:
            candidates = [pred_pose[index]]
        candidate_embeddings = encode_pose_sequences(
            encoder,
            np.stack(candidates, axis=0).astype(np.float32),
            device=device,
            include_root_translation=include_root_translation,
        )
        diversity_at_10 = compute_embedding_diversity_at_k(candidate_embeddings[:10])
        diversity_values.append(diversity_at_10)
        jerk_violations.append(float(jerk_violation))
        rom_violations.append(float(rom_violation))
        heading_violations.append(float(heading_violation))
        per_window_rows.append(
            {
                "diversity_at_10": float(diversity_at_10),
                "p95_joint_jerk": float(pred_stats["p95_joint_jerk"]),
                "p95_heading_delta": float(pred_stats["p95_heading_delta"]),
                "jerk_band_violation": int(jerk_violation),
                "heading_delta_band_violation": int(heading_violation),
                "rom_band_violation_fraction": float(rom_violation),
            }
        )

    global_metrics = {
        "fid_motion_encoder": float(fid),
        "diversity_at_10": float(np.mean(diversity_values)) if diversity_values else 0.0,
        "jerk_band_violation_rate": float(np.mean(jerk_violations)) if jerk_violations else 0.0,
        "heading_delta_band_violation_rate": float(np.mean(heading_violations)) if heading_violations else 0.0,
        "rom_band_violation_rate": float(np.mean(rom_violations)) if rom_violations else 0.0,
        "motion_encoder_checkpoint": str(motion_encoder_checkpoint),
        "motion_encoder_run_name": str(encoder_args.get("results_dir", "")),
        "real_bands": {
            "jerk_band": [float(real_bands["jerk_band"][0]), float(real_bands["jerk_band"][1])],
            "heading_delta_band": [
                float(real_bands["heading_delta_band"][0]),
                float(real_bands["heading_delta_band"][1]),
            ],
            "rom_joint_axis_low": real_bands["rom_low"].tolist(),
            "rom_joint_axis_high": real_bands["rom_high"].tolist(),
            "rom_joint_names": list(ROM_JOINT_NAMES),
            "rom_axis_names": list(ROM_AXIS_NAMES),
        },
    }
    return per_window_rows, global_metrics
