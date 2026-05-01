from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
for path in (CURRENT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from fid import compute_frechet_distance, fit_gaussian
from model import MotionConvEncoder, MotionEncoderBiGRUCo, MotionEncoderConfig


def build_encoder(config: MotionEncoderConfig, encoder_type: str) -> torch.nn.Module:
    if encoder_type == "gru":
        return MotionEncoderBiGRUCo(config)
    if encoder_type == "conv":
        return MotionConvEncoder(config)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


def flatten_motion_features(relative_positions: np.ndarray, root_heading_6d: np.ndarray) -> np.ndarray:
    if relative_positions.shape[0] != root_heading_6d.shape[0]:
        raise ValueError("relative_positions and root_heading_6d must share the same frame axis")
    rel_flat = relative_positions.reshape(relative_positions.shape[0], -1)
    return np.concatenate([rel_flat, root_heading_6d], axis=-1).astype(np.float32)


def normalize_per_clip(sequence: np.ndarray) -> np.ndarray:
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((sequence - mean) / std).astype(np.float32)


def build_sequence(
    payload: dict[str, np.ndarray],
    *,
    source: str,
    clip_mode: str,
    include_heading: bool,
    normalize_clip: bool,
) -> np.ndarray:
    cond_rel = np.asarray(payload["cond_relative_positions"], dtype=np.float32)
    cond_heading = np.asarray(payload["cond_root_heading_6d"], dtype=np.float32)

    if source == "pred":
        future_root = np.asarray(payload["pred_root"], dtype=np.float32)
        future_heading = np.asarray(payload["pred_root_heading_6d"], dtype=np.float32)
        future_body = np.asarray(payload["pred_body_rel"], dtype=np.float32)
    elif source == "gt":
        future_root = np.asarray(payload["gt_root"], dtype=np.float32)
        future_heading = np.asarray(payload["gt_root_heading_6d"], dtype=np.float32)
        future_body = np.asarray(payload["gt_body_rel"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported source: {source}")

    future_rel = np.concatenate([future_root, future_body], axis=1).astype(np.float32)
    if clip_mode in {"future_only", "dit_conditional_target"}:
        rel = future_rel
        heading = future_heading
    elif clip_mode == "full_clip":
        rel = np.concatenate([cond_rel, future_rel], axis=0).astype(np.float32)
        heading = np.concatenate([cond_heading, future_heading], axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported clip_mode: {clip_mode}")

    if include_heading:
        sequence = flatten_motion_features(rel, heading)
    else:
        sequence = rel.reshape(rel.shape[0], -1).astype(np.float32)

    if normalize_clip:
        sequence = normalize_per_clip(sequence)
    return sequence


class MotionSequencePairDataset(Dataset):
    def __init__(
        self,
        eval_root: Path,
        *,
        clip_mode: str,
        include_heading: bool,
        normalize_per_clip: bool,
        max_clips: int | None = None,
    ):
        self.eval_root = Path(eval_root)
        self.clip_mode = clip_mode
        self.include_heading = include_heading
        self.normalize_per_clip = normalize_per_clip
        self.samples = sorted(self.eval_root.glob("*/*.npz"))
        if max_clips is not None:
            self.samples = self.samples[:max_clips]
        if not self.samples:
            raise FileNotFoundError(f"No evaluation npz files found under {self.eval_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.samples[index]
        with np.load(path, allow_pickle=False) as payload:
            payload_dict = {key: payload[key] for key in payload.files}

        pred_sequence = build_sequence(
            payload_dict,
            source="pred",
            clip_mode=self.clip_mode,
            include_heading=self.include_heading,
            normalize_clip=self.normalize_per_clip,
        )
        gt_sequence = build_sequence(
            payload_dict,
            source="gt",
            clip_mode=self.clip_mode,
            include_heading=self.include_heading,
            normalize_clip=self.normalize_per_clip,
        )
        return {
            "pred": torch.from_numpy(pred_sequence),
            "gt": torch.from_numpy(gt_sequence),
            "meta": {
                "path": str(path),
            },
        }


def pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    pred_lengths = torch.as_tensor([int(item["pred"].shape[0]) for item in batch], dtype=torch.long)
    gt_lengths = torch.as_tensor([int(item["gt"].shape[0]) for item in batch], dtype=torch.long)
    pred_max_len = int(pred_lengths.max().item())
    gt_max_len = int(gt_lengths.max().item())
    pred_feat_dim = int(batch[0]["pred"].shape[-1])
    gt_feat_dim = int(batch[0]["gt"].shape[-1])

    pred_sequence = torch.zeros(len(batch), pred_max_len, pred_feat_dim, dtype=torch.float32)
    gt_sequence = torch.zeros(len(batch), gt_max_len, gt_feat_dim, dtype=torch.float32)
    for index, item in enumerate(batch):
        pred_sequence[index, : pred_lengths[index]] = item["pred"]
        gt_sequence[index, : gt_lengths[index]] = item["gt"]

    return {
        "pred_sequence": pred_sequence,
        "pred_lengths": pred_lengths,
        "gt_sequence": gt_sequence,
        "gt_lengths": gt_lengths,
        "meta": [item["meta"] for item in batch],
    }


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Compute motion FID from DiT evaluation npz outputs")
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True, help="Directory containing DiT eval output user/*.npz files")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "motion_encoder" / "fid_results")
    parser.add_argument("--clip-mode", choices=["future_only", "dit_conditional_target", "full_clip"], default="future_only")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--feature-space", choices=["embedding", "projection"], default="embedding")
    parser.add_argument("--no-heading", action="store_true", help="Drop root heading from the encoder input features")
    parser.add_argument("--no-normalize-per-clip", action="store_true", help="Disable per-clip feature normalization")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def encode_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    feature_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    pred_features = []
    gt_features = []
    with torch.no_grad():
        for batch in loader:
            pred_sequence = batch["pred_sequence"].to(device=device, dtype=torch.float32)
            pred_lengths = batch["pred_lengths"].to(device=device, dtype=torch.long)
            gt_sequence = batch["gt_sequence"].to(device=device, dtype=torch.float32)
            gt_lengths = batch["gt_lengths"].to(device=device, dtype=torch.long)

            pred_embedding, pred_projection = model(pred_sequence, pred_lengths)
            gt_embedding, gt_projection = model(gt_sequence, gt_lengths)

            pred_value = pred_embedding if feature_space == "embedding" else pred_projection
            gt_value = gt_embedding if feature_space == "embedding" else gt_projection
            pred_features.append(pred_value.cpu().numpy())
            gt_features.append(gt_value.cpu().numpy())

    return np.concatenate(pred_features, axis=0), np.concatenate(gt_features, axis=0)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    dataset = MotionSequencePairDataset(
        args.eval_root,
        clip_mode=args.clip_mode,
        include_heading=not args.no_heading,
        normalize_per_clip=not args.no_normalize_per_clip,
        max_clips=args.max_clips,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=pair_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.encoder_checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint["args"]
    sample = dataset[0]["pred"]
    config = MotionEncoderConfig(
        input_size=int(sample.shape[-1]),
        hidden_size=int(checkpoint_args["hidden_size"]),
        embedding_size=int(checkpoint_args["embedding_size"]),
        projection_size=int(checkpoint_args["projection_size"]),
    )
    model = build_encoder(config, str(checkpoint_args["encoder_type"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pred_features, gt_features = encode_batches(
        model,
        loader,
        device=device,
        feature_space=args.feature_space,
    )
    pred_mean, pred_cov = fit_gaussian(pred_features)
    gt_mean, gt_cov = fit_gaussian(gt_features)
    fid_value = compute_frechet_distance(pred_mean, pred_cov, gt_mean, gt_cov)

    summary = {
        "fid": float(fid_value),
        "num_samples": int(pred_features.shape[0]),
        "feature_dim": int(pred_features.shape[1]),
        "feature_space": args.feature_space,
        "clip_mode": args.clip_mode,
        "include_heading": not args.no_heading,
        "normalize_per_clip": not args.no_normalize_per_clip,
        "encoder_checkpoint": str(args.encoder_checkpoint),
        "eval_root": str(args.eval_root),
    }
    print(json.dumps(summary, indent=2))
    output_path = args.results_dir / "fid_metrics.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
