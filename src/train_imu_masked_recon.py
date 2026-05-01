#!/usr/bin/env python3
"""
Minimal masked reconstruction dataset and loss for stage 1 TDD.
"""

from __future__ import annotations

import argparse
import json
import random
import csv
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.imu_baselines import MaskedImuReconstructionTransformer


@dataclass(frozen=True)
class WindowRecord:
    feature_path: Path
    start_frame: int
    valid_frames: int


@dataclass(frozen=True)
class MaskedReconTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_frames: int = 240
    train_stride_frames: int = 20
    val_stride_frames: int = 120
    mask_ratio: float = 0.30
    mask_seed: int = 0
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 0
    scheduler_type: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.5
    scheduler_t_max: int = 0
    max_train_windows: int = 0
    max_val_windows: int = 0
    overfit_windows: int = 0
    sample_export_count: int = 0
    sample_seed: int = 0
    rot_loss_weight: float = 1.0
    gyr_loss_weight: float = 1.0
    freeacc_loss_weight: float = 1.0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def read_window_records_from_csv(window_index_csv: Path) -> list[WindowRecord]:
    records: list[WindowRecord] = []
    with Path(window_index_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_frame = int(row["start_frame_20hz"])
            end_frame = int(row["end_frame_20hz"])
            records.append(
                WindowRecord(
                    feature_path=Path(row["feature_path"]),
                    start_frame=start_frame,
                    valid_frames=end_frame - start_frame + 1,
                )
            )
    if not records:
        raise ValueError(f"No windows found in {window_index_csv}")
    return records


class MaskedImuWindowDataset(Dataset):
    def __init__(
        self,
        feature_paths: list[Path] | None = None,
        window_frames: int = 240,
        stride_frames: int = 20,
        mask_ratio: float = 0.30,
        mask_seed: int = 0,
        pad_short_windows: bool = False,
        window_records: list[WindowRecord] | None = None,
    ) -> None:
        self.feature_paths = [Path(path) for path in (feature_paths or [])]
        self.window_frames = int(window_frames)
        self.stride_frames = int(stride_frames)
        self.mask_ratio = float(mask_ratio)
        self.mask_seed = int(mask_seed)
        self.pad_short_windows = bool(pad_short_windows)
        self.window_records: list[WindowRecord] = list(window_records or [])
        self._payload_cache: dict[Path, dict[str, np.ndarray]] = {}

        if not self.window_records:
            for feature_path in self.feature_paths:
                payload = self._load_payload(feature_path)
                total_frames = int(payload["feature"].shape[0])
                if total_frames >= self.window_frames:
                    max_start = total_frames - self.window_frames
                    starts = list(range(0, max_start + 1, self.stride_frames))
                    if not starts or starts[-1] != max_start:
                        starts.append(max_start)
                    for start_frame in starts:
                        self.window_records.append(
                            WindowRecord(feature_path=feature_path, start_frame=int(start_frame), valid_frames=self.window_frames)
                        )
                elif self.pad_short_windows:
                    self.window_records.append(
                        WindowRecord(feature_path=feature_path, start_frame=0, valid_frames=total_frames)
                    )

        if not self.window_records:
            raise ValueError("No masked reconstruction windows available")

    @classmethod
    def from_window_index_csv(
        cls,
        *,
        window_index_csv: Path,
        mask_ratio: float = 0.30,
        mask_seed: int = 0,
        max_windows: int = 0,
    ) -> "MaskedImuWindowDataset":
        records = read_window_records_from_csv(window_index_csv)
        if max_windows > 0:
            records = records[:max_windows]
        return dataset_from_window_records(records=records, mask_ratio=mask_ratio, mask_seed=mask_seed)

    def _load_payload(self, feature_path: Path) -> dict[str, np.ndarray]:
        if feature_path not in self._payload_cache:
            with np.load(feature_path, allow_pickle=False) as payload:
                self._payload_cache[feature_path] = {key: payload[key] for key in payload.files}
        return self._payload_cache[feature_path]

    def __len__(self) -> int:
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.window_records[index]
        payload = self._load_payload(record.feature_path)
        feature = payload["feature"]
        interpolated = payload["is_interpolated"].astype(bool)
        packet_counter = payload["packet_counter"]

        end_frame = record.start_frame + min(self.window_frames, record.valid_frames)
        window_feature = feature[record.start_frame:end_frame]
        window_interpolated = interpolated[record.start_frame:end_frame]
        window_packets = packet_counter[record.start_frame:end_frame]

        input_window = np.zeros((self.window_frames, feature.shape[1], feature.shape[2]), dtype=np.float32)
        target_window = np.zeros((self.window_frames, feature.shape[1], 12), dtype=np.float32)
        target_weight = np.zeros((self.window_frames, feature.shape[1]), dtype=np.float32)
        valid_frames = window_feature.shape[0]
        input_window[:valid_frames] = window_feature
        target_window[:valid_frames] = window_feature[:, :, :12]
        target_weight[:valid_frames] = (~window_interpolated).astype(np.float32)

        rng = np.random.default_rng(self.mask_seed + index)
        mask = rng.random((self.window_frames, feature.shape[1])) < self.mask_ratio
        if valid_frames > 0 and not np.any(mask[:valid_frames]):
            mask[0, 0] = True
        mask[valid_frames:] = False

        participant = str(payload["participant"]) if "participant" in payload else record.feature_path.parent.name
        segment_id = str(payload["segment_id"]) if "segment_id" in payload else record.feature_path.stem.replace("segment_", "")
        return {
            "input": torch.from_numpy(input_window),
            "target": torch.from_numpy(target_window),
            "mask": torch.from_numpy(mask.astype(bool)),
            "target_weight": torch.from_numpy(target_weight),
            "meta": {
                "participant": participant,
                "segment_id": segment_id,
                "start_frame_20hz": int(record.start_frame),
                "valid_frames": int(valid_frames),
                "packet_start_20hz": int(window_packets[0]) if valid_frames > 0 else -1,
                "packet_end_20hz": int(window_packets[-1]) if valid_frames > 0 else -1,
            },
        }


def dataset_from_window_records(
    *,
    records: list[WindowRecord],
    mask_ratio: float,
    mask_seed: int,
) -> MaskedImuWindowDataset:
    if not records:
        raise ValueError("No masked reconstruction windows available")
    return MaskedImuWindowDataset(
        feature_paths=[],
        window_frames=records[0].valid_frames,
        stride_frames=records[0].valid_frames,
        mask_ratio=mask_ratio,
        mask_seed=mask_seed,
        pad_short_windows=False,
        window_records=records,
    )


def limit_dataset_windows(
    dataset: MaskedImuWindowDataset,
    *,
    max_windows: int,
    mask_seed: int,
) -> MaskedImuWindowDataset:
    if max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    return dataset_from_window_records(
        records=dataset.window_records[:max_windows],
        mask_ratio=dataset.mask_ratio,
        mask_seed=mask_seed,
    )


def compute_masked_recon_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_weight: torch.Tensor,
    rot_loss_weight: float = 1.0,
    gyr_loss_weight: float = 1.0,
    freeacc_loss_weight: float = 1.0,
) -> torch.Tensor:
    valid_mask = mask & (target_weight > 0.0)
    if not torch.any(valid_mask):
        return prediction.new_zeros(())
    channel_weights = prediction.new_tensor(
        [rot_loss_weight] * 6 + [gyr_loss_weight] * 3 + [freeacc_loss_weight] * 3
    )
    if torch.sum(channel_weights) <= 0.0:
        return prediction.new_zeros(())
    squared_error = (prediction - target) ** 2
    valid_error = squared_error[valid_mask]
    weighted_error = valid_error * channel_weights
    return weighted_error.sum() / (valid_mask.sum().to(prediction.dtype) * channel_weights.sum())


def rot6d_to_rotation_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    first = rot6d[..., 0:3]
    second = rot6d[..., 3:6]
    basis_x = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    second = second - (basis_x * second).sum(dim=-1, keepdim=True) * basis_x
    basis_y = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    basis_z = torch.cross(basis_x, basis_y, dim=-1)
    return torch.stack([basis_x, basis_y, basis_z], dim=-1)


def _compute_batch_metric_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    target_weight: torch.Tensor,
    rot_loss_weight: float = 1.0,
    gyr_loss_weight: float = 1.0,
    freeacc_loss_weight: float = 1.0,
) -> dict[str, float]:
    valid_mask = (mask & (target_weight > 0.0)).reshape(-1)
    prediction_flat = prediction.reshape(-1, prediction.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])

    if not torch.any(valid_mask):
        return {
            "masked_loss_sum": 0.0,
            "masked_loss_count": 0.0,
            "masked_mse_sum": 0.0,
            "masked_mse_count": 0.0,
            "rot_mse_sum": 0.0,
            "rot_mse_count": 0.0,
            "rot_geodesic_sum": 0.0,
            "rot_geodesic_count": 0.0,
            "gyr_sse_sum": 0.0,
            "gyr_sse_count": 0.0,
            "freeacc_sse_sum": 0.0,
            "freeacc_sse_count": 0.0,
        }

    valid_prediction = prediction_flat[valid_mask]
    valid_target = target_flat[valid_mask]
    squared_error = (valid_prediction - valid_target) ** 2
    channel_weights = prediction.new_tensor(
        [rot_loss_weight] * 6 + [gyr_loss_weight] * 3 + [freeacc_loss_weight] * 3
    )
    weighted_error = squared_error * channel_weights

    pred_rot = rot6d_to_rotation_matrix(valid_prediction[:, :6])
    target_rot = rot6d_to_rotation_matrix(valid_target[:, :6])
    relative_rotation = torch.matmul(pred_rot.transpose(-1, -2), target_rot)
    trace = relative_rotation[..., 0, 0] + relative_rotation[..., 1, 1] + relative_rotation[..., 2, 2]
    cosine = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    rot_geodesic = torch.acos(cosine)

    return {
        "masked_loss_sum": float(weighted_error.sum().item()),
        "masked_loss_count": float(valid_prediction.shape[0] * channel_weights.sum().item()),
        "masked_mse_sum": float(squared_error.sum().item()),
        "masked_mse_count": float(squared_error.numel()),
        "rot_mse_sum": float(squared_error[:, :6].sum().item()),
        "rot_mse_count": float(squared_error[:, :6].numel()),
        "rot_geodesic_sum": float(rot_geodesic.sum().item()),
        "rot_geodesic_count": float(rot_geodesic.numel()),
        "gyr_sse_sum": float(squared_error[:, 6:9].sum().item()),
        "gyr_sse_count": float(squared_error[:, 6:9].numel()),
        "freeacc_sse_sum": float(squared_error[:, 9:12].sum().item()),
        "freeacc_sse_count": float(squared_error[:, 9:12].numel()),
    }


def finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float]:
    masked_loss = (
        metric_sums["masked_loss_sum"] / metric_sums["masked_loss_count"]
        if metric_sums["masked_loss_count"] > 0.0
        else 0.0
    )
    masked_mse = (
        metric_sums["masked_mse_sum"] / metric_sums["masked_mse_count"]
        if metric_sums["masked_mse_count"] > 0.0
        else 0.0
    )
    rot_mse = (
        metric_sums["rot_mse_sum"] / metric_sums["rot_mse_count"]
        if metric_sums["rot_mse_count"] > 0.0
        else 0.0
    )
    rot_geodesic = (
        metric_sums["rot_geodesic_sum"] / metric_sums["rot_geodesic_count"]
        if metric_sums["rot_geodesic_count"] > 0.0
        else 0.0
    )
    gyr_mse = (
        metric_sums["gyr_sse_sum"] / metric_sums["gyr_sse_count"]
        if metric_sums["gyr_sse_count"] > 0.0
        else 0.0
    )
    gyr_rmse = (
        float(np.sqrt(metric_sums["gyr_sse_sum"] / metric_sums["gyr_sse_count"]))
        if metric_sums["gyr_sse_count"] > 0.0
        else 0.0
    )
    freeacc_mse = (
        metric_sums["freeacc_sse_sum"] / metric_sums["freeacc_sse_count"]
        if metric_sums["freeacc_sse_count"] > 0.0
        else 0.0
    )
    freeacc_rmse = (
        float(np.sqrt(metric_sums["freeacc_sse_sum"] / metric_sums["freeacc_sse_count"]))
        if metric_sums["freeacc_sse_count"] > 0.0
        else 0.0
    )
    return {
        "masked_loss": float(masked_loss),
        "masked_mse": float(masked_mse),
        "rot_mse": float(rot_mse),
        "masked_rot_geodesic_error": float(rot_geodesic),
        "gyr_mse": float(gyr_mse),
        "gyr_rmse": float(gyr_rmse),
        "freeacc_mse": float(freeacc_mse),
        "freeacc_rmse": float(freeacc_rmse),
    }


def merge_metric_sums(total: dict[str, float], update: dict[str, float]) -> dict[str, float]:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)
    return total


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: MaskedReconTrainingConfig,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["input"], batch["mask"])
        loss = compute_masked_recon_loss(
            prediction=prediction,
            target=batch["target"],
            mask=batch["mask"],
            target_weight=batch["target_weight"],
            rot_loss_weight=config.rot_loss_weight,
            gyr_loss_weight=config.gyr_loss_weight,
            freeacc_loss_weight=config.freeacc_loss_weight,
        )
        loss.backward()
        optimizer.step()
        merge_metric_sums(
            metric_sums,
            _compute_batch_metric_sums(
                prediction=prediction.detach(),
                target=batch["target"],
                mask=batch["mask"],
                target_weight=batch["target_weight"],
                rot_loss_weight=config.rot_loss_weight,
                gyr_loss_weight=config.gyr_loss_weight,
                freeacc_loss_weight=config.freeacc_loss_weight,
            ),
        )
    return finalize_metric_sums(metric_sums)


@torch.no_grad()
def evaluate_masked_recon(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: MaskedReconTrainingConfig,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        prediction = model(batch["input"], batch["mask"])
        merge_metric_sums(
            metric_sums,
            _compute_batch_metric_sums(
                prediction=prediction,
                target=batch["target"],
                mask=batch["mask"],
                target_weight=batch["target_weight"],
                rot_loss_weight=config.rot_loss_weight,
                gyr_loss_weight=config.gyr_loss_weight,
                freeacc_loss_weight=config.freeacc_loss_weight,
            ),
        )
    return finalize_metric_sums(metric_sums)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: MaskedReconTrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.scheduler_type == "none":
        return None
    if config.scheduler_type == "step":
        if config.scheduler_step_size <= 0:
            raise ValueError("scheduler_step_size must be positive for step scheduler")
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )
    if config.scheduler_type == "cosine":
        t_max = config.scheduler_t_max if config.scheduler_t_max > 0 else config.epochs
        if t_max <= 0:
            raise ValueError("scheduler_t_max or epochs must be positive for cosine scheduler")
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    raise ValueError(f"Unsupported scheduler_type: {config.scheduler_type}")


@torch.no_grad()
def export_masked_recon_samples(
    *,
    model: torch.nn.Module,
    dataset: MaskedImuWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
) -> None:
    if sample_count <= 0:
        return
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(sample_count), len(dataset))
    rng = np.random.default_rng(sample_seed + epoch)
    if count == len(dataset):
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.sort(rng.choice(len(dataset), size=count, replace=False))

    samples = [dataset[int(index)] for index in indices]
    input_tensor = torch.stack([sample["input"] for sample in samples], dim=0).to(device)
    target_tensor = torch.stack([sample["target"] for sample in samples], dim=0).to(device)
    mask_tensor = torch.stack([sample["mask"] for sample in samples], dim=0).to(device)
    target_weight_tensor = torch.stack([sample["target_weight"] for sample in samples], dim=0).to(device)
    prediction = model(input_tensor, mask_tensor)
    meta_json = json.dumps([sample["meta"] for sample in samples])

    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        input=input_tensor.detach().cpu().numpy().astype(np.float32),
        target=target_tensor.detach().cpu().numpy().astype(np.float32),
        prediction=prediction.detach().cpu().numpy().astype(np.float32),
        mask=mask_tensor.detach().cpu().numpy().astype(bool),
        target_weight=target_weight_tensor.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: MaskedReconTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "config": asdict(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def run_masked_recon_training(
    train_feature_paths: list[Path] | None,
    val_feature_paths: list[Path] | None,
    output_dir: Path,
    config: MaskedReconTrainingConfig,
    train_window_index_csv: Path | None = None,
    val_window_index_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    if train_window_index_csv is not None:
        train_dataset = MaskedImuWindowDataset.from_window_index_csv(
            window_index_csv=train_window_index_csv,
            mask_ratio=config.mask_ratio,
            mask_seed=config.mask_seed,
        )
    else:
        train_dataset = MaskedImuWindowDataset(
            feature_paths=[Path(path) for path in (train_feature_paths or [])],
            window_frames=config.window_frames,
            stride_frames=config.train_stride_frames,
            mask_ratio=config.mask_ratio,
            mask_seed=config.mask_seed,
            pad_short_windows=True,
        )
    if val_window_index_csv is not None:
        val_dataset = MaskedImuWindowDataset.from_window_index_csv(
            window_index_csv=val_window_index_csv,
            mask_ratio=config.mask_ratio,
            mask_seed=config.mask_seed + 10_000,
        )
    else:
        val_dataset = MaskedImuWindowDataset(
            feature_paths=[Path(path) for path in (val_feature_paths or [])],
            window_frames=config.window_frames,
            stride_frames=config.val_stride_frames,
            mask_ratio=config.mask_ratio,
            mask_seed=config.mask_seed + 10_000,
            pad_short_windows=True,
        )

    if config.overfit_windows > 0:
        overfit_records = train_dataset.window_records[: config.overfit_windows]
        train_dataset = dataset_from_window_records(
            records=overfit_records,
            mask_ratio=config.mask_ratio,
            mask_seed=config.mask_seed,
        )
        val_dataset = dataset_from_window_records(
            records=overfit_records,
            mask_ratio=config.mask_ratio,
            mask_seed=config.mask_seed + 10_000,
        )
    else:
        train_dataset = limit_dataset_windows(
            train_dataset,
            max_windows=config.max_train_windows,
            mask_seed=config.mask_seed,
        )
        val_dataset = limit_dataset_windows(
            val_dataset,
            max_windows=config.max_val_windows,
            mask_seed=config.mask_seed + 10_000,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = MaskedImuReconstructionTransformer(
        input_dim=13,
        target_dim=12,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
        max_frames=config.window_frames,
        num_sensors=10,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer=optimizer, config=config)

    args_payload = asdict(config) | {
        "train_feature_paths": [str(path) for path in (train_feature_paths or [])],
        "val_feature_paths": [str(path) for path in (val_feature_paths or [])],
        "train_window_index_csv": None if train_window_index_csv is None else str(train_window_index_csv),
        "val_window_index_csv": None if val_window_index_csv is None else str(val_window_index_csv),
        "device_resolved": str(device),
    }
    write_json(output_dir / "args.json", args_payload)

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_metric = float("inf")
    best_epoch = 0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
        )
        val_metrics = evaluate_masked_recon(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config,
        )

        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_masked_loss": train_metrics["masked_loss"],
            "train_masked_mse": train_metrics["masked_mse"],
            "train_rot_mse": train_metrics["rot_mse"],
            "train_masked_rot_geodesic_error": train_metrics["masked_rot_geodesic_error"],
            "train_gyr_mse": train_metrics["gyr_mse"],
            "train_gyr_rmse": train_metrics["gyr_rmse"],
            "train_freeacc_mse": train_metrics["freeacc_mse"],
            "train_freeacc_rmse": train_metrics["freeacc_rmse"],
            "val_masked_loss": val_metrics["masked_loss"],
            "val_masked_mse": val_metrics["masked_mse"],
            "val_rot_mse": val_metrics["rot_mse"],
            "val_masked_rot_geodesic_error": val_metrics["masked_rot_geodesic_error"],
            "val_gyr_mse": val_metrics["gyr_mse"],
            "val_gyr_rmse": val_metrics["gyr_rmse"],
            "val_freeacc_mse": val_metrics["freeacc_mse"],
            "val_freeacc_rmse": val_metrics["freeacc_rmse"],
        }
        append_jsonl(metrics_path, record)
        export_masked_recon_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_predictions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
        )
        if scheduler is not None:
            scheduler.step()
        save_checkpoint(
            output_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        if record["val_masked_rot_geodesic_error"] <= best_metric:
            best_metric = record["val_masked_rot_geodesic_error"]
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

    return {
        "best_epoch": int(best_epoch),
        "best_val_masked_rot_geodesic_error": float(best_metric),
        "output_dir": str(output_dir),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masked IMU reconstruction utilities")
    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--train-feature-path", type=Path, action="append", default=[])
    parser.add_argument("--val-feature-path", type=Path, action="append", default=[])
    parser.add_argument("--train-window-index-csv", type=Path, default=None)
    parser.add_argument("--val-window-index-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--val-stride-frames", type=int, default=120)
    parser.add_argument("--mask-ratio", type=float, default=0.30)
    parser.add_argument("--mask-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler-type", type=str, default="none", choices=("none", "step", "cosine"))
    parser.add_argument("--scheduler-step-size", type=int, default=1)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--overfit-windows", type=int, default=0)
    parser.add_argument("--sample-export-count", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--rot-loss-weight", type=float, default=1.0)
    parser.add_argument("--gyr-loss-weight", type=float, default=1.0)
    parser.add_argument("--freeacc-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.summary_json is not None:
        if args.feature_path is None:
            raise ValueError("--feature-path is required when --summary-json is used")
        dataset = MaskedImuWindowDataset(
            feature_paths=[args.feature_path],
            window_frames=args.window_frames,
            stride_frames=args.train_stride_frames,
            mask_ratio=args.mask_ratio,
            mask_seed=args.mask_seed,
            pad_short_windows=True,
        )
        first_sample = dataset[0]
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "num_windows": len(dataset),
                    "input_shape": list(first_sample["input"].shape),
                    "target_shape": list(first_sample["target"].shape),
                    "mask_count": int(first_sample["mask"].sum().item()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(args.summary_json)
        return

    train_source_present = bool(args.train_feature_path) or args.train_window_index_csv is not None
    val_source_present = bool(args.val_feature_path) or args.val_window_index_csv is not None
    if not train_source_present or not val_source_present or args.output_dir is None:
        raise ValueError(
            "Training mode requires train/val data via --train-feature-path or --train-window-index-csv, "
            "--val-feature-path or --val-window-index-csv, and --output-dir"
        )

    summary = run_masked_recon_training(
        train_feature_paths=args.train_feature_path,
        val_feature_paths=args.val_feature_path,
        output_dir=args.output_dir,
        config=MaskedReconTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            window_frames=args.window_frames,
            train_stride_frames=args.train_stride_frames,
            val_stride_frames=args.val_stride_frames,
            mask_ratio=args.mask_ratio,
            mask_seed=args.mask_seed,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_workers=args.num_workers,
            device=args.device,
            seed=args.seed,
            scheduler_type=args.scheduler_type,
            scheduler_step_size=args.scheduler_step_size,
            scheduler_gamma=args.scheduler_gamma,
            scheduler_t_max=args.scheduler_t_max,
            max_train_windows=args.max_train_windows,
            max_val_windows=args.max_val_windows,
            overfit_windows=args.overfit_windows,
            sample_export_count=args.sample_export_count,
            sample_seed=args.sample_seed,
            rot_loss_weight=args.rot_loss_weight,
            gyr_loss_weight=args.gyr_loss_weight,
            freeacc_loss_weight=args.freeacc_loss_weight,
        ),
        train_window_index_csv=args.train_window_index_csv,
        val_window_index_csv=args.val_window_index_csv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
