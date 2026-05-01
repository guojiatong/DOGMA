#!/usr/bin/env python3
"""
20Hz IMU-only future prediction baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.imu_baselines import ImuFuturePredictionTransformer
from train_imu_masked_recon import (
    _compute_batch_metric_sums,
    append_jsonl,
    build_scheduler,
    compute_masked_recon_loss,
    finalize_metric_sums,
    merge_metric_sums,
    resolve_device,
    set_global_seed,
    write_json,
)


@dataclass(frozen=True)
class FutureWindowRecord:
    feature_path: Path
    start_frame: int
    valid_frames: int


@dataclass(frozen=True)
class FuturePredictionTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    window_frames: int = 240
    past_frames: int = 120
    future_frames: int = 120
    train_stride_frames: int = 20
    val_stride_frames: int = 120
    d_model: int = 256
    nhead: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
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
    gyr_loss_weight: float = 0.2
    freeacc_loss_weight: float = 0.02


def read_future_window_records_from_csv(window_index_csv: Path) -> list[FutureWindowRecord]:
    records: list[FutureWindowRecord] = []
    with Path(window_index_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            start_frame = int(row["start_frame_20hz"])
            end_frame = int(row["end_frame_20hz"])
            records.append(
                FutureWindowRecord(
                    feature_path=Path(row["feature_path"]),
                    start_frame=start_frame,
                    valid_frames=end_frame - start_frame + 1,
                )
            )
    if not records:
        raise ValueError(f"No windows found in {window_index_csv}")
    return records


class FuturePredictionWindowDataset(Dataset):
    def __init__(
        self,
        feature_paths: list[Path] | None = None,
        window_frames: int = 240,
        past_frames: int = 120,
        future_frames: int = 120,
        stride_frames: int = 20,
        pad_short_windows: bool = False,
        window_records: list[FutureWindowRecord] | None = None,
    ) -> None:
        self.feature_paths = [Path(path) for path in (feature_paths or [])]
        self.window_frames = int(window_frames)
        self.past_frames = int(past_frames)
        self.future_frames = int(future_frames)
        self.stride_frames = int(stride_frames)
        self.pad_short_windows = bool(pad_short_windows)
        self.window_records: list[FutureWindowRecord] = list(window_records or [])
        self._payload_cache: dict[Path, dict[str, np.ndarray]] = {}

        min_frames = self.past_frames + self.future_frames
        if self.window_frames < min_frames:
            raise ValueError("window_frames must be at least past_frames + future_frames")

        if not self.window_records:
            for feature_path in self.feature_paths:
                payload = self._load_payload(feature_path)
                total_frames = int(payload["feature"].shape[0])
                if total_frames >= min_frames:
                    max_start = total_frames - min_frames
                    starts = list(range(0, max_start + 1, self.stride_frames))
                    if not starts or starts[-1] != max_start:
                        starts.append(max_start)
                    for start_frame in starts:
                        self.window_records.append(
                            FutureWindowRecord(
                                feature_path=feature_path,
                                start_frame=int(start_frame),
                                valid_frames=min_frames,
                            )
                        )
                elif self.pad_short_windows:
                    self.window_records.append(
                        FutureWindowRecord(feature_path=feature_path, start_frame=0, valid_frames=total_frames)
                    )

        if not self.window_records:
            raise ValueError("No future prediction windows available")

    @classmethod
    def from_window_index_csv(
        cls,
        *,
        window_index_csv: Path,
        past_frames: int = 120,
        future_frames: int = 120,
        max_windows: int = 0,
    ) -> "FuturePredictionWindowDataset":
        records = read_future_window_records_from_csv(window_index_csv)
        if max_windows > 0:
            records = records[:max_windows]
        return dataset_from_future_records(
            records=records,
            past_frames=past_frames,
            future_frames=future_frames,
        )

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

        end_frame = record.start_frame + min(self.past_frames + self.future_frames, record.valid_frames)
        window_feature = feature[record.start_frame:end_frame]
        window_interpolated = interpolated[record.start_frame:end_frame]
        window_packets = packet_counter[record.start_frame:end_frame]

        past_window = np.zeros((self.past_frames, feature.shape[1], feature.shape[2]), dtype=np.float32)
        target_window = np.zeros((self.future_frames, feature.shape[1], 12), dtype=np.float32)
        target_weight = np.zeros((self.future_frames, feature.shape[1]), dtype=np.float32)

        past_valid = min(self.past_frames, window_feature.shape[0])
        future_available = max(0, window_feature.shape[0] - self.past_frames)
        future_valid = min(self.future_frames, future_available)
        past_window[:past_valid] = window_feature[:past_valid]
        if future_valid > 0:
            future_slice = window_feature[self.past_frames : self.past_frames + future_valid]
            future_interp = window_interpolated[self.past_frames : self.past_frames + future_valid]
            target_window[:future_valid] = future_slice[:, :, :12]
            target_weight[:future_valid] = (~future_interp).astype(np.float32)

        participant = str(payload["participant"]) if "participant" in payload else record.feature_path.parent.name
        segment_id = str(payload["segment_id"]) if "segment_id" in payload else record.feature_path.stem.replace("segment_", "")
        return {
            "past": torch.from_numpy(past_window),
            "target": torch.from_numpy(target_window),
            "target_weight": torch.from_numpy(target_weight),
            "meta": {
                "participant": participant,
                "segment_id": segment_id,
                "start_frame_20hz": int(record.start_frame),
                "past_frames": int(past_valid),
                "future_frames": int(future_valid),
                "packet_start_20hz": int(window_packets[0]) if window_packets.shape[0] > 0 else -1,
                "packet_end_20hz": int(window_packets[-1]) if window_packets.shape[0] > 0 else -1,
            },
        }


def dataset_from_future_records(
    *,
    records: list[FutureWindowRecord],
    past_frames: int,
    future_frames: int,
) -> FuturePredictionWindowDataset:
    if not records:
        raise ValueError("No future prediction windows available")
    return FuturePredictionWindowDataset(
        feature_paths=[],
        window_frames=past_frames + future_frames,
        past_frames=past_frames,
        future_frames=future_frames,
        stride_frames=past_frames + future_frames,
        pad_short_windows=False,
        window_records=records,
    )


def limit_future_dataset_windows(
    dataset: FuturePredictionWindowDataset,
    *,
    max_windows: int,
) -> FuturePredictionWindowDataset:
    if max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    return dataset_from_future_records(
        records=dataset.window_records[:max_windows],
        past_frames=dataset.past_frames,
        future_frames=dataset.future_frames,
    )


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_persistence_prediction(past: torch.Tensor, future_frames: int) -> torch.Tensor:
    last_frame = past[:, -1:, :, :12]
    return last_frame.expand(past.shape[0], future_frames, past.shape[2], 12).contiguous()


def compute_future_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    config: FuturePredictionTrainingConfig,
) -> torch.Tensor:
    mask = torch.ones_like(target_weight, dtype=torch.bool)
    return compute_masked_recon_loss(
        prediction=prediction,
        target=target,
        mask=mask,
        target_weight=target_weight,
        rot_loss_weight=config.rot_loss_weight,
        gyr_loss_weight=config.gyr_loss_weight,
        freeacc_loss_weight=config.freeacc_loss_weight,
    )


def _compute_future_metric_sums(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    config: FuturePredictionTrainingConfig,
) -> dict[str, float]:
    mask = torch.ones_like(target_weight, dtype=torch.bool)
    return _compute_batch_metric_sums(
        prediction=prediction,
        target=target,
        mask=mask,
        target_weight=target_weight,
        rot_loss_weight=config.rot_loss_weight,
        gyr_loss_weight=config.gyr_loss_weight,
        freeacc_loss_weight=config.freeacc_loss_weight,
    )


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def run_train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: FuturePredictionTrainingConfig,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch["past"], future_frames=config.future_frames)
        loss = compute_future_prediction_loss(
            prediction=prediction,
            target=batch["target"],
            target_weight=batch["target_weight"],
            config=config,
        )
        loss.backward()
        optimizer.step()
        merge_metric_sums(
            metric_sums,
            _compute_future_metric_sums(
                prediction=prediction.detach(),
                target=batch["target"],
                target_weight=batch["target_weight"],
                config=config,
            ),
        )
    return finalize_metric_sums(metric_sums)


@torch.no_grad()
def evaluate_future_prediction(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: FuturePredictionTrainingConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    model_sums: dict[str, float] = {}
    persistence_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        prediction = model(batch["past"], future_frames=config.future_frames)
        persistence = compute_persistence_prediction(past=batch["past"], future_frames=config.future_frames)
        merge_metric_sums(
            model_sums,
            _compute_future_metric_sums(
                prediction=prediction,
                target=batch["target"],
                target_weight=batch["target_weight"],
                config=config,
            ),
        )
        merge_metric_sums(
            persistence_sums,
            _compute_future_metric_sums(
                prediction=persistence,
                target=batch["target"],
                target_weight=batch["target_weight"],
                config=config,
            ),
        )
    return finalize_metric_sums(model_sums), finalize_metric_sums(persistence_sums)


@torch.no_grad()
def export_future_prediction_samples(
    *,
    model: torch.nn.Module,
    dataset: FuturePredictionWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    future_frames: int,
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
    past = torch.stack([sample["past"] for sample in samples], dim=0).to(device)
    target = torch.stack([sample["target"] for sample in samples], dim=0).to(device)
    target_weight = torch.stack([sample["target_weight"] for sample in samples], dim=0).to(device)
    prediction = model(past, future_frames=future_frames)
    persistence = compute_persistence_prediction(past=past, future_frames=future_frames)
    meta_json = json.dumps([sample["meta"] for sample in samples])

    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        past=past.detach().cpu().numpy().astype(np.float32),
        target=target.detach().cpu().numpy().astype(np.float32),
        prediction=prediction.detach().cpu().numpy().astype(np.float32),
        persistence=persistence.detach().cpu().numpy().astype(np.float32),
        target_weight=target_weight.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: FuturePredictionTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    persistence_metrics: dict[str, float],
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
            "persistence_metrics": persistence_metrics,
        },
        path,
    )


def run_future_prediction_training(
    train_feature_paths: list[Path] | None,
    val_feature_paths: list[Path] | None,
    output_dir: Path,
    config: FuturePredictionTrainingConfig,
    train_window_index_csv: Path | None = None,
    val_window_index_csv: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    if train_window_index_csv is not None:
        train_dataset = FuturePredictionWindowDataset.from_window_index_csv(
            window_index_csv=train_window_index_csv,
            past_frames=config.past_frames,
            future_frames=config.future_frames,
        )
    else:
        train_dataset = FuturePredictionWindowDataset(
            feature_paths=[Path(path) for path in (train_feature_paths or [])],
            window_frames=config.window_frames,
            past_frames=config.past_frames,
            future_frames=config.future_frames,
            stride_frames=config.train_stride_frames,
            pad_short_windows=True,
        )
    if val_window_index_csv is not None:
        val_dataset = FuturePredictionWindowDataset.from_window_index_csv(
            window_index_csv=val_window_index_csv,
            past_frames=config.past_frames,
            future_frames=config.future_frames,
        )
    else:
        val_dataset = FuturePredictionWindowDataset(
            feature_paths=[Path(path) for path in (val_feature_paths or [])],
            window_frames=config.window_frames,
            past_frames=config.past_frames,
            future_frames=config.future_frames,
            stride_frames=config.val_stride_frames,
            pad_short_windows=True,
        )

    if config.overfit_windows > 0:
        overfit_records = train_dataset.window_records[: config.overfit_windows]
        train_dataset = dataset_from_future_records(
            records=overfit_records,
            past_frames=config.past_frames,
            future_frames=config.future_frames,
        )
        val_dataset = dataset_from_future_records(
            records=overfit_records,
            past_frames=config.past_frames,
            future_frames=config.future_frames,
        )
    else:
        train_dataset = limit_future_dataset_windows(train_dataset, max_windows=config.max_train_windows)
        val_dataset = limit_future_dataset_windows(val_dataset, max_windows=config.max_val_windows)

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

    model = ImuFuturePredictionTransformer(
        input_dim=13,
        target_dim=12,
        d_model=config.d_model,
        nhead=config.nhead,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        dropout=config.dropout,
        max_past_frames=config.past_frames,
        max_future_frames=config.future_frames,
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
        val_metrics, persistence_metrics = evaluate_future_prediction(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config,
        )
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        record.update(_prefix_metrics("train_future", train_metrics))
        record.update(_prefix_metrics("val_future", val_metrics))
        record.update(_prefix_metrics("val_persistence", persistence_metrics))
        append_jsonl(metrics_path, record)

        export_future_prediction_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_predictions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            future_frames=config.future_frames,
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
            persistence_metrics=persistence_metrics,
        )
        if record["val_future_masked_rot_geodesic_error"] <= best_metric:
            best_metric = record["val_future_masked_rot_geodesic_error"]
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
                persistence_metrics=persistence_metrics,
            )

    return {
        "best_epoch": int(best_epoch),
        "best_val_future_masked_rot_geodesic_error": float(best_metric),
        "output_dir": str(output_dir),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="20Hz IMU future prediction baseline")
    parser.add_argument("--train-feature-path", type=Path, action="append", default=[])
    parser.add_argument("--val-feature-path", type=Path, action="append", default=[])
    parser.add_argument("--train-window-index-csv", type=Path, default=None)
    parser.add_argument("--val-window-index-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=240)
    parser.add_argument("--past-frames", type=int, default=120)
    parser.add_argument("--future-frames", type=int, default=120)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--val-stride-frames", type=int, default=120)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
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
    parser.add_argument("--gyr-loss-weight", type=float, default=0.2)
    parser.add_argument("--freeacc-loss-weight", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_source_present = bool(args.train_feature_path) or args.train_window_index_csv is not None
    val_source_present = bool(args.val_feature_path) or args.val_window_index_csv is not None
    if not train_source_present or not val_source_present:
        raise ValueError(
            "Training mode requires train/val data via --train-feature-path or --train-window-index-csv "
            "and --val-feature-path or --val-window-index-csv"
        )

    summary = run_future_prediction_training(
        train_feature_paths=args.train_feature_path,
        val_feature_paths=args.val_feature_path,
        output_dir=args.output_dir,
        config=FuturePredictionTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            window_frames=args.window_frames,
            past_frames=args.past_frames,
            future_frames=args.future_frames,
            train_stride_frames=args.train_stride_frames,
            val_stride_frames=args.val_stride_frames,
            d_model=args.d_model,
            nhead=args.nhead,
            encoder_layers=args.encoder_layers,
            decoder_layers=args.decoder_layers,
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
