from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from dataset import MotionClipDataset, augment_motion_sequence, motion_collate
from losses import ContrastiveLoss, NTXentLoss
from model import MotionConvEncoder, MotionEncoderBiGRUCo, MotionEncoderConfig


def parse_gpu_ids(gpus_arg: str | None) -> list[int]:
    if gpus_arg is None:
        return []

    gpus_arg = gpus_arg.strip()
    if not gpus_arg:
        return []

    gpu_ids = []
    for item in gpus_arg.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    return gpu_ids


def resolve_training_device(gpus_arg: str | None) -> tuple[torch.device, list[int]]:
    requested_gpu_ids = parse_gpu_ids(gpus_arg)

    if not torch.cuda.is_available():
        if requested_gpu_ids:
            raise RuntimeError("CUDA is not available, but --gpus was provided.")
        return torch.device("cpu"), []

    available_gpu_count = torch.cuda.device_count()
    if requested_gpu_ids:
        invalid_gpu_ids = [gpu_id for gpu_id in requested_gpu_ids if gpu_id < 0 or gpu_id >= available_gpu_count]
        if invalid_gpu_ids:
            raise ValueError(
                f"Invalid GPU ids {invalid_gpu_ids}. Available CUDA devices: 0..{available_gpu_count - 1}"
            )
        return torch.device(f"cuda:{requested_gpu_ids[0]}"), requested_gpu_ids

    return torch.device("cuda:0"), [0]


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a standalone motion encoder for contrastive representation learning")
    parser.add_argument("--pose-root", type=Path, default=repo_root / "from_jiatong" / "pseudo_pose_20hz")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "motion_encoder" / "results")
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--clip-mode", choices=["full_clip", "future_only", "dit_conditional_target"], default="future_only")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--embedding-size", type=int, default=256)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--encoder-type", choices=["gru", "conv"], default="gru")
    parser.add_argument("--loss-type", choices=["nt_xent", "pairwise"], default="nt_xent")
    parser.add_argument("--margin", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--jitter-std", type=float, default=0.01)
    parser.add_argument("--frame-dropout-prob", type=float, default=0.05)
    parser.add_argument("--scale-jitter", type=float, default=0.05)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated CUDA device ids for training, e.g. '0' or '0,1'. Default: use cuda:0 if available, else CPU.",
    )
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def create_output_dir(results_dir: Path) -> Path:
    output_dir = results_dir / datetime.now().strftime("encoder_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def build_encoder(config: MotionEncoderConfig, encoder_type: str) -> torch.nn.Module:
    if encoder_type == "gru":
        return MotionEncoderBiGRUCo(config)
    if encoder_type == "conv":
        return MotionConvEncoder(config)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    base_model = unwrap_model(model)
    torch.save(
        {
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = create_output_dir(args.results_dir)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    device, gpu_ids = resolve_training_device(args.gpus)
    print(f"Using device: {device}")
    if gpu_ids:
        print(f"CUDA devices for training: {gpu_ids}")
    else:
        print("Training on CPU")

    dataset = MotionClipDataset(
        pose_root=args.pose_root,
        split=args.split,
        clip_mode=args.clip_mode,
        include_heading=True,
        normalize_per_clip=True,
        max_clips=args.max_clips,
        augment=False,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=motion_collate,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    input_size = int(dataset[0].sequence.shape[-1])
    config = MotionEncoderConfig(
        input_size=input_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        projection_size=args.projection_size,
    )
    model = build_encoder(config, args.encoder_type).to(device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Enabled DataParallel across {len(gpu_ids)} GPUs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pairwise_loss = ContrastiveLoss(margin=args.margin)
    ntxent_loss = NTXentLoss(temperature=args.temperature)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch in loader:
            sequence = batch["sequence"].to(device=device, dtype=torch.float32)
            lengths = batch["lengths"].to(device=device, dtype=torch.long)

            seq_np = sequence.detach().cpu().numpy()
            view1 = []
            view2 = []
            for sample in seq_np:
                view1.append(
                    augment_motion_sequence(
                        sample,
                        jitter_std=args.jitter_std,
                        frame_dropout_prob=args.frame_dropout_prob,
                        scale_jitter=args.scale_jitter,
                        rng=dataset.rng,
                    )
                )
                view2.append(
                    augment_motion_sequence(
                        sample,
                        jitter_std=args.jitter_std,
                        frame_dropout_prob=args.frame_dropout_prob,
                        scale_jitter=args.scale_jitter,
                        rng=dataset.rng,
                    )
                )
            view1 = torch.from_numpy(np.stack(view1, axis=0)).to(device=device, dtype=torch.float32)
            view2 = torch.from_numpy(np.stack(view2, axis=0)).to(device=device, dtype=torch.float32)

            _, proj1 = model(view1, lengths)
            _, proj2 = model(view2, lengths)
            if args.loss_type == "nt_xent":
                loss = ntxent_loss(proj1, proj2)
            else:
                positive = pairwise_loss(
                    proj1,
                    proj2,
                    torch.zeros(proj1.shape[0], 1, device=device, dtype=proj1.dtype),
                )
                perm = torch.randperm(proj2.shape[0], device=device)
                negative = pairwise_loss(
                    proj1,
                    proj2[perm],
                    torch.ones(proj1.shape[0], 1, device=device, dtype=proj1.dtype),
                )
                loss = 0.5 * (positive + negative)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        epoch_loss = running_loss / max(num_batches, 1)
        print(f"Epoch {epoch:03d} | contrastive_loss={epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(output_dir / "best.pt", model=model, optimizer=optimizer, args=args, epoch=epoch)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model=model, optimizer=optimizer, args=args, epoch=epoch)
            save_checkpoint(output_dir / "latest.pt", model=model, optimizer=optimizer, args=args, epoch=epoch)


if __name__ == "__main__":
    main()
