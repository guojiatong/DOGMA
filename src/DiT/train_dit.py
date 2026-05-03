#!/usr/bin/env python3
"""
Train a conditional DiT directly in pose space.

This keeps the latent-diffusion data pipeline and training loop structure, but
replaces the frozen-VAE latent target with direct diffusion over the flattened
pose window. The DiT block structure follows the adaLN-zero style used in
`DiT/motion_models.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from timm.models.vision_transformer import Attention, Mlp
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from train_imu_masked_recon import append_jsonl, build_scheduler, resolve_device, set_global_seed, write_json
from train_temporal_vae import (
    DEFAULT_FEATURE_ROOT,
    DEFAULT_POSE_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    PoseWindowRecord,
    apply_position_temporal_filter,
    apply_data_config_to_args,
    compute_heading_forward_loss,
    compute_pose_recon_metrics,
    flatten_pose_window,
    load_pose_window_records,
    pose_heading_slice,
    rebase_root_heading_6d,
    rebase_root_translation,
    resolve_position_smoothing_kernel,
    select_pose_window_records,
    write_pose_window_subset_csv,
)


CONDITION_DIM = 10 * 13
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_TRAIN_WINDOW_INDEX_CSV = DEFAULT_CONFIG_DIR / "vae_window_index_train.csv"
DEFAULT_VAL_WINDOW_INDEX_CSV = DEFAULT_CONFIG_DIR / "vae_window_index_val.csv"


@dataclass(frozen=True)
class DiTTrainingConfig:
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    past_frames: int = 120
    future_window_frames: int = 40
    stride_frames: int = 0
    train_stride_frames: int = 20
    val_stride_frames: int = 120
    condition_hidden_dim: int = 128
    denoiser_num_blocks: int = 4
    fusion_mode: str = "add"
    diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    dropout: float = 0.1
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 0
    scheduler_type: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.5
    scheduler_t_max: int = 0
    max_train_windows: int = 0
    max_val_windows: int = 0
    overfit_windows: int = 0
    shuffle_train_windows: bool = False
    shuffle_val_windows: bool = False
    window_subset_seed: int = 0
    sample_export_count: int = 0
    sample_seed: int = 0
    sample_position_smoothing_kernel: str = "tri5"
    future_heading_loss_weight: float = 0.0
    transition_heading_loss_weight: float = 0.0
    transition_heading_frames: int = 10
    dit_num_heads: int = 8
    dit_mlp_ratio: float = 4.0
    include_root_translation: bool = False


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        angles = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(timesteps, self.frequency_embedding_size))


class TokenEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class PoseDiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class PoseDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, positions)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError(f"Expected even embed_dim, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    positions = positions.reshape(-1)
    out = np.einsum("m,d->md", positions, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    embedding = np.concatenate([emb_sin, emb_cos], axis=1)
    if embedding.shape[1] != embed_dim:
        raise ValueError(f"Unexpected positional embedding shape: {embedding.shape}")
    return embedding


class ConditionalPoseDiT(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int,
        condition_dim: int,
        hidden_size: int,
        pose_frames: int,
        condition_frames: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if pose_frames <= 0:
            raise ValueError(f"pose_frames must be positive, got {pose_frames}")
        if condition_frames <= 0:
            raise ValueError(f"condition_frames must be positive, got {condition_frames}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0 or hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.pose_dim = int(pose_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_size = int(hidden_size)
        self.pose_frames = int(pose_frames)
        self.condition_frames = int(condition_frames)
        self.total_frames = self.condition_frames + self.pose_frames
        self.condition_embedder = TokenEmbedder(self.condition_dim, self.hidden_size)
        self.pose_embedder = TokenEmbedder(self.pose_dim, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size)
        self.type_embed = nn.Embedding(2, self.hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_frames, self.hidden_size), requires_grad=False)
        self.blocks = nn.ModuleList(
            [PoseDiTBlock(self.hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio) for _ in range(int(depth))]
        )
        self.final_layer = PoseDiTFinalLayer(self.hidden_size, self.pose_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.total_frames)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.condition_embedder.proj.weight)
        nn.init.constant_(self.condition_embedder.proj.bias, 0)
        nn.init.xavier_uniform_(self.pose_embedder.proj.weight)
        nn.init.constant_(self.pose_embedder.proj.bias, 0)
        nn.init.normal_(self.type_embed.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        noisy_pose: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_pose.ndim != 3:
            raise ValueError(f"noisy_pose must have shape [B,T,C], got {noisy_pose.shape}")
        if condition.ndim != 3:
            raise ValueError(f"condition must have shape [B,T,C], got {condition.shape}")
        if noisy_pose.shape[1] != self.pose_frames:
            raise ValueError(f"Expected {self.pose_frames} pose frames, got {noisy_pose.shape[1]}")
        if noisy_pose.shape[2] != self.pose_dim:
            raise ValueError(f"Expected pose dim {self.pose_dim}, got {noisy_pose.shape[2]}")
        if condition.shape[1] != self.condition_frames:
            raise ValueError(f"Expected {self.condition_frames} condition frames, got {condition.shape[1]}")
        if condition.shape[2] != self.condition_dim:
            raise ValueError(f"Expected condition dim {self.condition_dim}, got {condition.shape[2]}")
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy_pose.shape[0]:
            raise ValueError(f"timesteps must have shape [{noisy_pose.shape[0]}], got {timesteps.shape}")

        batch_size = noisy_pose.shape[0]
        condition_tokens = self.condition_embedder(condition)
        pose_tokens = self.pose_embedder(noisy_pose)
        x = torch.cat([condition_tokens, pose_tokens], dim=1)
        token_types = torch.cat(
            [
                torch.zeros(self.condition_frames, dtype=torch.long, device=noisy_pose.device),
                torch.ones(self.pose_frames, dtype=torch.long, device=noisy_pose.device),
            ],
            dim=0,
        )
        x = x + self.pos_embed[:, : self.total_frames] + self.type_embed(token_types).unsqueeze(0)
        c = self.t_embedder(timesteps)
        for block in self.blocks:
            x = block(x, c)
        prediction = self.final_layer(x[:, self.condition_frames :, :], c)
        if prediction.shape != (batch_size, self.pose_frames, self.pose_dim):
            raise ValueError(
                f"Unexpected prediction shape: expected {(batch_size, self.pose_frames, self.pose_dim)}, "
                f"got {tuple(prediction.shape)}"
            )
        return prediction


def build_dit_model(
    *,
    pose_dim: int,
    condition_frames: int,
    condition_hidden_dim: int,
    pose_frames: int,
    dropout: float,
    denoiser_num_blocks: int,
    fusion_mode: str,
    dit_num_heads: int,
    dit_mlp_ratio: float,
) -> ConditionalPoseDiT:
    del dropout
    del fusion_mode
    return ConditionalPoseDiT(
        pose_dim=pose_dim,
        condition_dim=CONDITION_DIM,
        hidden_size=condition_hidden_dim,
        pose_frames=pose_frames,
        condition_frames=condition_frames,
        depth=denoiser_num_blocks,
        num_heads=dit_num_heads,
        mlp_ratio=dit_mlp_ratio,
    )


class DiTWindowDataset(Dataset):
    def __init__(
        self,
        *,
        window_records: list[PoseWindowRecord],
        past_frames: int = 120,
        future_window_frames: int = 40,
        include_root_translation: bool = False,
    ) -> None:
        self.window_records = [
            record
            for record in window_records
            if record.start_frame >= past_frames and record.valid_frames >= future_window_frames
        ]
        self.past_frames = int(past_frames)
        self.future_window_frames = int(future_window_frames)
        self.include_root_translation = bool(include_root_translation)
        self._feature_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._pose_cache: dict[Path, dict[str, np.ndarray]] = {}
        if not self.window_records:
            raise ValueError("No DiT windows available")

    @classmethod
    def from_window_index_csv(
        cls,
        *,
        window_index_csv: Path,
        past_frames: int = 120,
        future_window_frames: int = 40,
        max_windows: int = 0,
        shuffle: bool = False,
        subset_seed: int = 0,
        include_root_translation: bool = False,
    ) -> "DiTWindowDataset":
        records = read_pose_window_records_from_csv(window_index_csv)
        records = [
            record
            for record in records
            if record.start_frame >= past_frames and record.valid_frames >= future_window_frames
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
            include_root_translation=include_root_translation,
        )

    def _load_feature_payload(self, feature_path: Path) -> dict[str, np.ndarray]:
        if feature_path not in self._feature_cache:
            with np.load(feature_path, allow_pickle=False) as payload:
                self._feature_cache[feature_path] = {key: payload[key] for key in payload.files}
        return self._feature_cache[feature_path]

    def _load_pose_payload(self, pose_path: Path) -> dict[str, np.ndarray]:
        if pose_path not in self._pose_cache:
            with np.load(pose_path, allow_pickle=False) as payload:
                self._pose_cache[pose_path] = {key: payload[key] for key in payload.files}
        return self._pose_cache[pose_path]

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

        condition = (
            feature_payload["feature"][past_start:past_end]
            .reshape(self.past_frames, CONDITION_DIM)
            .astype(np.float32)
        )
        relative_positions = pose_payload["relative_positions"][combined_start:combined_end].astype(np.float32)
        root_heading_6d = rebase_root_heading_6d(
            pose_payload["root_heading_6d"][combined_start:combined_end].astype(np.float32)
        )
        root_translation = None
        if self.include_root_translation:
            if "root_translation" not in feature_payload:
                raise KeyError(f"root_translation not found in {record.feature_path}")
            root_translation = rebase_root_translation(
                feature_payload["root_translation"][combined_start:combined_end].astype(np.float32)
            )
        target_pose = flatten_pose_window(relative_positions, root_heading_6d, root_translation).astype(np.float32)
        packet_counter = pose_payload["packet_counter"][record.start_frame:future_end].astype(np.int64)
        participant = (
            str(feature_payload["participant"])
            if "participant" in feature_payload
            else record.feature_path.parent.name
        )
        segment_id = (
            str(feature_payload["segment_id"])
            if "segment_id" in feature_payload
            else record.feature_path.stem.replace("segment_", "")
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
                "packet_start_20hz": int(packet_counter[0]),
                "packet_end_20hz": int(packet_counter[-1]),
            },
        }


def compute_condition_normalization_stats(dataset: DiTWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((CONDITION_DIM,), dtype=np.float64)
    sums_sq = np.zeros((CONDITION_DIM,), dtype=np.float64)
    count = 0
    for index in range(len(dataset)):
        condition = dataset[index]["condition"].numpy().astype(np.float64)
        sums += condition.sum(axis=0)
        sums_sq += np.square(condition).sum(axis=0)
        count += condition.shape[0]
    if count <= 0:
        raise ValueError("No DiT condition frames available for normalization")
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def compute_target_pose_normalization_stats(dataset: DiTWindowDataset) -> tuple[np.ndarray, np.ndarray]:
    sums: np.ndarray | None = None
    sums_sq: np.ndarray | None = None
    count = 0
    for index in range(len(dataset)):
        target_pose = dataset[index]["target_pose"].numpy().astype(np.float64)
        flattened = target_pose.reshape(-1, target_pose.shape[-1])
        if sums is None:
            sums = flattened.sum(axis=0)
            sums_sq = np.square(flattened).sum(axis=0)
        else:
            sums += flattened.sum(axis=0)
            sums_sq += np.square(flattened).sum(axis=0)
        count += flattened.shape[0]
    if sums is None or sums_sq is None or count <= 0:
        raise ValueError("No target pose frames available for normalization")
    mean = sums / count
    variance = np.maximum(sums_sq / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def normalize_condition_tensor(
    condition: torch.Tensor,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
) -> torch.Tensor:
    return (condition - condition_mean) / condition_std


def normalize_target_tensor(
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return (target - target_mean) / target_std


def denormalize_target_tensor(
    target: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    return target * target_std + target_mean


def predict_x0_from_noise(
    *,
    noisy_target: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_alpha_cumprod = _extract(diffusion_buffers["sqrt_alphas_cumprod"], timesteps, noisy_target.ndim)
    sqrt_one_minus = _extract(diffusion_buffers["sqrt_one_minus_alphas_cumprod"], timesteps, noisy_target.ndim)
    return (noisy_target - sqrt_one_minus * predicted_noise) / torch.clamp(sqrt_alpha_cumprod, min=1e-8)


def _move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_beta_schedule(*, diffusion_steps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    if diffusion_steps <= 0:
        raise ValueError(f"diffusion_steps must be positive, got {diffusion_steps}")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise ValueError("beta_start and beta_end must be positive")
    return torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)


def build_diffusion_buffers(
    *,
    diffusion_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    betas = build_beta_schedule(diffusion_steps=diffusion_steps, beta_start=beta_start, beta_end=beta_end).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def load_dit_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[
    ConditionalPoseDiT,
    DiTTrainingConfig,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = DiTTrainingConfig(**checkpoint["config"])
    pose_mean_np = checkpoint.get("target_mean")
    pose_std_np = checkpoint.get("target_std")
    if pose_mean_np is None or pose_std_np is None:
        raise ValueError("Checkpoint is missing target_mean/target_std for direct DiT evaluation")
    condition_mean = torch.as_tensor(checkpoint["condition_mean"], dtype=torch.float32, device=device).view(1, 1, -1)
    condition_std = torch.as_tensor(checkpoint["condition_std"], dtype=torch.float32, device=device).view(1, 1, -1)
    target_mean = torch.as_tensor(pose_mean_np, dtype=torch.float32, device=device).view(1, 1, -1)
    target_std = torch.as_tensor(pose_std_np, dtype=torch.float32, device=device).view(1, 1, -1)
    pose_frames = int(checkpoint["pose_frames"])
    pose_dim = int(checkpoint["pose_dim"])
    model = build_dit_model(
        pose_dim=pose_dim,
        condition_frames=config.past_frames,
        condition_hidden_dim=config.condition_hidden_dim,
        pose_frames=pose_frames,
        dropout=config.dropout,
        denoiser_num_blocks=config.denoiser_num_blocks,
        fusion_mode=config.fusion_mode,
        dit_num_heads=config.dit_num_heads,
        dit_mlp_ratio=config.dit_mlp_ratio,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion_buffers = build_diffusion_buffers(
        diffusion_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    return (
        model,
        config,
        condition_mean,
        condition_std,
        target_mean,
        target_std,
        diffusion_buffers,
        checkpoint,
    )


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_ndim: int) -> torch.Tensor:
    values = buffer.index_select(0, timesteps.long())
    return values.view(values.shape[0], *([1] * (target_ndim - 1)))


def q_sample(
    *,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (
        _extract(diffusion_buffers["sqrt_alphas_cumprod"], timesteps, x_start.ndim) * x_start
        + _extract(diffusion_buffers["sqrt_one_minus_alphas_cumprod"], timesteps, x_start.ndim) * noise
    )


def compute_x0_heading_metrics(
    *,
    denoised_pose: torch.Tensor,
    target_pose: torch.Tensor,
    past_frames: int,
    future_frames: int,
    transition_frames: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    future_start = int(past_frames)
    future_end = int(past_frames + future_frames)
    prediction_future = denoised_pose[:, future_start:future_end]
    target_future = target_pose[:, future_start:future_end]
    heading_slice = pose_heading_slice()
    future_heading_loss = compute_heading_forward_loss(
        prediction_future[..., heading_slice],
        target_future[..., heading_slice],
    )
    future_metrics = compute_pose_recon_metrics(
        prediction=prediction_future,
        target=target_future,
    )

    transition_window = max(int(transition_frames), 0)
    transition_start = max(int(past_frames) - 1, 0)
    transition_end = min(int(denoised_pose.shape[1]), int(past_frames) + transition_window)
    if transition_end - transition_start <= 0:
        transition_heading_loss = denoised_pose.new_zeros(())
        transition_heading_error_deg = 0.0
    else:
        prediction_transition = denoised_pose[:, transition_start:transition_end]
        target_transition = target_pose[:, transition_start:transition_end]
        transition_heading_loss = compute_heading_forward_loss(
            prediction_transition[..., heading_slice],
            target_transition[..., heading_slice],
        )
        transition_heading_error_deg = float(
            compute_pose_recon_metrics(
                prediction=prediction_transition,
                target=target_transition,
            )["root_heading_error_deg"]
        )

    loss_terms = {
        "x0_future_heading_forward_loss": future_heading_loss,
        "x0_transition_heading_forward_loss": transition_heading_loss,
    }
    metric_terms = {
        "x0_future_recon_mpjpe": float(future_metrics["recon_mpjpe"]),
        "x0_future_root_heading_error_deg": float(future_metrics["root_heading_error_deg"]),
        "x0_future_jerk_error": float(future_metrics["jerk_error"]),
        "x0_future_heading_forward_loss": float(future_heading_loss.item()),
        "x0_transition_root_heading_error_deg": float(transition_heading_error_deg),
        "x0_transition_heading_forward_loss": float(transition_heading_loss.item()),
    }
    return loss_terms, metric_terms


def compute_noise_prediction_loss(
    *,
    model: torch.nn.Module,
    condition: torch.Tensor,
    target_pose_normalized: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    target_pose_raw: torch.Tensor | None = None,
    target_mean: torch.Tensor | None = None,
    target_std: torch.Tensor | None = None,
    position_smoothing_kernel: tuple[float, ...] | None = None,
    past_frames: int = 0,
    future_frames: int = 0,
    future_heading_loss_weight: float = 0.0,
    transition_heading_loss_weight: float = 0.0,
    transition_heading_frames: int = 0,
    track_x0_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    timesteps = torch.randint(
        low=0,
        high=int(diffusion_buffers["betas"].shape[0]),
        size=(target_pose_normalized.shape[0],),
        device=target_pose_normalized.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(target_pose_normalized)
    noisy_target = q_sample(
        x_start=target_pose_normalized,
        timesteps=timesteps,
        noise=noise,
        diffusion_buffers=diffusion_buffers,
    )
    predicted_noise = model(noisy_target, condition, timesteps)
    noise_mse = torch.mean((predicted_noise - noise) ** 2)
    loss = noise_mse
    metrics = {
        "noise_mse": float(noise_mse.item()),
        "target_std": float(target_pose_normalized.std(unbiased=False).item()),
    }
    should_track_x0 = bool(track_x0_metrics or future_heading_loss_weight > 0.0 or transition_heading_loss_weight > 0.0)
    if should_track_x0:
        if target_pose_raw is None or target_mean is None or target_std is None:
            raise ValueError("x0 metrics require target_pose_raw, target_mean, and target_std")
        x0_prediction = predict_x0_from_noise(
            noisy_target=noisy_target,
            predicted_noise=predicted_noise,
            timesteps=timesteps,
            diffusion_buffers=diffusion_buffers,
        )
        denoised_pose = denormalize_target_tensor(x0_prediction, target_mean, target_std)
        denoised_pose = apply_position_temporal_filter(
            denoised_pose,
            kernel_weights=position_smoothing_kernel,
        )
        x0_loss_terms, x0_metrics = compute_x0_heading_metrics(
            denoised_pose=denoised_pose,
            target_pose=target_pose_raw,
            past_frames=past_frames,
            future_frames=future_frames,
            transition_frames=transition_heading_frames,
        )
        loss = (
            loss
            + float(future_heading_loss_weight) * x0_loss_terms["x0_future_heading_forward_loss"]
            + float(transition_heading_loss_weight) * x0_loss_terms["x0_transition_heading_forward_loss"]
        )
        metrics.update(x0_metrics)
    return loss, metrics


def _accumulate_metric_sums(metric_sums: dict[str, float], metrics: dict[str, float], batch_size: int) -> None:
    metric_sums["window_count"] = metric_sums.get("window_count", 0.0) + float(batch_size)
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * float(batch_size)


def _finalize_metric_sums(metric_sums: dict[str, float]) -> dict[str, float]:
    count = max(metric_sums.get("window_count", 0.0), 1.0)
    return {
        key: float(value / count)
        for key, value in metric_sums.items()
        if key != "window_count"
    }


def run_train_epoch(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    position_smoothing_kernel: tuple[float, ...] | None,
    past_frames: int,
    future_frames: int,
    future_heading_loss_weight: float,
    transition_heading_loss_weight: float,
    transition_heading_frames: int,
) -> dict[str, float]:
    model.train()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_pose_raw = batch["target_pose"]
        target_pose = normalize_target_tensor(target_pose_raw, target_mean, target_std)
        loss, metrics = compute_noise_prediction_loss(
            model=model,
            condition=condition,
            target_pose_normalized=target_pose,
            diffusion_buffers=diffusion_buffers,
            target_pose_raw=target_pose_raw,
            target_mean=target_mean,
            target_std=target_std,
            position_smoothing_kernel=position_smoothing_kernel,
            past_frames=past_frames,
            future_frames=future_frames,
            future_heading_loss_weight=future_heading_loss_weight,
            transition_heading_loss_weight=transition_heading_loss_weight,
            transition_heading_frames=transition_heading_frames,
            track_x0_metrics=(future_heading_loss_weight > 0.0 or transition_heading_loss_weight > 0.0),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _accumulate_metric_sums(metric_sums, metrics, target_pose_raw.shape[0])
    return _finalize_metric_sums(metric_sums)


@torch.no_grad()
def evaluate_dit(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    position_smoothing_kernel: tuple[float, ...] | None,
    past_frames: int,
    future_frames: int,
    transition_heading_frames: int,
) -> dict[str, float]:
    model.eval()
    metric_sums: dict[str, float] = {}
    for batch in dataloader:
        batch = _move_tensor_batch(batch, device)
        condition = normalize_condition_tensor(batch["condition"], condition_mean, condition_std)
        target_pose_raw = batch["target_pose"]
        target_pose = normalize_target_tensor(target_pose_raw, target_mean, target_std)
        _, metrics = compute_noise_prediction_loss(
            model=model,
            condition=condition,
            target_pose_normalized=target_pose,
            diffusion_buffers=diffusion_buffers,
            target_pose_raw=target_pose_raw,
            target_mean=target_mean,
            target_std=target_std,
            position_smoothing_kernel=position_smoothing_kernel,
            past_frames=past_frames,
            future_frames=future_frames,
            transition_heading_frames=transition_heading_frames,
            track_x0_metrics=True,
        )
        _accumulate_metric_sums(metric_sums, metrics, target_pose_raw.shape[0])
    return _finalize_metric_sums(metric_sums)


@torch.no_grad()
def sample_pose_diffusion(
    *,
    model: torch.nn.Module,
    condition: torch.Tensor,
    pose_shape: tuple[int, int, int],
    diffusion_buffers: dict[str, torch.Tensor],
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    model.eval()
    current = torch.randn(pose_shape, device=condition.device, dtype=torch.float32, generator=generator)
    total_steps = int(diffusion_buffers["betas"].shape[0])
    for step in range(total_steps - 1, -1, -1):
        timesteps = torch.full((pose_shape[0],), step, dtype=torch.long, device=condition.device)
        predicted_noise = model(current, condition, timesteps)
        alpha = _extract(diffusion_buffers["alphas"], timesteps, current.ndim)
        alpha_cumprod = _extract(diffusion_buffers["alphas_cumprod"], timesteps, current.ndim)
        beta = _extract(diffusion_buffers["betas"], timesteps, current.ndim)
        if step > 0:
            noise = torch.randn(current.shape, device=current.device, dtype=current.dtype, generator=generator)
        else:
            noise = torch.zeros_like(current)
        current = (1.0 / torch.sqrt(alpha)) * (
            current - ((1.0 - alpha) / torch.sqrt(torch.clamp(1.0 - alpha_cumprod, min=1e-8))) * predicted_noise
        ) + torch.sqrt(beta) * noise
    return current


def build_sampling_generator(
    *,
    device: torch.device,
    base_seed: int,
    meta: dict[str, Any],
    sample_index: int,
) -> torch.Generator:
    stable_key = "{participant}|{segment_id}|{start_frame_20hz}|{packet_start_20hz}".format(
        participant=meta["participant"],
        segment_id=meta["segment_id"],
        start_frame_20hz=meta["start_frame_20hz"],
        packet_start_20hz=meta["packet_start_20hz"],
    )
    stable_seed = zlib.adler32(stable_key.encode("utf-8")) & 0xFFFFFFFF
    seed = int((int(base_seed) + stable_seed + 9973 * int(sample_index)) % (2**31 - 1))
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def export_dit_samples(
    *,
    model: torch.nn.Module,
    dataset: DiTWindowDataset,
    device: torch.device,
    output_path: Path,
    epoch: int,
    sample_count: int,
    sample_seed: int,
    condition_mean: torch.Tensor,
    condition_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    diffusion_buffers: dict[str, torch.Tensor],
    position_smoothing_kernel: tuple[float, ...] | None,
) -> None:
    if sample_count <= 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = min(int(sample_count), len(dataset))
    rng = np.random.default_rng(sample_seed + epoch)
    if count == len(dataset):
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        indices = np.sort(rng.choice(len(dataset), size=count, replace=False))

    samples = [dataset[int(index)] for index in indices]
    condition = torch.stack([sample["condition"] for sample in samples], dim=0).to(device)
    target_pose_raw = torch.stack([sample["target_pose"] for sample in samples], dim=0).to(device)
    condition = normalize_condition_tensor(condition, condition_mean, condition_std)
    target_pose = normalize_target_tensor(target_pose_raw, target_mean, target_std)
    sampled_pose_normalized = sample_pose_diffusion(
        model=model,
        condition=condition,
        pose_shape=tuple(target_pose.shape),
        diffusion_buffers=diffusion_buffers,
        generator=torch.Generator(device=device.type).manual_seed(int(sample_seed + epoch)),
    )
    sampled_pose = denormalize_target_tensor(sampled_pose_normalized, target_mean, target_std)
    sampled_pose = apply_position_temporal_filter(
        sampled_pose,
        kernel_weights=position_smoothing_kernel,
    )
    meta_json = json.dumps([sample["meta"] for sample in samples])
    np.savez_compressed(
        output_path,
        epoch=np.asarray(epoch, dtype=np.int64),
        indices=indices.astype(np.int64),
        condition=condition.detach().cpu().numpy().astype(np.float32),
        target_pose=target_pose_raw.detach().cpu().numpy().astype(np.float32),
        sampled_pose=sampled_pose.detach().cpu().numpy().astype(np.float32),
        sampled_pose_normalized=sampled_pose_normalized.detach().cpu().numpy().astype(np.float32),
        meta_json=np.asarray(meta_json),
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: DiTTrainingConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    pose_frames: int,
    pose_dim: int,
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
            "condition_mean": condition_mean.astype(np.float32),
            "condition_std": condition_std.astype(np.float32),
            "target_mean": target_mean.astype(np.float32),
            "target_std": target_std.astype(np.float32),
            "pose_frames": int(pose_frames),
            "pose_dim": int(pose_dim),
        },
        path,
    )


def run_dit_training(
    *,
    output_dir: Path,
    config: DiTTrainingConfig,
    train_window_index_csv: Path | None,
    val_window_index_csv: Path | None,
    split_manifest_path: Path | None,
    pose_root: Path,
    feature_root: Path,
    train_split: str = "train",
    val_split: str = "val",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(config.seed)
    device = resolve_device(config.device)

    combined_window_frames = int(config.past_frames + config.future_window_frames)
    shared_stride = int(config.stride_frames)
    train_stride = shared_stride if shared_stride > 0 else int(config.train_stride_frames)
    val_stride = shared_stride if shared_stride > 0 else int(config.val_stride_frames)
    train_records = load_pose_window_records(
        window_index_csv=train_window_index_csv,
        split_manifest_path=split_manifest_path,
        split=train_split,
        window_frames=combined_window_frames,
        stride_frames=train_stride,
        pose_root=pose_root,
        feature_root=feature_root,
    )
    val_records = load_pose_window_records(
        window_index_csv=val_window_index_csv,
        split_manifest_path=split_manifest_path,
        split=val_split,
        window_frames=combined_window_frames,
        stride_frames=val_stride,
        pose_root=pose_root,
        feature_root=feature_root,
    )

    if config.overfit_windows > 0:
        overfit_records = select_pose_window_records(
            train_records,
            max_windows=config.overfit_windows,
            shuffle=config.shuffle_train_windows,
            subset_seed=config.window_subset_seed,
        )
        train_selected_records = list(overfit_records)
        val_selected_records = list(overfit_records)
    else:
        train_selected_records = select_pose_window_records(
            train_records,
            max_windows=config.max_train_windows,
            shuffle=config.shuffle_train_windows,
            subset_seed=config.window_subset_seed,
        )
        val_selected_records = select_pose_window_records(
            val_records,
            max_windows=config.max_val_windows,
            shuffle=config.shuffle_val_windows,
            subset_seed=config.window_subset_seed + 1,
        )

    write_pose_window_subset_csv(output_dir / "train_window_subset.csv", train_selected_records)
    write_pose_window_subset_csv(output_dir / "val_window_subset.csv", val_selected_records)

    train_dataset = DiTWindowDataset(
        window_records=train_selected_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        include_root_translation=config.include_root_translation,
    )
    val_dataset = DiTWindowDataset(
        window_records=val_selected_records,
        past_frames=config.past_frames,
        future_window_frames=config.future_window_frames,
        include_root_translation=config.include_root_translation,
    )
    condition_mean_np, condition_std_np = compute_condition_normalization_stats(train_dataset)
    target_mean_np, target_std_np = compute_target_pose_normalization_stats(train_dataset)

    condition_mean = torch.from_numpy(condition_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    condition_std = torch.from_numpy(condition_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    target_mean = torch.from_numpy(target_mean_np).to(device=device, dtype=torch.float32).view(1, 1, -1)
    target_std = torch.from_numpy(target_std_np).to(device=device, dtype=torch.float32).view(1, 1, -1)

    np.savez_compressed(
        output_dir / "condition_normalization_stats.npz",
        condition_mean=condition_mean_np.astype(np.float32),
        condition_std=condition_std_np.astype(np.float32),
    )
    np.savez_compressed(
        output_dir / "target_pose_normalization_stats.npz",
        target_mean=target_mean_np.astype(np.float32),
        target_std=target_std_np.astype(np.float32),
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

    sample_shape = train_dataset[0]["target_pose"].shape
    pose_frames = int(sample_shape[0])
    pose_dim = int(sample_shape[1])
    if pose_frames != int(config.past_frames + config.future_window_frames):
        raise ValueError(
            f"Unexpected pose frame count {pose_frames}; expected {config.past_frames + config.future_window_frames}"
        )

    model = build_dit_model(
        pose_dim=pose_dim,
        condition_frames=config.past_frames,
        condition_hidden_dim=config.condition_hidden_dim,
        pose_frames=pose_frames,
        dropout=config.dropout,
        denoiser_num_blocks=config.denoiser_num_blocks,
        fusion_mode=config.fusion_mode,
        dit_num_heads=config.dit_num_heads,
        dit_mlp_ratio=config.dit_mlp_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = build_scheduler(optimizer=optimizer, config=config)
    diffusion_buffers = build_diffusion_buffers(
        diffusion_steps=config.diffusion_steps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        device=device,
    )
    sample_position_smoothing_kernel = resolve_position_smoothing_kernel(config.sample_position_smoothing_kernel)

    args_payload = asdict(config) | {
        "train_window_index_csv": None if train_window_index_csv is None else str(train_window_index_csv),
        "val_window_index_csv": None if val_window_index_csv is None else str(val_window_index_csv),
        "split_manifest_path": None if split_manifest_path is None else str(split_manifest_path),
        "pose_root": str(pose_root),
        "feature_root": str(feature_root),
        "train_split": train_split,
        "val_split": val_split,
        "train_stride_frames_resolved": int(train_stride),
        "val_stride_frames_resolved": int(val_stride),
        "combined_target_window_frames": pose_frames,
        "target_pose_dim": pose_dim,
        "target_normalization": "train_split_per_dim",
        "device_resolved": str(device),
    }
    write_json(output_dir / "args.json", args_payload)

    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    best_metric = float("inf")
    best_epoch = 0
    best_heading_metric = float("inf")
    best_heading_epoch = 0
    epoch_progress = tqdm(range(1, config.epochs + 1), desc="epochs")
    for epoch in epoch_progress:
        train_metrics = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            condition_mean=condition_mean,
            condition_std=condition_std,
            target_mean=target_mean,
            target_std=target_std,
            diffusion_buffers=diffusion_buffers,
            position_smoothing_kernel=sample_position_smoothing_kernel,
            past_frames=config.past_frames,
            future_frames=config.future_window_frames,
            future_heading_loss_weight=config.future_heading_loss_weight,
            transition_heading_loss_weight=config.transition_heading_loss_weight,
            transition_heading_frames=config.transition_heading_frames,
        )
        val_metrics = evaluate_dit(
            model=model,
            dataloader=val_loader,
            device=device,
            condition_mean=condition_mean,
            condition_std=condition_std,
            target_mean=target_mean,
            target_std=target_std,
            diffusion_buffers=diffusion_buffers,
            position_smoothing_kernel=sample_position_smoothing_kernel,
            past_frames=config.past_frames,
            future_frames=config.future_window_frames,
            transition_heading_frames=config.transition_heading_frames,
        )
        record = {
            "epoch": int(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(value) for key, value in train_metrics.items()},
            **{f"val_{key}": float(value) for key, value in val_metrics.items()},
        }
        epoch_progress.set_postfix(
            train_noise_mse=f"{record['train_noise_mse']:.4f}",
            val_noise_mse=f"{record['val_noise_mse']:.4f}",
        )
        append_jsonl(metrics_path, record)

        export_dit_samples(
            model=model,
            dataset=val_dataset,
            device=device,
            output_path=output_dir / "sample_predictions" / f"epoch_{epoch:04d}.npz",
            epoch=epoch,
            sample_count=config.sample_export_count,
            sample_seed=config.sample_seed,
            condition_mean=condition_mean,
            condition_std=condition_std,
            target_mean=target_mean,
            target_std=target_std,
            diffusion_buffers=diffusion_buffers,
            position_smoothing_kernel=sample_position_smoothing_kernel,
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
            condition_mean=condition_mean_np,
            condition_std=condition_std_np,
            target_mean=target_mean_np,
            target_std=target_std_np,
            pose_frames=pose_frames,
            pose_dim=pose_dim,
        )
        if record["val_noise_mse"] <= best_metric:
            best_metric = record["val_noise_mse"]
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
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
                target_mean=target_mean_np,
                target_std=target_std_np,
                pose_frames=pose_frames,
                pose_dim=pose_dim,
            )
        if record.get("val_x0_future_root_heading_error_deg", float("inf")) <= best_heading_metric:
            best_heading_metric = float(record["val_x0_future_root_heading_error_deg"])
            best_heading_epoch = epoch
            save_checkpoint(
                output_dir / "best_by_heading.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                condition_mean=condition_mean_np,
                condition_std=condition_std_np,
                target_mean=target_mean_np,
                target_std=target_std_np,
                pose_frames=pose_frames,
                pose_dim=pose_dim,
            )

    return {
        "best_epoch": int(best_epoch),
        "best_val_noise_mse": float(best_metric),
        "best_heading_epoch": int(best_heading_epoch),
        "best_val_x0_future_root_heading_error_deg": float(best_heading_metric),
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train direct pose-space DiT")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--train-window-index-csv", type=Path, default=None)
    parser.add_argument("--val-window-index-csv", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--pose-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--past-frames", type=int, default=120)
    parser.add_argument("--future-window-frames", type=int, default=40)
    parser.add_argument("--stride-frames", type=int, default=0)
    parser.add_argument("--train-stride-frames", type=int, default=20)
    parser.add_argument("--val-stride-frames", type=int, default=120)
    parser.add_argument("--condition-hidden-dim", type=int, default=128)
    parser.add_argument("--denoiser-num-blocks", type=int, default=4)
    parser.add_argument("--fusion-mode", type=str, default="add", choices=("add", "gated", "concat"))
    parser.add_argument("--dit-num-heads", type=int, default=8)
    parser.add_argument("--dit-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scheduler-type", type=str, default="none", choices=("none", "step", "cosine"))
    parser.add_argument("--scheduler-step-size", type=int, default=1)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--overfit-windows", type=int, default=0)
    parser.add_argument("--shuffle-train-windows", action="store_true")
    parser.add_argument("--shuffle-val-windows", action="store_true")
    parser.add_argument("--window-subset-seed", type=int, default=0)
    parser.add_argument("--sample-export-count", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-position-smoothing-kernel", type=str, default="tri5", choices=("none", "tri3", "tri5"))
    parser.add_argument("--future-heading-loss-weight", type=float, default=0.0)
    parser.add_argument("--transition-heading-loss-weight", type=float, default=0.0)
    parser.add_argument("--transition-heading-frames", type=int, default=10)
    parser.add_argument("--include-root-translation", action="store_true")
    return apply_data_config_to_args(parser.parse_args())


def main() -> None:
    args = parse_args()
    summary = run_dit_training(
        output_dir=args.output_dir,
        config=DiTTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            past_frames=args.past_frames,
            future_window_frames=args.future_window_frames,
            stride_frames=args.stride_frames,
            train_stride_frames=args.train_stride_frames,
            val_stride_frames=args.val_stride_frames,
            condition_hidden_dim=args.condition_hidden_dim,
            denoiser_num_blocks=args.denoiser_num_blocks,
            fusion_mode=args.fusion_mode,
            dit_num_heads=args.dit_num_heads,
            dit_mlp_ratio=args.dit_mlp_ratio,
            diffusion_steps=args.diffusion_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
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
            shuffle_train_windows=args.shuffle_train_windows,
            shuffle_val_windows=args.shuffle_val_windows,
            window_subset_seed=args.window_subset_seed,
            sample_export_count=args.sample_export_count,
            sample_seed=args.sample_seed,
            sample_position_smoothing_kernel=args.sample_position_smoothing_kernel,
            future_heading_loss_weight=args.future_heading_loss_weight,
            transition_heading_loss_weight=args.transition_heading_loss_weight,
            transition_heading_frames=args.transition_heading_frames,
            include_root_translation=args.include_root_translation,
        ),
        train_window_index_csv=args.train_window_index_csv,
        val_window_index_csv=args.val_window_index_csv,
        split_manifest_path=args.split_manifest,
        pose_root=args.pose_root,
        feature_root=args.feature_root,
        train_split=args.train_split,
        val_split=args.val_split,
    )
    print(summary)


if __name__ == "__main__":
    main()
