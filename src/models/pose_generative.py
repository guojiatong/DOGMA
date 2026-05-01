from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class TemporalUpsampleBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upsampled = self.upsample(x)
        return self.activation(upsampled + self.block(upsampled))


class TemporalPoseVAE(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 36,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        window_frames: int = 240,
        dropout: float = 0.1,
        decoder_mode: str = "transpose_conv",
        logvar_min: float = -8.0,
        logvar_max: float = 8.0,
    ) -> None:
        super().__init__()
        if window_frames % 8 != 0:
            raise ValueError(f"window_frames must be divisible by 8, got {window_frames}")
        self.window_frames = int(window_frames)
        self.latent_frames = int(window_frames // 8)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.decoder_mode = str(decoder_mode)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.encoder = nn.Sequential(
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
        )
        self.mu_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.logvar_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        self.latent_projection = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        if self.decoder_mode == "transpose_conv":
            self.decoder = nn.Sequential(
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
            )
        elif self.decoder_mode == "upsample_conv":
            self.decoder = nn.Sequential(
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                TemporalUpsampleBlock(hidden_dim, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                TemporalUpsampleBlock(hidden_dim, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                TemporalUpsampleBlock(hidden_dim, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
            )
        else:
            raise ValueError(f"Unsupported decoder_mode: {self.decoder_mode}")
        self.output_projection = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def encode(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input_projection(pose.transpose(1, 2))
        hidden = self.encoder(hidden)
        mu = self.mu_projection(hidden).transpose(1, 2)
        logvar = self.logvar_projection(hidden).transpose(1, 2)
        logvar = torch.nan_to_num(
            logvar,
            nan=0.0,
            posinf=self.logvar_max,
            neginf=self.logvar_min,
        ).clamp_(min=self.logvar_min, max=self.logvar_max)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.latent_projection(latent.transpose(1, 2))
        hidden = self.decoder(hidden)
        reconstruction = self.output_projection(hidden).transpose(1, 2)
        if reconstruction.shape[1] != self.window_frames:
            raise ValueError(
                f"Decoded temporal length mismatch: expected {self.window_frames}, got {reconstruction.shape[1]}"
            )
        return reconstruction

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(pose)
        latent = self.reparameterize(mu, logvar)
        reconstruction = self.decode(latent)
        return reconstruction, mu, logvar


class ConditionalFuturePoseVAE(nn.Module):
    def __init__(
        self,
        *,
        pose_dim: int = 36,
        condition_dim: int = 130,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        future_frames: int = 80,
        dropout: float = 0.1,
        decoder_mode: str = "transpose_conv",
        use_condition_skip_decoder: bool = False,
        use_split_pose_heads: bool = False,
    ) -> None:
        super().__init__()
        if future_frames % 8 != 0:
            raise ValueError(f"future_frames must be divisible by 8, got {future_frames}")
        self.future_frames = int(future_frames)
        self.latent_frames = int(future_frames // 8)
        self.pose_dim = int(pose_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.decoder_mode = str(decoder_mode)
        self.use_condition_skip_decoder = bool(use_condition_skip_decoder)
        self.use_split_pose_heads = bool(use_split_pose_heads)

        self.pose_input_projection = nn.Conv1d(pose_dim, hidden_dim, kernel_size=1)
        self.pose_encoder = nn.Sequential(
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
        )

        self.condition_projection = nn.Conv1d(condition_dim, hidden_dim, kernel_size=1)
        self.condition_encoder = nn.Sequential(
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
        )
        self.condition_pool = nn.AdaptiveAvgPool1d(self.latent_frames)
        self.condition_skip20_pool = nn.AdaptiveAvgPool1d(self.future_frames // 4)
        self.condition_skip40_pool = nn.AdaptiveAvgPool1d(self.future_frames // 2)
        self.condition_skip80_pool = nn.AdaptiveAvgPool1d(self.future_frames)

        self.posterior_projection = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.posterior_mu_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.posterior_logvar_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        self.prior_projection = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        self.prior_mu_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.prior_logvar_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        self.latent_projection = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        if self.decoder_mode == "transpose_conv":
            self.decoder = nn.Sequential(
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
            )
        elif self.decoder_mode == "upsample_conv":
            self.decoder = nn.Sequential(
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                TemporalUpsampleBlock(hidden_dim, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                TemporalUpsampleBlock(hidden_dim, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
                TemporalUpsampleBlock(hidden_dim, dropout=dropout),
                ResidualTemporalBlock(hidden_dim, dropout=dropout),
            )
        else:
            raise ValueError(f"Unsupported decoder_mode: {self.decoder_mode}")
        self.output_projection = nn.Conv1d(hidden_dim, pose_dim, kernel_size=1)
        self.condition_skip20_projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.condition_skip40_projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.condition_skip80_projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.position_refinement_projection = nn.Conv1d(hidden_dim, pose_dim - 6, kernel_size=1)
        self.heading_refinement_projection = nn.Conv1d(hidden_dim, 6, kernel_size=1)
        for module in (
            self.condition_skip20_projection,
            self.condition_skip40_projection,
            self.condition_skip80_projection,
            self.position_refinement_projection,
            self.heading_refinement_projection,
        ):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode_condition(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.condition_projection(condition.transpose(1, 2))
        hidden = self.condition_encoder[0](hidden)
        hidden = self.condition_encoder[1](hidden)
        hidden = self.condition_encoder[2](hidden)
        skip80 = hidden
        hidden = self.condition_encoder[3](hidden)
        hidden = self.condition_encoder[4](hidden)
        skip40 = hidden
        hidden = self.condition_encoder[5](hidden)
        hidden = self.condition_encoder[6](hidden)
        skip20 = hidden

        latent = self.condition_pool(hidden)
        if latent.shape[-1] != self.latent_frames:
            raise ValueError(
                f"Condition encoder temporal length mismatch: expected {self.latent_frames}, got {latent.shape[-1]}"
            )
        return {
            "latent": latent,
            "skip20": self.condition_skip20_pool(skip20),
            "skip40": self.condition_skip40_pool(skip40),
            "skip80": self.condition_skip80_pool(skip80),
        }

    def encode_posterior(
        self,
        future_pose: torch.Tensor,
        condition_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pose_hidden = self.pose_input_projection(future_pose.transpose(1, 2))
        pose_hidden = self.pose_encoder(pose_hidden)
        if pose_hidden.shape[-1] != self.latent_frames:
            raise ValueError(
                f"Pose encoder temporal length mismatch: expected {self.latent_frames}, got {pose_hidden.shape[-1]}"
            )
        posterior_hidden = self.posterior_projection(torch.cat([pose_hidden, condition_hidden], dim=1))
        posterior_mu = self.posterior_mu_projection(posterior_hidden).transpose(1, 2)
        posterior_logvar = self.posterior_logvar_projection(posterior_hidden).transpose(1, 2)
        return posterior_mu, posterior_logvar

    def encode_prior(self, condition_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior_hidden = self.prior_projection(condition_hidden)
        prior_mu = self.prior_mu_projection(prior_hidden).transpose(1, 2)
        prior_logvar = self.prior_logvar_projection(prior_hidden).transpose(1, 2)
        return prior_mu, prior_logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent: torch.Tensor, condition_features: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.latent_projection(latent.transpose(1, 2)) + condition_features["latent"]
        hidden = self.decoder[0](hidden)
        hidden = self.decoder[1](hidden)
        if self.use_condition_skip_decoder:
            hidden = hidden + self.condition_skip20_projection(condition_features["skip20"])
        hidden = self.decoder[2](hidden)
        hidden = self.decoder[3](hidden)
        if self.use_condition_skip_decoder:
            hidden = hidden + self.condition_skip40_projection(condition_features["skip40"])
        hidden = self.decoder[4](hidden)
        hidden = self.decoder[5](hidden)
        if self.use_condition_skip_decoder:
            hidden = hidden + self.condition_skip80_projection(condition_features["skip80"])
        hidden = self.decoder[6](hidden)

        reconstruction = self.output_projection(hidden)
        if self.use_split_pose_heads:
            reconstruction = reconstruction + torch.cat(
                [
                    self.position_refinement_projection(hidden),
                    self.heading_refinement_projection(hidden),
                ],
                dim=1,
            )
        reconstruction = reconstruction.transpose(1, 2)
        if reconstruction.shape[1] != self.future_frames:
            raise ValueError(
                f"Decoded temporal length mismatch: expected {self.future_frames}, got {reconstruction.shape[1]}"
            )
        return reconstruction

    def forward(
        self,
        condition: torch.Tensor,
        future_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        condition_features = self.encode_condition(condition)
        posterior_mu, posterior_logvar = self.encode_posterior(future_pose, condition_features["latent"])
        prior_mu, prior_logvar = self.encode_prior(condition_features["latent"])
        latent = self.reparameterize(posterior_mu, posterior_logvar)
        reconstruction = self.decode(latent, condition_features)
        return reconstruction, posterior_mu, posterior_logvar, prior_mu, prior_logvar

    def predict(
        self,
        condition: torch.Tensor,
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition_features = self.encode_condition(condition)
        prior_mu, prior_logvar = self.encode_prior(condition_features["latent"])
        if sample:
            std = torch.exp(0.5 * prior_logvar)
            noise = torch.randn(
                prior_mu.shape,
                device=prior_mu.device,
                dtype=prior_mu.dtype,
                generator=generator,
            )
            latent = prior_mu + std * noise
        else:
            latent = prior_mu
        return self.decode(latent, condition_features)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape [B], got {timesteps.shape}")
        half_dim = self.dim // 2
        if half_dim == 0:
            return torch.zeros((timesteps.shape[0], 0), dtype=torch.float32, device=timesteps.device)
        exponent = torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        scale = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(exponent * scale)
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding


class ConditionalLatentDenoiser(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int = 64,
        condition_dim: int = 130,
        hidden_dim: int = 128,
        latent_frames: int = 30,
        dropout: float = 0.1,
        denoiser_num_blocks: int = 4,
        fusion_mode: str = "add",
    ) -> None:
        super().__init__()
        if latent_frames <= 0:
            raise ValueError(f"latent_frames must be positive, got {latent_frames}")
        if denoiser_num_blocks <= 0:
            raise ValueError(f"denoiser_num_blocks must be positive, got {denoiser_num_blocks}")
        if fusion_mode not in {"add", "gated", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_frames = int(latent_frames)
        self.denoiser_num_blocks = int(denoiser_num_blocks)
        self.fusion_mode = str(fusion_mode)

        self.latent_projection = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.condition_projection = nn.Conv1d(condition_dim, hidden_dim, kernel_size=1)
        self.condition_encoder = nn.Sequential(
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            ResidualTemporalBlock(hidden_dim, dropout=dropout),
        )
        self.condition_pool = nn.AdaptiveAvgPool1d(self.latent_frames)
        self.time_embedding = SinusoidalTimeEmbedding(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        if self.fusion_mode == "gated":
            self.fusion_projection = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        elif self.fusion_mode == "concat":
            self.fusion_projection = nn.Sequential(
                nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            )
        else:
            self.fusion_projection = None
        self.denoiser = nn.Sequential(
            *[ResidualTemporalBlock(hidden_dim, dropout=dropout) for _ in range(self.denoiser_num_blocks)]
        )
        self.output_projection = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def _pool_condition_hidden(self, condition_hidden: torch.Tensor) -> torch.Tensor:
        if condition_hidden.device.type == "mps" and condition_hidden.shape[-1] % self.latent_frames != 0:
            return F.interpolate(
                condition_hidden,
                size=self.latent_frames,
                mode="linear",
                align_corners=False,
            )
        return self.condition_pool(condition_hidden)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 3:
            raise ValueError(f"noisy_latent must have shape [B,T,C], got {noisy_latent.shape}")
        if condition.ndim != 3:
            raise ValueError(f"condition must have shape [B,T,C], got {condition.shape}")
        if noisy_latent.shape[1] != self.latent_frames:
            raise ValueError(
                f"noisy_latent second dimension must be {self.latent_frames}, got {noisy_latent.shape[1]}"
            )

        latent_hidden = self.latent_projection(noisy_latent.transpose(1, 2))
        condition_hidden = self.condition_projection(condition.transpose(1, 2))
        condition_hidden = self.condition_encoder(condition_hidden)
        condition_hidden = self._pool_condition_hidden(condition_hidden)
        if condition_hidden.shape[-1] != self.latent_frames:
            raise ValueError(
                f"Condition encoder temporal length mismatch: expected {self.latent_frames}, got {condition_hidden.shape[-1]}"
            )

        time_hidden = self.time_mlp(self.time_embedding(timesteps)).unsqueeze(-1)
        if self.fusion_mode == "add":
            hidden = latent_hidden + condition_hidden + time_hidden
        elif self.fusion_mode == "gated":
            gate = torch.sigmoid(self.fusion_projection(torch.cat([latent_hidden, condition_hidden], dim=1)))
            hidden = latent_hidden + gate * condition_hidden + time_hidden
        else:
            time_expanded = time_hidden.expand(-1, -1, self.latent_frames)
            hidden = self.fusion_projection(torch.cat([latent_hidden, condition_hidden, time_expanded], dim=1))
        hidden = self.denoiser(hidden)
        return self.output_projection(hidden).transpose(1, 2)
