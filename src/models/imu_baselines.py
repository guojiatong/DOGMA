from __future__ import annotations

import torch
from torch import nn


class MaskedImuReconstructionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        target_dim: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_frames: int = 240,
        num_sensors: int = 10,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, target_dim)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.time_embedding = nn.Embedding(max_frames, d_model)
        self.sensor_embedding = nn.Embedding(num_sensors, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, frames, sensors, _ = feature.shape
        hidden = self.input_projection(feature)
        mask_token = self.mask_token.view(1, 1, 1, -1).expand(batch_size, frames, sensors, -1)
        hidden = torch.where(mask.unsqueeze(-1), mask_token, hidden)

        time_ids = torch.arange(frames, device=feature.device)
        sensor_ids = torch.arange(sensors, device=feature.device)
        hidden = hidden + self.time_embedding(time_ids)[None, :, None, :]
        hidden = hidden + self.sensor_embedding(sensor_ids)[None, None, :, :]

        hidden = hidden.reshape(batch_size, frames * sensors, -1)
        hidden = self.encoder(hidden)
        hidden = hidden.reshape(batch_size, frames, sensors, -1)
        return self.output_projection(hidden)


class ImuFuturePredictionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        target_dim: int = 12,
        d_model: int = 256,
        nhead: int = 8,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        dropout: float = 0.1,
        max_past_frames: int = 120,
        max_future_frames: int = 120,
        num_sensors: int = 10,
    ) -> None:
        super().__init__()
        self.max_future_frames = int(max_future_frames)
        self.num_sensors = int(num_sensors)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.output_projection = nn.Linear(d_model, target_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.past_time_embedding = nn.Embedding(max_past_frames, d_model)
        self.future_time_embedding = nn.Embedding(max_future_frames, d_model)
        self.sensor_embedding = nn.Embedding(num_sensors, d_model)
        self.future_query = nn.Parameter(torch.zeros(max_future_frames, num_sensors, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)

    def forward(self, past: torch.Tensor, future_frames: int | None = None) -> torch.Tensor:
        batch_size, past_frames, sensors, _ = past.shape
        future_frames = self.max_future_frames if future_frames is None else int(future_frames)
        if future_frames > self.max_future_frames:
            raise ValueError(f"future_frames={future_frames} exceeds max_future_frames={self.max_future_frames}")
        if sensors > self.num_sensors:
            raise ValueError(f"sensors={sensors} exceeds num_sensors={self.num_sensors}")

        past_hidden = self.input_projection(past)
        past_time_ids = torch.arange(past_frames, device=past.device)
        sensor_ids = torch.arange(sensors, device=past.device)
        past_hidden = past_hidden + self.past_time_embedding(past_time_ids)[None, :, None, :]
        past_hidden = past_hidden + self.sensor_embedding(sensor_ids)[None, None, :, :]
        memory = self.encoder(past_hidden.reshape(batch_size, past_frames * sensors, -1))

        future_query = self.future_query[:future_frames, :sensors]
        future_time_ids = torch.arange(future_frames, device=past.device)
        future_hidden = future_query[None].expand(batch_size, future_frames, sensors, -1)
        future_hidden = future_hidden + self.future_time_embedding(future_time_ids)[None, :, None, :]
        future_hidden = future_hidden + self.sensor_embedding(sensor_ids)[None, None, :, :]
        future_hidden = future_hidden.reshape(batch_size, future_frames * sensors, -1)
        decoded = self.decoder(tgt=future_hidden, memory=memory)
        decoded = decoded.reshape(batch_size, future_frames, sensors, -1)
        persistence = past[:, -1:, :, : self.output_projection.out_features].expand(
            batch_size,
            future_frames,
            sensors,
            self.output_projection.out_features,
        )
        return persistence + self.output_projection(decoded)
