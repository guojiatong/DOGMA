from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Linear, nn.ConvTranspose1d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


@dataclass(frozen=True)
class MotionEncoderConfig:
    input_size: int
    hidden_size: int = 256
    embedding_size: int = 256
    projection_size: int = 128


class MotionEncoderBiGRUCo(nn.Module):
    """
    Adapted from the HumanML3D motion encoder, reduced to the motion-only part.
    It returns both a sequence embedding and an optional projection head output
    for contrastive training.
    """

    def __init__(self, config: MotionEncoderConfig):
        super().__init__()
        self.config = config
        self.input_emb = nn.Linear(config.input_size, config.hidden_size)
        self.gru = nn.GRU(
            config.hidden_size,
            config.hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.output_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config.hidden_size, config.embedding_size),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.LayerNorm(config.embedding_size),
            nn.GELU(),
            nn.Linear(config.embedding_size, config.projection_size),
        )
        self.hidden = nn.Parameter(torch.randn((2, 1, config.hidden_size), requires_grad=True))

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.projection_head.apply(init_weight)

    def encode(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        num_samples = inputs.shape[0]
        embedded = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu().tolist(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, last_hidden = self.gru(packed, hidden)
        last_hidden = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)
        return self.output_net(last_hidden)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(inputs, lengths)
        projection = self.projection_head(embedding)
        projection = torch.nn.functional.normalize(projection, dim=-1)
        return embedding, projection


class MotionConvEncoder(nn.Module):
    """
    Lightweight temporal convolutional encoder adapted from HumanML3D's movement
    encoder. This is useful as an alternative to the GRU encoder for ablations.
    """

    def __init__(self, config: MotionEncoderConfig):
        super().__init__()
        self.config = config
        self.main = nn.Sequential(
            nn.Conv1d(config.input_size, config.hidden_size, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(config.hidden_size, config.embedding_size, kernel_size=4, stride=2, padding=1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(config.embedding_size, config.embedding_size)
        self.projection_head = nn.Sequential(
            nn.Linear(config.embedding_size, config.embedding_size),
            nn.LayerNorm(config.embedding_size),
            nn.GELU(),
            nn.Linear(config.embedding_size, config.projection_size),
        )
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)
        self.projection_head.apply(init_weight)

    def encode(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        del lengths
        features = self.main(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        pooled = features.mean(dim=1)
        return self.out_net(pooled)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(inputs, lengths)
        projection = self.projection_head(embedding)
        projection = torch.nn.functional.normalize(projection, dim=-1)
        return embedding, projection
