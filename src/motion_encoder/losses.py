from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Classic pairwise contrastive loss from Hadsell et al.
    label=0 means positive pair, label=1 means negative pair.
    """

    def __init__(self, margin: float = 3.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        return torch.mean(
            (1 - label) * torch.pow(distance, 2)
            + label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        )


class NTXentLoss(nn.Module):
    """
    A simple SimCLR-style loss. This is often a better default than classic
    pairwise contrastive loss when training an encoder for downstream FID-like
    feature extraction.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.shape[0]
        z = torch.cat([F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)], dim=0)
        logits = torch.matmul(z, z.t()) / self.temperature
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(mask, float("-inf"))
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        return F.cross_entropy(logits, labels)
