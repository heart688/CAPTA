"""Shared temporal encoder and latent-slot backbone for CAPTA."""
from __future__ import annotations

import math

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentSlotBlock(nn.Module):
    """Pool current sensor tokens into slots and retrieve their shared context.

    The builder attaches the client-private residual after initializing the
    complete shared backbone, preserving the released seeded initialization.
    """

    def __init__(self, dim: int, roles: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.assign = nn.Linear(dim, roles)
        self.value = nn.Linear(dim, dim)
        self.role_key = nn.Linear(dim, dim)
        self.role_value = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.role_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.spatial_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        assignment = torch.softmax(self.assign(z), -1)
        slots = torch.einsum("bnk,bnd->bkd", assignment, self.value(z))
        slots = slots / assignment.sum(1).clamp_min(1e-6).unsqueeze(-1)
        slots = slots + self.drop(self.role_mixer(slots))
        q = self.query(z)
        k = self.role_key(slots)
        v = self.role_value(slots)
        read = torch.softmax(
            torch.einsum("bnd,bkd->bnk", q, k) / math.sqrt(q.shape[-1]), -1
        )
        context = torch.einsum("bnk,bkd->bnd", read, v)
        gate = torch.sigmoid(self.spatial_scale)
        x = x + gate * self.drop(self.out(context))
        # The observed-context hook fills the second half before this map.
        private_input = torch.cat([z, torch.zeros_like(z)], -1)
        x = x + torch.sigmoid(self.private_scale) * self.private(private_input)
        return x + self.ffn(self.norm2(x))


class CAPTAForecaster(nn.Module):
    """Shared 12-step history encoder, three routing blocks and forecast head."""

    def __init__(self, horizon: int = 6):
        super().__init__()
        self.history = 12
        self.horizon = horizon
        dim, ff_dim, dropout = 128, 256, 0.1
        self.history_embed = nn.Sequential(
            nn.Linear(self.history, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        # The historical constructor initialized this disabled path before the
        # blocks. Consume those same draws so every retained weight and the
        # caller's random stream keep their original seeded values.
        discarded_calendar = nn.Sequential(
            nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        del discarded_calendar
        self.blocks = nn.ModuleList(
            [LatentSlotBlock(dim, 16, ff_dim, dropout) for _ in range(3)]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, horizon)

    def forward(
        self, data: torch.Tensor, profile: torch.Tensor | None = None
    ) -> torch.Tensor:
        # profile is accepted for the existing data-loader/engine call contract.
        speed = data[..., 0].transpose(1, 2)
        mean = speed.mean(-1, keepdim=True)
        std = speed.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized = (speed - mean) / std
        x = self.history_embed(normalized)
        # Preserve the main model's original addition of its disabled calendar
        # output without computing or retaining calendar features.
        x = x + x.new_zeros(x.shape[0], 1, x.shape[-1])
        for block in self.blocks:
            x = block(x)
        return self.head(self.final_norm(x)) * std + mean
