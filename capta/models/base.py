"""Variable-token backbone building blocks."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class IndependentBlock(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class DenseVariableBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        update, _ = self.attn(z, z, z, need_weights=False)
        x = x + self.drop(update)
        return x + self.ffn(self.norm2(x))


class GraphVariableBlock(nn.Module):
    """Topology-bound spatial control with transferable shared weights."""

    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.message = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: torch.Tensor) -> torch.Tensor:
        if graph.ndim != 2 or graph.shape[0] != x.shape[1] or graph.shape[1] != x.shape[1]:
            raise ValueError(f"graph shape {tuple(graph.shape)} does not match {x.shape[1]} nodes")
        update = torch.einsum("nm,bmd->bnd", graph, self.norm1(x))
        x = x + self.drop(self.message(update))
        return x + self.ffn(self.norm2(x))


class DynamicLatentRouter(nn.Module):
    """Aggregate sensor histories into episode-conditioned states and read back.

    Assignments and reads are conditioned on the current history token.  The
    number of roles is fixed across cities, while the number and identity of
    sensors may change.  No adjacency or N x N interaction matrix is formed.
    """

    def __init__(self, dim: int, roles: int, ff_dim: int, dropout: float,
                 center_routing: bool = False, residual_gate: str = "scalar",
                 assignment_topk: int | None = None):
        super().__init__()
        self.roles = roles
        self.center_routing = center_routing
        self.residual_gate = residual_gate
        self.assignment_topk = assignment_topk
        if assignment_topk is not None and not 1 <= assignment_topk <= roles:
            raise ValueError(f"assignment_topk must be in [1, {roles}]")
        if residual_gate not in {"scalar", "conditional"}:
            raise ValueError(f"unknown residual gate: {residual_gate}")
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
        if residual_gate == "scalar":
            self.spatial_scale = nn.Parameter(torch.tensor(-2.0))
        else:
            self.gate = nn.Linear(3 * dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        routed = z - z.mean(dim=1, keepdim=True) if self.center_routing else z
        logits = self.assign(routed)
        if self.assignment_topk is not None and self.assignment_topk < self.roles:
            top_values, top_indices = torch.topk(logits, self.assignment_topk, dim=-1)
            sparse_logits = torch.full_like(logits, -torch.inf)
            logits = sparse_logits.scatter(-1, top_indices, top_values)
        assignment = torch.softmax(logits, dim=-1)  # [B,N,K]
        values = self.value(routed)
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        slots = torch.einsum("bnk,bnd->bkd", assignment, values) / mass.unsqueeze(-1)
        slots = slots + self.drop(self.role_mixer(slots))

        q = self.query(routed)
        k = self.role_key(slots)
        v = self.role_value(slots)
        read = torch.softmax(torch.einsum("bnd,bkd->bnk", q, k) / math.sqrt(q.shape[-1]), dim=-1)
        context = torch.einsum("bnk,bkd->bnd", read, v)
        if self.residual_gate == "scalar":
            gate = torch.sigmoid(self.spatial_scale)
        else:
            gate = torch.sigmoid(self.gate(torch.cat([z, context, z - context], dim=-1)))
        x = x + gate * self.drop(self.out(context))
        return x + self.ffn(self.norm2(x))


class PersistentPrototypeRouter(nn.Module):
    """Graph-free control with persistent source-trained role prototypes.

    Unlike :class:`DynamicLatentRouter`, the role keys are model parameters and
    therefore carry source-city state directly into the target city.  The
    episode values still use assignment-mass normalization so city node counts
    do not change their scale.  This is an internal TAPR-like control, not a
    claim of reproducing the published TAPSTAR implementation.
    """

    def __init__(self, dim: int, roles: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.prototypes = nn.Parameter(torch.empty(roles, dim))
        nn.init.normal_(self.prototypes, std=dim ** -0.5)
        self.value = nn.Linear(dim, dim)
        self.role_mixer = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, dim),
        )
        self.query = nn.Linear(dim, dim)
        self.role_value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.spatial_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm1(x)
        prototypes = F.normalize(self.prototypes, dim=-1)
        assignment = torch.softmax(
            torch.einsum("bnd,kd->bnk", F.normalize(z, dim=-1), prototypes)
            * math.sqrt(z.shape[-1]),
            dim=-1,
        )
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        slots = torch.einsum("bnk,bnd->bkd", assignment, self.value(z))
        slots = slots / mass.unsqueeze(-1) + self.prototypes.unsqueeze(0)
        slots = slots + self.drop(self.role_mixer(slots))
        read = torch.softmax(
            torch.einsum("bnd,kd->bnk", self.query(z), prototypes)
            / math.sqrt(z.shape[-1]),
            dim=-1,
        )
        context = torch.einsum("bnk,bkd->bnd", read, self.role_value(slots))
        x = x + torch.sigmoid(self.spatial_scale) * self.drop(self.out(context))
        return x + self.ffn(self.norm2(x))


class StrongTransferForecaster(nn.Module):
    def __init__(
        self,
        variant: str,
        history: int = 12,
        horizon: int = 12,
        dim: int = 128,
        heads: int = 4,
        ff_dim: int = 256,
        layers: int = 3,
        roles: int = 16,
        dropout: float = 0.1,
        profile_features: int = 20,
        use_profile: bool = True,
        center_routing: bool = False,
        residual_gate: str = "scalar",
        assignment_topk: int | None = None,
    ):
        super().__init__()
        if variant not in {"independent", "graph", "dense", "tapr_like", "dynamic"}:
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.history = history
        self.horizon = horizon
        self.use_profile = use_profile
        self.register_buffer("graph", torch.empty(0), persistent=False)

        # Entire normalized history becomes one variable token (iTransformer).
        self.history_embed = nn.Sequential(
            nn.Linear(history, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        # Calendar context is city-independent and shared by all nodes.
        self.calendar_embed = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        if use_profile:
            self.profile_embed = nn.Sequential(
                nn.LayerNorm(profile_features),
                nn.Linear(profile_features, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.profile_scale = nn.Parameter(torch.tensor(-2.0))

        blocks = []
        for _ in range(layers):
            if variant == "independent":
                blocks.append(IndependentBlock(dim, ff_dim, dropout))
            elif variant == "graph":
                blocks.append(GraphVariableBlock(dim, ff_dim, dropout))
            elif variant == "dense":
                blocks.append(DenseVariableBlock(dim, heads, ff_dim, dropout))
            elif variant == "tapr_like":
                blocks.append(PersistentPrototypeRouter(dim, roles, ff_dim, dropout))
            else:
                blocks.append(DynamicLatentRouter(
                    dim, roles, ff_dim, dropout, center_routing, residual_gate, assignment_topk
                ))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, horizon)

    def set_graph(self, graph: torch.Tensor) -> None:
        """Swap the city graph without changing learned parameters."""
        self.graph = graph

    def forward(self, data: torch.Tensor, profile: torch.Tensor | None = None) -> torch.Tensor:
        # data: [B,T,N,3], output: [B,N,H]
        speed = data[..., 0].transpose(1, 2)  # [B,N,T]
        mean = speed.mean(dim=-1, keepdim=True)
        std = speed.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized = (speed - mean) / std
        x = self.history_embed(normalized)

        # Calendar channels are shared by sensors in the benchmark data.  Use a
        # permutation-invariant reduction instead of privileging node zero so
        # the model contract remains valid under arbitrary sensor ordering.
        last_tod = data[:, -1, :, 1].mean(dim=1)
        last_dow = data[:, -1, :, 2].mean(dim=1)
        calendar = torch.stack(
            [
                torch.sin(2 * math.pi * last_tod),
                torch.cos(2 * math.pi * last_tod),
                torch.sin(2 * math.pi * last_dow),
                torch.cos(2 * math.pi * last_dow),
            ],
            dim=-1,
        )
        x = x + self.calendar_embed(calendar).unsqueeze(1)
        if self.use_profile:
            if profile is None:
                raise ValueError("profile is required when use_profile=True")
            x = x + torch.sigmoid(self.profile_scale) * self.profile_embed(profile)

        for block in self.blocks:
            if self.variant == "graph":
                if self.graph.numel() == 0:
                    raise ValueError("graph variant requires set_graph() before forward")
                x = block(x, self.graph)
            else:
                x = block(x)
        prediction = self.head(self.final_norm(x))
        return prediction * std + mean
