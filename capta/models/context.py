"""CAPTA trend/innovation projection and client-local observed context."""
from __future__ import annotations

import weakref

import torch
from torch import nn
from torch.nn import functional as F

from .residual import build as _build_base

REGIME_NAMES = ("stable", "rise", "fall")
REGIME_THRESHOLD = 0.1
STD_FLOOR = 1e-4
PRIVATE_DIM = 128


def _speed(data: torch.Tensor) -> torch.Tensor:
    """Return the client's observed signal [B,N,T], never a future label."""
    if data.ndim != 4 or data.shape[1] < 2 or data.shape[2] < 1 or data.shape[3] < 1:
        raise ValueError("expected observed input [B,T>=2,N>=1,C>=1]")
    if not data.is_floating_point():
        raise TypeError("observed input must be floating point")
    return data[..., 0].transpose(1, 2)


def observed_state(data: torch.Tensor) -> torch.Tensor:
    """Observed-only [B,N,4]: mean, std, half-window change/std, roughness."""
    speed = _speed(data)
    mean = speed.mean(-1)
    std = speed.std(-1, unbiased=False)
    safe_std = std.clamp_min(STD_FLOOR)
    half = speed.shape[-1] // 2
    first, last = speed[..., :half], speed[..., half:]
    change = (last.mean(-1) - first.mean(-1)) / safe_std
    normalized = (speed - mean.unsqueeze(-1)) / safe_std.unsqueeze(-1)
    roughness = normalized.diff(dim=-1).abs().mean(-1)
    return torch.stack((mean, std, change, roughness), dim=-1)


def regime_weights(data: torch.Tensor) -> torch.Tensor:
    """Hard observed-history partition [B,N,3], in stable/rise/fall order.

    Boundaries +-0.1 belong to stable. Empty states/windows are skipped by the
    source objective; these groups are not ground-truth congestion classes.
    """
    change = observed_state(data)[..., 2]
    return torch.stack(
        (change.abs() <= REGIME_THRESHOLD,
         change > REGIME_THRESHOLD,
         change < -REGIME_THRESHOLD), dim=-1
    ).to(data.dtype)


class TrendInnovationProjection(nn.Module):
    """Two independently learned maps with identical initial history weights."""

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        with torch.random.fork_rng(devices=[]):
            self.innovation = nn.Linear(self.in_features, self.out_features, bias=False)
        self.innovation.to(device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            self.innovation.weight.copy_(self.weight)

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.shape[-1] != self.in_features:
            raise ValueError("history length does not match the existing projection")
        flat = normalized.reshape(-1, 1, self.in_features)
        trend = F.avg_pool1d(
            flat, kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        ).reshape_as(normalized)
        residual = normalized - trend
        return F.linear(trend, self.weight, self.bias) + self.innovation(residual)


class CandidateHook:
    """Fill existing private inputs with observed mean/std context, kept in RAM."""

    def __init__(self, model: nn.Module):
        existing = getattr(model, "_client_candidate_hook", None)
        if existing is not None and not existing.removed:
            raise ValueError("model already has an active context hook")
        self.recipe = "trend_innovation"
        self.context = None
        self.calls = 0
        self.removed = False
        self.keys = tuple(model.state_dict())
        self.handles = []
        self._model_ref = weakref.ref(model)
        self.handles.append(model.register_forward_pre_hook(self.observe))
        for block in model.blocks:
            self.handles.append(block.private.register_forward_pre_hook(self.inject))
        self.handles.append(model.register_forward_hook(self.clear, always_call=True))
        model._client_candidate_hook = self
        if tuple(model.state_dict()) != self.keys:
            raise AssertionError("installing hooks changed the model state keys")

    def observe(self, model: nn.Module, args: tuple) -> None:
        if not args:
            raise ValueError("pass observed data as the first positional argument")
        speed = _speed(args[0])
        mean = speed.mean(-1)
        std = speed.std(-1, unbiased=False)
        features = torch.stack((mean, std), -1)
        self.context = features.tanh().repeat_interleave(64, -1)
        self.calls += 1

    def inject(self, module: nn.Module, args: tuple) -> tuple:
        z = args[0]
        if self.context is None or z.shape[-1] != 2 * PRIVATE_DIM:
            raise RuntimeError("private context not observed or incompatible private width")
        if tuple(self.context.shape) != tuple(z[..., PRIVATE_DIM:].shape):
            raise RuntimeError("private context does not match this client/node batch")
        return (torch.cat((z[..., :PRIVATE_DIM], self.context), -1), *args[1:])

    def clear(self, model: nn.Module, args: tuple, output: torch.Tensor) -> None:
        self.context = None

    def remove(self, model: nn.Module | None = None) -> None:
        owner = self._model_ref()
        if model is not None and model is not owner:
            raise ValueError("hook removal target is not its owning model")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.context = None
        self.removed = True
        if owner is not None:
            if tuple(owner.state_dict()) != self.keys:
                raise AssertionError("hook lifecycle changed the model state keys")
            if getattr(owner, "_client_candidate_hook", None) is self:
                delattr(owner, "_client_candidate_hook")


def build_candidate(recipe: str, horizon: int = 6) -> nn.Module:
    """Build the CAPTA main model; retain its existing engine entry point."""
    if recipe != "trend_innovation":
        raise ValueError("only the CAPTA trend_innovation main model is supported")
    model = _build_base(horizon=horizon)
    model.history_embed[0] = TrendInnovationProjection(model.history_embed[0])
    CandidateHook(model)
    return model
