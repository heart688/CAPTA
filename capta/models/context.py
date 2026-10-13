"""Trend/innovation projection and client-local observed context."""
from __future__ import annotations
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from .residual import build as _build_base

RECIPES = ('local_context', 'local_context_packed4', 'evolution_context', 'trend_innovation', 'state_routing', 'spatial_deviation', 'regime_balance', 'regime_alignment')
REGIME_NAMES = ('stable', 'rise', 'fall')
REGIME_THRESHOLD = 0.1
STATE_SIGNATURE_SCALE = 0.1
STD_FLOOR = 1e-4
PRIVATE_DIM = 128

def _recipe_name(recipe: str) -> str:
    # The baseline alias always denotes the previously supported local context.
    name = "local_context" if recipe == "baseline" else recipe
    if name not in RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; expected one of {RECIPES}")
    return name


def _speed(data: torch.Tensor) -> torch.Tensor:
    """Return the client's observed signal [B,N,T], never a future label."""
    if data.ndim != 4 or data.shape[1] < 2 or data.shape[2] < 1 or data.shape[3] < 1:
        raise ValueError("expected observed input [B,T>=2,N>=1,C>=1]")
    if not data.is_floating_point():
        raise TypeError("observed input must be floating point")
    return data[..., 0].transpose(1, 2)


def _statistics(speed: torch.Tensor) -> tuple[torch.Tensor, ...]:
    mean = speed.mean(-1)
    std = speed.std(-1, unbiased=False)
    safe_std = std.clamp_min(STD_FLOOR)
    half = speed.shape[-1] // 2
    first, last = speed[..., :half], speed[..., half:]
    change = (last.mean(-1) - first.mean(-1)) / safe_std
    # Deliberately a signed std difference, not a new forecast statistic.
    volatility_change = last.std(-1, unbiased=False) - first.std(-1, unbiased=False)
    normalized = (speed - mean.unsqueeze(-1)) / safe_std.unsqueeze(-1)
    roughness = normalized.diff(dim=-1).abs().mean(-1)
    return mean, std, change, volatility_change, roughness, normalized


def observed_state(data: torch.Tensor) -> torch.Tensor:
    """Observed-only [B,N,4]: mean, std, half-window change/std, roughness.

    Mean/std are in the already client-normalized input's units.  Roughness is
    the mean absolute first difference of each instance-normalized history.
    This descriptor is RAM-only and is not a congestion label or causal state.
    """
    mean, std, change, _, roughness, _ = _statistics(_speed(data))
    return torch.stack((mean, std, change, roughness), dim=-1)


def regime_weights(data: torch.Tensor) -> torch.Tensor:
    """Nonnegative hard partition [B,N,3] in stable/rise/fall order.

    Boundaries +-0.1 belong to stable.  Each node-window belongs to exactly one
    group; an empty group's loss must be skipped by the runner.  Groups use
    observed histories only, and must not be described as congestion classes.
    """
    change = observed_state(data)[..., 2]
    return torch.stack((change.abs() <= REGIME_THRESHOLD,
                        change > REGIME_THRESHOLD,
                        change < -REGIME_THRESHOLD), dim=-1).to(data.dtype)


class TrendInnovationProjection(nn.Module):
    """Shared two-branch replacement of the existing history_embed[0].

    Old weight/bias names and their Parameter objects are retained EXACTLY.
    A new bias-free innovation.weight is initialized from the same old weight.
    With trend=avgpool3(x) and residual=x-trend, the initial output is
    W*trend+b + W*residual = W*x+b (up to floating-point summation order).
    The following original GELU/LayerNorm and all other common parameters stay
    in place.  Both branches subsequently learn independently.
    """

    def __init__(self, original: nn.Linear):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias
        # Preserve the caller's random stream as well as common initialization.
        with torch.random.fork_rng(devices=[]):
            self.innovation = nn.Linear(self.in_features, self.out_features, bias=False)
        self.innovation.to(device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            self.innovation.weight.copy_(self.weight)

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        if normalized.shape[-1] != self.in_features:
            raise ValueError("history length does not match the existing projection")
        flat = normalized.reshape(-1, 1, self.in_features)
        trend = F.avg_pool1d(flat, kernel_size=3, stride=1, padding=1,
                             count_include_pad=False).reshape_as(normalized)
        residual = normalized - trend
        return F.linear(trend, self.weight, self.bias) + self.innovation(residual)


class CandidateHook:
    """Client-local input hooks, with no new parameter/buffer/state_dict key.

    Input is positional model(data, profile), matching the inherited runners.
    The first private 128-vector z is passed through unchanged in every recipe.
    state_routing only changes a block's separate signature argument; it never
    substitutes routed states for private z.  Features are tanh-bounded and
    expanded to 128 by repeat_interleave(64) or repeat_interleave(32).
    """

    def __init__(self, model: nn.Module, recipe: str):
        existing = getattr(model, "_client_candidate_hook", None)
        if existing is not None and not existing.removed:
            raise ValueError("model already has an active CandidateHook")
        self.recipe = _recipe_name(recipe)
        self.context = None
        self.signature = None
        self.calls = 0
        self.removed = False
        self.keys = tuple(model.state_dict())
        self.handles = []
        self._model_ref = weakref.ref(model)
        if any(not getattr(block, "use_private", False) for block in model.blocks):
            raise ValueError("candidate requires the existing private residual blocks")
        self.handles.append(model.register_forward_pre_hook(self.observe))
        for block in model.blocks:
            self.handles.append(block.private.register_forward_pre_hook(self.inject))
            if self.recipe == "state_routing":
                self.handles.append(block.register_forward_pre_hook(self.inject_signature))
        self.handles.append(model.register_forward_hook(self.clear, always_call=True))
        # A plain object attachment: no tensors are registered on the model.
        model._client_candidate_hook = self
        if tuple(model.state_dict()) != self.keys:
            raise AssertionError("installing hooks changed the model state keys")

    def observe(self, model: nn.Module, args: tuple) -> None:
        if not args:
            raise ValueError("pass observed data as the first positional argument")
        speed = _speed(args[0])
        mean, std, change, vol_change, roughness, normalized = _statistics(speed)
        if self.recipe == "local_context_packed4":
            # Layout-matched C1/C4 control: no extra state information/parameters.
            features = torch.stack((mean, std, torch.zeros_like(mean),
                                    torch.zeros_like(std)), -1)
        elif self.recipe == "evolution_context":
            features = torch.stack((mean, std, change, vol_change), -1)
        elif self.recipe == "spatial_deviation":
            # Least-squares slope on a centered [-1,1] time coordinate, in
            # client-normalized signal units.  Cross-node pooling is entirely
            # local and permutation-equivariant; no N x N matrix is created.
            time = torch.linspace(-1, 1, speed.shape[-1], device=speed.device,
                                  dtype=speed.dtype)
            slope = (speed * time).sum(-1) / time.square().sum()
            slope_deviation = slope - slope.mean(1, keepdim=True)
            common_shape = normalized.mean(1, keepdim=True)
            shape_deviation = (normalized - common_shape).square().mean(-1).sqrt()
            features = torch.stack((mean, std, slope_deviation, shape_deviation), -1)
        else:
            # EXACTLY the supported mean/std-tanh repeated64 input recipe.
            features = torch.stack((mean, std), -1)
        self.context = features.tanh().repeat_interleave(PRIVATE_DIM // features.shape[-1], -1)
        self.signature = None
        if self.recipe == "state_routing":
            state = torch.stack((mean, std, change, roughness), -1)
            self.signature = STATE_SIGNATURE_SCALE * state.tanh().repeat_interleave(32, -1)
        self.calls += 1

    def inject(self, module: nn.Module, args: tuple) -> tuple:
        z = args[0]
        if self.context is None or z.shape[-1] != 2 * PRIVATE_DIM:
            raise RuntimeError("private context not observed or incompatible private width")
        if tuple(self.context.shape) != tuple(z[..., PRIVATE_DIM:].shape):
            raise RuntimeError("private context does not match this client/node batch")
        return (torch.cat((z[..., :PRIVATE_DIM], self.context), -1), *args[1:])

    def inject_signature(self, module: nn.Module, args: tuple) -> tuple:
        if self.signature is None:
            raise RuntimeError("state routing signature not observed")
        if len(args) > 2 or tuple(args[0].shape) != tuple(self.signature.shape):
            raise RuntimeError("unexpected block signature contract")
        previous = args[1] if len(args) == 2 else None
        signature = self.signature if previous is None else previous + self.signature
        return args[0], signature

    def clear(self, model: nn.Module, args: tuple, output: torch.Tensor) -> None:
        # Do not retain a completed batch or its graph in a long-lived client.
        self.context = None
        self.signature = None

    def remove(self, model: nn.Module | None = None) -> None:
        owner = self._model_ref()
        if model is not None and model is not owner:
            raise ValueError("hook removal target is not its owning model")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.context = None
        self.signature = None
        self.removed = True
        if owner is not None:
            if tuple(owner.state_dict()) != self.keys:
                raise AssertionError("hook lifecycle changed the model state keys")
            if getattr(owner, "_client_candidate_hook", None) is self:
                delattr(owner, "_client_candidate_hook")


def build_candidate(recipe: str, horizon: int = 6) -> nn.Module:
    """Build the unchanged residual base plus one prescribed intervention."""
    recipe = _recipe_name(recipe)
    model = _build_base("no_calendar_residual", horizon=horizon)
    if recipe == "trend_innovation":
        if not isinstance(model.history_embed, nn.Sequential) or not isinstance(model.history_embed[0], nn.Linear):
            raise TypeError("expected the original Sequential history_embed with Linear at index 0")
        model.history_embed[0] = TrendInnovationProjection(model.history_embed[0])
    CandidateHook(model, recipe)
    return model
