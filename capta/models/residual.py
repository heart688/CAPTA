"""Client-private residual construction and target adaptation policy."""
import torch
from torch import nn

from .base import CAPTAForecaster


def build(horizon: int = 6) -> nn.Module:
    model = CAPTAForecaster(horizon=horizon)
    for index, block in enumerate(model.blocks):
        # All shared parameters are initialized before any private network.
        # Keep private initialization independent of the shared random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(torch.initial_seed() + 1000 + index)
            block.private = nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
            )
            block.private_scale = nn.Parameter(torch.tensor(-2.0))
    return model


def adapt_policy(model: nn.Module, candidate: str = "no_calendar_residual") -> None:
    """Freeze shared routing while fitting temporal, private and head weights."""
    if candidate != "no_calendar_residual":
        raise ValueError("only the CAPTA main-model adaptation policy is supported")
    frozen = {
        "assign", "value", "role_key", "role_value", "query", "role_mixer",
        "out", "spatial_scale",
    }
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not bool(set(name.split(".")) & frozen)
