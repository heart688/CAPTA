"""CAPTA main-model construction and observed-history traffic states."""
from .context import build_candidate, observed_state, regime_weights


def build_model(horizon=6):
    return build_candidate("trend_innovation", horizon=horizon)
