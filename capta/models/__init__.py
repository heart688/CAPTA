"""CAPTA backbone construction."""
from .context import build_candidate, regime_weights

def build_model(horizon=6):
    return build_candidate("trend_innovation", horizon=horizon)
