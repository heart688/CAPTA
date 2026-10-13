"""Fixed CAPTA experiment settings; no training happens on import."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CITIES = ("metr-la", "pems-bay", "chengdu", "shenzhen")
SOURCE_LOSSES = ("ordinary", "c5", "h2", "random_c5", "pseudo_h2")


def load_settings(path=None):
    with Path(path or ROOT / "configs/default.json").open() as handle:
        cfg = json.load(handle)
    required = {
        "history": 12, "horizon": 6, "rounds": 20, "local_epochs": 2,
        "local_batch_size": 32, "local_lr": 0.001, "weight_decay": 0.0001,
        "calibration_epochs": 5, "target_epochs": 50, "target_batch_size": 8,
        "head_lr": 0.0004, "private_lr": 0.0002, "rest_lr": 0.0001,
        "beta": 0.6, "source_train_fraction": 0.8, "quant_bits": 32,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Fixed setting {key} must be {expected!r}")
    if cfg.get("source_window_cap") is not None:
        raise ValueError("Source training uses all legal windows")
    return cfg


def make_job(city="shenzhen", seed=0, support="full", source_loss="h2",
             aggregation="tra", context="local"):
    if city not in CITIES or seed not in range(5):
        raise ValueError("Use a registered city and seed0–4")
    if support not in ("full", "half") or source_loss not in SOURCE_LOSSES:
        raise ValueError("Unknown support budget or source objective")
    if aggregation not in ("tra", "fedavg", "off") or context not in ("local", "zero"):
        raise ValueError("Unknown aggregation/context control")
    return {"job_id": f"{city}_seed{seed}_{support}_{source_loss}_{aggregation}_{context}",
            "stage": "public_reproduction", "source_mode": source_loss,
            "aggregation": aggregation, "seed": seed, "city": city,
            "support_fraction": 1.0 if support == "full" else 0.5,
            "context": context}


def describe_job(base, job):
    cfg = copy.deepcopy(base)
    full = job["support_fraction"] == 1.0
    cfg.update(target=job["city"], seed=job["seed"], source_loss=job["source_mode"],
               aggregation=job["aggregation"], private_context_mode=job["context"],
               support_start=0 if full else 432, support_end=864,
               support_fit_end=432 if full else 648,
               support_gate_end=648 if full else 756,
               support_steps=864 if full else 432)
    return cfg
