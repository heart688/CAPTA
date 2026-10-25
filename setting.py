"""Fixed CAPTA experiment settings; no training happens on import."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CITIES = ("metr-la", "pems-bay", "chengdu", "shenzhen")


def load_settings(path=None):
    with Path(path or ROOT / "configs/default.json").open() as handle:
        cfg = json.load(handle)
    required = {
        "history": 12, "horizon": 6, "rounds": 20, "local_epochs": 2,
        "local_batch_size": 32, "local_lr": 0.001, "weight_decay": 0.0001,
        "calibration_epochs": 5, "target_epochs": 50, "target_batch_size": 8,
        "head_lr": 0.0004, "private_lr": 0.0002, "rest_lr": 0.0001,
        "beta": 0.6, "source_train_fraction": 0.8, "quant_bits": 32,
        "source_loss": "h2", "aggregation": "tra", "private_context_mode": "local",
        "backbone": "no_calendar_residual", "candidate": "trend_innovation",
        "support_start": 0, "support_end": 864, "support_steps": 864,
        "support_fit_end": 432, "support_gate_end": 648,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Fixed setting {key} must be {expected!r}")
    if cfg.get("source_window_cap") is not None:
        raise ValueError("Source training uses all legal windows")
    return cfg


def make_job(city="shenzhen", seed=0):
    if city not in CITIES or seed not in range(5):
        raise ValueError("Use a registered city and seed0–4")
    return {"job_id": f"{city}_seed{seed}_capta", "seed": seed, "city": city}


def describe_job(base, job):
    cfg = copy.deepcopy(base)
    cfg.update(target=job["city"], seed=job["seed"])
    return cfg
