"""Purged chronological windows and training-range-only normalization.

CAPTA does not consume adjacency; series hashes remain mandatory.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class Range:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid half-open range [{self.start}, {self.end})")


@dataclass
class CityData:
    name: str
    features: np.ndarray  # [T,N,3]: speed, time-of-day, day-of-week
    graph_path: Path
    steps_per_day: int
    interval_minutes: int


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _day_of_week(raw: np.ndarray, spec: dict) -> np.ndarray:
    steps = int(spec["steps_per_day"])
    channel = spec.get("time_index_channel")
    if channel is None:
        day = (np.arange(len(raw)) // steps) % 7
    else:
        time_index = raw[:, 0, int(channel)]
        week_steps = 2016 if np.max(time_index) >= 1500 else 1008 if np.max(time_index) >= 900 else steps * 7
        day = (time_index.astype(np.int64) // (week_steps // 7)) % 7
    return day.astype(np.float32) / 7.0


def load_city(
    manifest: dict,
    city: str,
    verify_hash: bool | None = None,
    max_steps: int | None = None,
) -> CityData:
    root = Path(manifest["root"])
    spec = manifest["cities"][city]
    series_path = root / spec["series_file"]
    graph_path = root / spec.get("graph_file", "unused_graph.npy")
    raw = np.load(series_path, mmap_mode="r")
    expected = tuple(spec["expected_shape"])
    if tuple(raw.shape) != expected:
        raise ValueError(f"{city} shape {tuple(raw.shape)} != manifest {expected}")
    strict = manifest.get("strict_hash", False) if verify_hash is None else verify_hash
    if strict:
        for path, key in ((series_path, "series_sha256"),):
            expected_hash = spec.get(key)
            if not expected_hash or sha256_file(path) != expected_hash:
                raise ValueError(f"{city} {key} missing or mismatched")
    stop = len(raw) if max_steps is None else int(max_steps)
    if stop <= 0 or stop > len(raw):
        raise ValueError(f"invalid max_steps={stop} for {city} length={len(raw)}")
    visible = raw[:stop]
    speed = np.asarray(visible[:, :, int(spec["speed_channel"])], dtype=np.float32)
    tod = np.asarray(visible[:, :, int(spec["time_of_day_channel"])], dtype=np.float32)
    dow = np.repeat(_day_of_week(visible, spec)[:, None], speed.shape[1], axis=1)
    features = np.stack((speed, tod, dow), axis=-1)
    if not np.isfinite(features).all():
        raise ValueError(f"{city} contains non-finite features")
    return CityData(city, features, graph_path, int(spec["steps_per_day"]), int(spec["interval_minutes"]))


def window_starts_inside(bounds: Range, history: int, horizon: int) -> np.ndarray:
    last = bounds.end - history - horizon
    if last < bounds.start:
        return np.empty(0, dtype=np.int64)
    return np.arange(bounds.start, last + 1, dtype=np.int64)


def fit_speed_stats(features: np.ndarray, bounds: Range) -> tuple[float, float]:
    speed = features[bounds.start:bounds.end, :, 0]
    return float(speed.mean()), max(float(speed.std()), 1e-6)


def node_profile(features: np.ndarray, bounds: Range, mean: float, std: float) -> np.ndarray:
    speed = (features[bounds.start:bounds.end, :, 0] - mean) / std
    tod = features[bounds.start:bounds.end, 0, 1]
    phase = np.minimum((tod * 12).astype(np.int64), 11)
    daily = np.zeros((speed.shape[1], 12), dtype=np.float32)
    for index in range(12):
        selected = speed[phase == index]
        if len(selected):
            daily[:, index] = selected.mean(axis=0)
    node_mean = speed.mean(axis=0)
    node_std = np.maximum(speed.std(axis=0), 1e-5)
    daily = (daily - node_mean[:, None]) / node_std[:, None]
    quantiles = np.quantile(speed, (0.1, 0.5, 0.9), axis=0).T.astype(np.float32)
    volatility = np.abs(np.diff(speed, axis=0)).mean(axis=0)[:, None]
    correlations = []
    centered = speed - node_mean[None]
    for lag in (1, 3, 6):
        numerator = (centered[lag:] * centered[:-lag]).mean(axis=0)
        correlations.append((numerator / (node_std**2 + 1e-6))[:, None])
    return np.concatenate([daily, node_mean[:, None], node_std[:, None], quantiles, volatility, *correlations], axis=1).astype(np.float32)


class TrafficWindows(Dataset):
    def __init__(
        self,
        city: CityData,
        starts: np.ndarray,
        history: int,
        horizon: int,
        stats_range: Range | None,
    ):
        self.city, self.starts, self.history, self.horizon = city, np.asarray(starts), history, horizon
        if stats_range is None:
            # Strict zero-shot mode: the model receives raw traffic values and a
            # constant profile, so no target value outside each input history is
            # used for preprocessing or conditioning.
            self.mean, self.std = 0.0, 1.0
            self.profile = np.zeros((city.features.shape[1], 21), dtype=np.float32)
        else:
            self.mean, self.std = fit_speed_stats(city.features, stats_range)
            self.profile = node_profile(city.features, stats_range, self.mean, self.std)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, item: int):
        start = int(self.starts[item]); middle = start + self.history; end = middle + self.horizon
        x = self.city.features[start:middle].copy()
        x[..., 0] = (x[..., 0] - self.mean) / self.std
        y = ((self.city.features[middle:end, :, 0] - self.mean) / self.std).T.copy()
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(self.profile)
