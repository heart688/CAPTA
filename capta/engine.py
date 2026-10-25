"""Main CAPTA federated training, target calibration, adaptation and evaluation."""
from __future__ import annotations
import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from . import support as original
from . import utils as state_tools
from .models.context import build_candidate, regime_weights
from .data import CityData, Range, TrafficWindows, load_city, window_starts_inside
ROOT = Path(__file__).resolve().parents[1]
DATASET_MANIFEST = ROOT / 'configs/dataset.json'
DATA_ROOT = None
CITIES = original.CITIES
STATES = ('stable', 'rise', 'fall')
SOURCE_MODES = ('ordinary', 'h2')
JOB_FIELDS = ('job_id', 'seed', 'city')
sha, save, log = original.sha, original.save, original.log
tree_sha, capture_rng, restore_rng = state_tools.tree_sha, state_tools.capture_rng, state_tools.restore_rng
exact_equal = state_tools.exact_equal


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def private_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not original.is_shared(k, "full_private")}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def identity(config_path, job):
    paths = sorted((ROOT / 'capta').rglob('*.py')) + [ROOT / 'main.py', Path(config_path), DATASET_MANIFEST]
    hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name: sha(p) for p in paths}
    hashes['scientific_job'] = json_digest({key: job[key] for key in JOB_FIELDS})
    return hashes


def job_config(base, job):
    assert all(key in job for key in JOB_FIELDS)
    assert job["city"] in CITIES and job["seed"] in range(5)
    cfg = copy.deepcopy(base)
    cfg.update(seed=job["seed"], target=job["city"], scientific_job={k: job[k] for k in JOB_FIELDS})
    cfg["support_start"] = 0
    cfg["support_end"] = 864
    cfg["support_steps"] = 864
    cfg["support_fit_end"] = 432
    cfg["support_gate_end"] = 648
    validate_config(cfg)
    return cfg


def validate_config(cfg, synthetic=False):
    assert cfg.get("candidate", "trend_innovation") == "trend_innovation"
    assert cfg["backbone"] == "no_calendar_residual" and (cfg["history"], cfg["horizon"]) == (12, 6)
    assert cfg["source_loss"] == "h2" and cfg["aggregation"] == "tra"
    assert cfg["private_context_mode"] == "local"
    assert cfg["seed"] in range(5) and cfg["target"] in CITIES
    assert cfg["local_lr"] == .001 and cfg["weight_decay"] == .0001 and cfg["clip_grad_norm"] == 5.
    assert cfg["local_batch_size"] == 32 and cfg["local_epochs"] == 2
    assert cfg["quant_bits"] == 32 and cfg["beta"] == .6 and cfg["source_window_cap"] is None
    assert cfg["source_train_fraction"] == .8 and cfg["target_batch_size"] == 8
    assert (cfg["head_lr"], cfg["private_lr"], cfg["rest_lr"]) == (.0004, .0002, .0001)
    assert cfg["local_response_scale"] == .3
    if not synthetic:
        assert (cfg["rounds"], cfg["calibration_epochs"], cfg["target_epochs"]) == (20, 5, 50)
        assert (cfg["support_start"], cfg["support_fit_end"], cfg["support_gate_end"], cfg["support_end"]) == (0, 432, 648, 864)


def build_model(cfg, device="cuda"):
    model = build_candidate("trend_innovation", horizon=cfg["horizon"]).to(device)
    hook = model._client_candidate_hook
    assert hook.recipe == "trend_innovation"
    return model


class Indexed(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return (*self.data[index], index)


def _numpy_rng_safe(rng):
    """Self-saved checkpoint can be loaded with weights_only=True."""
    result = copy.deepcopy(rng)
    name, values, pos, gaussian, cached = result["numpy"]
    result["numpy"] = {"name": name, "values": values.astype(np.int64).tolist(),
                       "position": int(pos), "has_gaussian": int(gaussian), "cached": float(cached)}
    return result


def _numpy_rng_decode(rng):
    result = copy.deepcopy(rng)
    n = result["numpy"]
    result["numpy"] = (n["name"], np.asarray(n["values"], dtype=np.uint32), n["position"], n["has_gaussian"], n["cached"])
    return result


def _optimizer_steps(opt):
    values = [int(v["step"]) for v in opt.state.values() if "step" in v]
    assert not values or min(values) == max(values)
    return min(values) if values else 0


def _finite_optimizer(opt):
    for value in opt.state.values():
        assert all(not isinstance(tensor, torch.Tensor) or bool(torch.isfinite(tensor).all()) for tensor in value.values())


def source_objective(prediction, truth, x, mode):
    """Exact original FP32 mean order; compact independent FP64 audits.

    No micro-config assertion is bypassed. State/bin loss uses the actual B
    (including the last batch). H2 is a nested mean, not an auxiliary loss.
    """
    assert mode in SOURCE_MODES and prediction.shape == truth.shape
    error = prediction - truth
    node_error = error.abs().mean(-1)
    membership = regime_weights(x)
    observed = membership.argmax(-1)
    observed_cpu = observed.detach().cpu().numpy()
    values = node_error.detach().double().cpu().numpy()
    batch, nodes = observed.shape
    real_counts = np.stack([(observed_cpu == state).sum(1) for state in range(3)])
    group_counts = real_counts.sum(1)
    coefficients = np.zeros((batch, nodes), dtype=np.float64)
    window_ids = torch.arange(batch, device=membership.device).unsqueeze(1).expand(batch, nodes).clone()
    window_ids_cpu = window_ids.detach().cpu().numpy()
    bins_counts, bins_sums, pieces = [], [], []
    if mode == "ordinary":
        loss = error.abs().mean()
        coefficients.fill(1. / (batch * nodes))
    else:
        present_states = int((group_counts > 0).sum())
        for state in range(3):
            masks = [(membership[..., state].bool() & (window_ids == window)) for window in range(batch)]
            counts = [int(np.sum((observed_cpu == state) & (window_ids_cpu == window))) for window in range(batch)]
            bins_counts.append(counts)
            bins_sums.append([float(values[(observed_cpu == state) & (window_ids_cpu == window)].sum()) for window in range(batch)])
            nonempty = [index for index, count in enumerate(counts) if count]
            if nonempty:
                pieces.append(torch.stack([node_error[masks[index]].mean() for index in nonempty]).mean())
                for index in nonempty:
                    coefficients[(observed_cpu == state) & (window_ids_cpu == index)] = 1. / (present_states * len(nonempty) * counts[index])
        loss = torch.stack(pieces).mean()
        assert bins_counts == real_counts.tolist()
    present_count = int((group_counts > 0).sum())
    reconstructed = float((values * coefficients).sum())
    assert bool(torch.isfinite(loss)) and abs(float(coefficients.sum()) - 1.) < 1e-10
    audit = {"mode": mode, "actual_batch_size": batch, "nodes": nodes, "present_state_count": present_count,
             "observed_state_counts": group_counts.tolist(),
             "real_window_state_counts": real_counts.tolist(), "bin_counts_by_state": bins_counts,
             "bin_node_error_sums_by_state": bins_sums,
             "observed_group_node_error_sums": [float(values[observed_cpu == state].sum()) for state in range(3)],
             "observed_group_weight_mass": [float(coefficients[observed_cpu == state].sum()) for state in range(3)],
             "observed_window_group_weight_mass": [[float(coefficients[w][observed_cpu[w] == state].sum()) for w in range(batch)] for state in range(3)],
             "normalized_weight_sum": float(coefficients.sum()), "effective_sample_size": 1. / float(np.square(np.sort(coefficients.reshape(-1))).sum()),
             "coefficient_multiset_sha256": tree_sha(np.sort(coefficients.reshape(-1))),
             "observed_membership_sha256": tree_sha(membership), "prediction_loss": float(loss.detach()),
             "reconstructed_loss_from_detached_errors": reconstructed,
             "loss_reconstruction_absolute_error": abs(float(loss.detach()) - reconstructed),
             "state_labels_observed_only": True, "future_labels_not_grouping_inputs": True,
             "empty_states_and_windows_skipped": True,
             "source_shared_and_private_joint_gradients": True}
    return loss, audit


class StreamingLossAudit:
    """All batches enter sums/digests; only first/last details retained."""
    def __init__(self, mode, expected_windows=None):
        self.row = {"mode": mode, "batches": 0, "node_windows": 0, "state_counts": [0, 0, 0],
                    "state_weight_mass_sum": [0., 0., 0.], "state_node_error_sum": [0., 0., 0.],
                    "real_window_node_counts_histogram": [{}, {}, {}], "window_weight_mass_histogram": [[0] * 101 for _ in range(3)],
                    "real_window_node_count_weight_mass_joint": [{}, {}, {}],
                    "window_weight_mass_exact_sums": [0., 0., 0.], "window_weight_mass_exact_sumsq": [0., 0., 0.],
                    "window_weight_mass_exact_min": [None, None, None], "window_weight_mass_exact_max": [None, None, None],
                    "window_weight_mass_histogram_rule": "floor(100*mass), clamped to bins0..100; display only, loss unchanged",
                    "ESS_sum": 0., "ESS_min": None, "ESS_max": None, "max_loss_reconstruction_absolute_error": 0.,
                    "weight_normalization_violations": 0,
                    "first_batch": None, "last_batch": None, "epoch_visits": []}
        self.digest = hashlib.sha256()
        self.input_digest = hashlib.sha256()
        self.current_epoch = None
        self.epoch_digest = None
        self.epoch_count = 0
        self.expected_windows = expected_windows
        self.epoch_seen = None

    def add(self, audit, indices, epoch, preclip):
        audit = {**audit, "epoch": epoch, "pre_global_clip_gradient_norm": preclip,
                 "global_clip_multiplier": min(1., 5. / (preclip + 1e-6)), "loader_indices": indices.tolist()}
        r = self.row
        r["batches"] += 1
        r["node_windows"] += audit["actual_batch_size"] * audit["nodes"]
        for state in range(3):
            r["state_counts"][state] += audit["observed_state_counts"][state]
            r["state_weight_mass_sum"][state] += audit["observed_group_weight_mass"][state]
            r["state_node_error_sum"][state] += audit["observed_group_node_error_sums"][state]
            for count in audit["real_window_state_counts"][state]:
                key = str(count)
                r["real_window_node_counts_histogram"][state][key] = r["real_window_node_counts_histogram"][state].get(key, 0) + 1
            for mass in audit["observed_window_group_weight_mass"][state]:
                r["window_weight_mass_histogram"][state][min(100, max(0, int(math.floor(mass * 100))))] += 1
                r["window_weight_mass_exact_sums"][state] += mass
                r["window_weight_mass_exact_sumsq"][state] += mass * mass
                r["window_weight_mass_exact_min"][state] = mass if r["window_weight_mass_exact_min"][state] is None else min(r["window_weight_mass_exact_min"][state], mass)
                r["window_weight_mass_exact_max"][state] = mass if r["window_weight_mass_exact_max"][state] is None else max(r["window_weight_mass_exact_max"][state], mass)
            for count, mass in zip(audit["real_window_state_counts"][state], audit["observed_window_group_weight_mass"][state]):
                key = str(count)
                joint = r["real_window_node_count_weight_mass_joint"][state].setdefault(key,
                    {"window_count": 0, "weight_mass_sum": 0., "weight_mass_sumsq": 0., "weight_mass_min": mass, "weight_mass_max": mass})
                joint["window_count"] += 1; joint["weight_mass_sum"] += mass; joint["weight_mass_sumsq"] += mass * mass
                joint["weight_mass_min"] = min(joint["weight_mass_min"], mass)
                joint["weight_mass_max"] = max(joint["weight_mass_max"], mass)
        ess = audit["effective_sample_size"]
        r["ESS_sum"] += ess
        r["ESS_min"] = ess if r["ESS_min"] is None else min(r["ESS_min"], ess)
        r["ESS_max"] = ess if r["ESS_max"] is None else max(r["ESS_max"], ess)
        r["max_loss_reconstruction_absolute_error"] = max(r["max_loss_reconstruction_absolute_error"], audit["loss_reconstruction_absolute_error"])
        r["weight_normalization_violations"] += int(abs(audit["normalized_weight_sum"] - 1.) > 1e-10)
        if r["first_batch"] is None: r["first_batch"] = audit
        r["last_batch"] = audit
        self.digest.update(json_digest(audit).encode())
        self.input_digest.update(indices.numpy().tobytes())
        if self.current_epoch != epoch:
            self._end_epoch()
            self.current_epoch, self.epoch_digest, self.epoch_count = epoch, hashlib.sha256(), 0
            self.epoch_seen = np.zeros(self.expected_windows, np.int16) if self.expected_windows is not None else {}
        self.epoch_digest.update(indices.numpy().tobytes()); self.epoch_count += len(indices)
        for index in indices.tolist():
            if self.expected_windows is not None:
                assert 0 <= index < self.expected_windows
                self.epoch_seen[index] += 1
            else:
                self.epoch_seen[index] = self.epoch_seen.get(index, 0) + 1

    def _end_epoch(self):
        if self.current_epoch is not None:
            if self.expected_windows is not None:
                assert self.epoch_count == self.expected_windows and bool((self.epoch_seen == 1).all())
                visited = np.flatnonzero(self.epoch_seen)
            else:
                visited = np.asarray(sorted(self.epoch_seen), dtype=np.int64)
            self.row["epoch_visits"].append({"epoch": self.current_epoch, "windows": self.epoch_count,
                "loader_indices_sha256": self.epoch_digest.hexdigest(), "unique_windows": len(visited),
                "min_index": int(visited.min()), "max_index": int(visited.max()),
                "sorted_visited_indices_sha256": tree_sha(visited),
                "each_legal_window_visited_exactly_once": bool(self.expected_windows is not None and (self.epoch_seen == 1).all())})

    def finish(self):
        self._end_epoch()
        return {**self.row, "all_batch_scalar_count_coefficient_sha256": self.digest.hexdigest(),
                "all_loader_indices_sha256": self.input_digest.hexdigest(),
                "per_batch_full_node_matrices_not_saved": True,
                "histogram_weight_values_rounded_only_for_display": True}


class Client:
    def __init__(self, city, dataset, initial, cfg, folder, hashes, device="cuda"):
        self.city, self.dataset, self.cfg, self.folder, self.hashes, self.device = city, dataset, cfg, Path(folder), hashes, device
        self.model = build_model(cfg, device)
        self.model.load_state_dict(initial)
        self.shared = [k for k in initial if original.is_shared(k, "full_private")]
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["local_lr"], weight_decay=cfg["weight_decay"])
        self.last_shared_delta = None
        self.local_rng = capture_rng()
        self.loader_state = None
        self.position = {"round": 0, "epoch": 0, "batch": 0, "at_round_boundary": True}

    def broadcast(self, shared):
        assert set(shared) == set(self.shared)
        before = private_state(self.model)
        state = self.model.state_dict(); state.update({k: v.to(self.device) for k, v in shared.items()})
        self.model.load_state_dict(state)
        assert exact_equal(before, private_state(self.model))
        return tree_sha(before)

    def update(self, shared, round_id):
        private_before = self.broadcast(shared)
        cfg = self.cfg
        nseed = cfg["seed"] + round_id * 1000 + CITIES.index(self.city)
        original.seed(nseed)
        loader_generator = torch.Generator().manual_seed(nseed)
        loader = DataLoader(Indexed(self.dataset), batch_size=cfg["local_batch_size"], shuffle=True, num_workers=0, generator=loader_generator)
        before_steps = _optimizer_steps(self.opt)
        before_opt_sha = tree_sha(self.opt.state_dict())
        started, total, samples = time.perf_counter(), 0., 0
        mode = "ordinary" if self.city == cfg["target"] else cfg["source_loss"]
        stream = StreamingLossAudit(mode, expected_windows=len(self.dataset))
        hook_calls = self.model._client_candidate_hook.calls
        self.model.train()
        for epoch in range(cfg["local_epochs"]):
            for batch, (x, y, profile, indices) in enumerate(loader):
                x, y, profile = x.to(self.device), y.to(self.device), profile.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                prediction = self.model(x, profile)
                loss, audit = source_objective(prediction, y, x, mode)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip_grad_norm"])
                assert bool(torch.isfinite(norm))
                self.opt.step()
                stream.add(audit, indices, epoch + 1, float(norm))
                total += float(loss.detach()) * len(x); samples += len(x)
                if batch % 100 == 0:
                    log({"stage": "client_local", "city": self.city, "round": round_id, "epoch": epoch + 1,
                         "batch": batch + 1, "batches": len(loader), "loss_recipe": mode, "elapsed_s": time.perf_counter() - started})
        expected_steps = len(loader) * cfg["local_epochs"]
        assert _optimizer_steps(self.opt) == before_steps + expected_steps
        _finite_optimizer(self.opt)
        state = self.model.state_dict()
        delta = {k: state[k].detach().cpu().clone() - shared[k] for k in self.shared}
        assert all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in delta.values())
        self.last_shared_delta = delta
        self.local_rng, self.loader_state = capture_rng(), loader_generator.get_state().clone()
        self.position = {"round": round_id, "epoch": cfg["local_epochs"], "batch": len(loader), "at_round_boundary": True}
        audit = stream.finish()
        assert audit["batches"] == expected_steps and all(v["windows"] == len(self.dataset) for v in audit["epoch_visits"])
        assert audit["weight_normalization_violations"] == 0
        assert self.model._client_candidate_hook.calls - hook_calls == expected_steps
        return delta, {"city": self.city, "role": "target" if self.city == cfg["target"] else "source", "loss_recipe": mode,
            "loss": total / samples, "batches": expected_steps, "samples": samples, "epochs": cfg["local_epochs"],
            "elapsed_s": time.perf_counter() - started, "optimizer_persistent": True,
            "optimizer_steps_before": before_steps, "optimizer_steps_after": _optimizer_steps(self.opt),
            "optimizer_before_sha256": before_opt_sha, "optimizer_after_sha256": tree_sha(self.opt.state_dict()),
            "shared_anchor_sha256": tree_sha(shared), "last_shared_delta_sha256": tree_sha(delta),
            "private_broadcast_before_sha256": private_before, "private_broadcast_after_sha256": private_before,
            "private_end_sha256": tree_sha(private_state(self.model)), "model_end_sha256": tree_sha(clone_state(self.model)),
            "rng_seed": nseed, "dropout_seed": nseed, "shuffle_seed": nseed,
            "global_rng_final_sha256": tree_sha(self.local_rng), "loader_generator_final_sha256": tree_sha(self.loader_state),
            "loss_audit": audit, "private_broadcast_unchanged": True,
            "private_context_forward_calls": expected_steps}

    def snapshot(self, round_id):
        return {"city": self.city, "round": round_id, "hashes": self.hashes, "model": clone_state(self.model),
            "optimizer": self.opt.state_dict(), "last_shared_delta": self.last_shared_delta,
            "rng": _numpy_rng_safe(self.local_rng), "loader_generator": self.loader_state,
            "position": self.position,
            "requires_grad": {name: p.requires_grad for name, p in self.model.named_parameters()},
            "model_training_modes": {name: m.training for name, m in self.model.named_modules()},
            "client_private_context_is_RAM_only_not_in_state_dict": True}

    def restore(self, payload, round_id):
        assert payload["city"] == self.city and payload["round"] == round_id and payload["hashes"] == self.hashes
        self.model.load_state_dict(payload["model"])
        for name, parameter in self.model.named_parameters(): parameter.requires_grad_(payload["requires_grad"][name])
        for name, module in self.model.named_modules(): module.training = payload["model_training_modes"][name]
        self.opt.load_state_dict(payload["optimizer"])
        self.last_shared_delta = payload["last_shared_delta"]
        self.local_rng = _numpy_rng_decode(payload["rng"])
        self.loader_state, self.position = payload["loader_generator"], payload["position"]
        assert tree_sha(clone_state(self.model)) == tree_sha(payload["model"])
        assert tree_sha(self.opt.state_dict()) == tree_sha(payload["optimizer"])
        assert self.position["at_round_boundary"]
        _finite_optimizer(self.opt)


build_client = Client


def server_aggregate(shared, updates, cfg):
    """Only shared tensors cross the logical server API, every round."""
    assert set(updates) == set(CITIES) and all(set(v) == set(shared) for v in updates.values())
    source_order = [c for c in CITIES if c != cfg["target"]]
    assert cfg["aggregation"] == "tra"
    new, audit = original.server_aggregate(shared, [updates[c] for c in source_order], updates[cfg["target"]], cfg)
    audit.update(aggregation="tra", reference="same-round nonunit source arithmetic mean, negative parallel removed")
    keys = list(shared)
    blocks = [[key for key in keys if original.group(key) == name] for name in audit["groups"]]
    sources = [updates[c] for c in source_order]
    target = updates[cfg["target"]]
    uniform = {key: sum(source[key] for source in sources) / 3 for key in keys}
    reference = original.remove_conflicting_parallel(uniform, target, blocks)
    weighted, _ = original.mgda_weights(sources, keys)
    candidate = original.remove_conflicting_parallel(weighted, target, blocks)
    correction, _ = original.match(candidate, reference, blocks)
    # Read-only scalar geometry; does not modify the executed aggregation.
    audit["block_geometry"] = {}
    for name, block in zip(audit["groups"], blocks):
        audit["block_geometry"][name] = {
            "raw_source_gram": [[original.dot(a, b, block) for b in sources] for a in sources],
            "source_target_dots": [original.dot(source, target, block) for source in sources],
            "target_squared_norm": original.energy(target, block),
            "raw_uniform_squared_norm": original.energy(uniform, block),
            "nonconflicting_reference_squared_norm": original.energy(reference, block),
            "actual_nonconflicting_candidate_energy": original.energy(candidate, block),
            "candidate_target_dot": original.dot(candidate, target, block),
            "matched_correction_target_dot": original.dot(correction, target, block),
            "actual_matched_correction_energy": original.energy(correction, block)}
    audit["geometry_audit_deterministically_recomputed_from_uploaded_deltas"] = True
    assert all(original.is_shared(k, "full_private") and bool(torch.isfinite(v).all()) for k, v in new.items())
    audit.update(shared_anchor_sha256=tree_sha(shared), new_shared_sha256=tree_sha(new),
                 client_delta_sha256={c: tree_sha(updates[c]) for c in CITIES}, source_order=source_order,
                 target_delta_sha256=tree_sha(updates[cfg["target"]]))
    return new, audit


def datasets(target, cfg):
    manifest = json.loads(DATASET_MANIFEST.read_text())
    manifest["root"] = str(DATA_ROOT if DATA_ROOT is not None else (DATASET_MANIFEST.parent / manifest["root"]).resolve())
    clients, ranges = {}, {}
    stats = Range(cfg["support_start"], cfg["support_fit_end"])
    support = Range(cfg["support_start"], cfg["support_end"])
    for city in CITIES:
        spec = manifest["cities"][city]
        if city == target:
            end = spec["steps_per_day"] * cfg["target_support_days"][city]
            assert end == 864
            data = load_city(manifest, city, verify_hash=True, max_steps=end)
            bound = stats
        else:
            end = int(spec["expected_shape"][0] * cfg["source_train_fraction"])
            data = load_city(manifest, city, verify_hash=True, max_steps=end)
            bound = Range(0, end)
        starts = window_starts_inside(bound, 12, 6)
        clients[city] = TrafficWindows(data, starts, 12, 6, bound)
        d = clients[city]
        ranges[city] = {"train": [bound.start, bound.end], "stats": [bound.start, bound.end],
            "windows": len(starts), "nodes": data.features.shape[1], "cap": None,
            "starts_sha256": tree_sha(starts), "first_last_starts": [int(starts[0]), int(starts[-1])],
            "normalization_mean": d.mean, "normalization_std": d.std, "profile_sha256": tree_sha(d.profile)}
    td = clients[target].city
    gate_range, val_range = Range(cfg["support_fit_end"], cfg["support_gate_end"]), Range(cfg["support_gate_end"], 864)
    gate = TrafficWindows(td, window_starts_inside(gate_range, 12, 6), 12, 6, stats)
    validation = TrafficWindows(td, window_starts_inside(val_range, 12, 6), 12, 6, stats)
    full = TrafficWindows(td, window_starts_inside(support, 12, 6), 12, 6, support)
    fit = clients[target]
    assert max(fit.starts) + 18 <= min(gate.starts) and max(gate.starts) + 18 <= min(validation.starts)
    assert len(fit) == 415
    assert len(gate) == len(validation) == 199
    assert len(full) == 847
    ranges["target_partitions"] = {"fit": [stats.start, stats.end], "gate": [gate_range.start, gate_range.end],
        "validation": [val_range.start, val_range.end], "final_support": [support.start, support.end],
        "fit_stats": [stats.start, stats.end], "gate_stats": [stats.start, stats.end], "validation_stats": [stats.start, stats.end],
        "final_and_test_stats": [support.start, support.end], "fit_windows": len(fit), "gate_windows": len(gate),
        "validation_windows": len(validation), "final_windows": len(full), "final_mean": full.mean, "final_std": full.std,
        "final_profile_sha256": tree_sha(full.profile), "all_windows_purged_inside_partition": True}
    return manifest, clients, fit, gate, validation, full, ranges


def _write_tensor(path, payload, out):
    original.tensor_save(path, payload)
    return {"relative_path": str(path.relative_to(out)), "sha256": sha(path), "bytes": path.stat().st_size}


def _read_tensor(ack, out, hashes):
    path = (out / ack["relative_path"]).resolve()
    assert path.is_relative_to(out.resolve()) and sha(path) == ack["sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["hashes"] == hashes
    return payload


def commit_round(out, target, clients, shared, target_delta, history, hashes, round_id):
    slot = round_id % 2
    acknowledgements = {c: _write_tensor(out / "client_checkpoints" / target / c / ("latest_slot%d.pt" % slot),
                                       clients[c].snapshot(round_id), out) for c in CITIES}
    server = _write_tensor(out / "server_shared_checkpoints" / target / ("latest_slot%d.pt" % slot),
        {"hashes": hashes, "round": round_id, "shared": shared, "last_target_update": target_delta,
         "history": history, "server_contains_only_shared_tensors_and_scalar_audits": True}, out)
    progress = {"status": "running", "hashes": hashes, "target": target, "round": round_id, "slot": slot,
                "clients": acknowledgements, "server_checkpoint": server,
                "transaction": "write inactive two-slot bank, then atomically publish JSON commit"}
    save(out / (target + ".progress.json"), progress)
    return progress


def final_adapt(client, global_shared, last_target_update, fit, gate, full, out):
    """Original native 4x5 calibration, same reset and genuine 50-epoch final.

    Epoch-boundary final checkpoints contain optimizer AND RNG AND loader
    state; interrupted calibration is deterministically rebuilt and counted.
    """
    cfg, hashes, model = client.cfg, client.hashes, client.model
    assert exact_equal(last_target_update, client.last_shared_delta)
    final_progress = out / (client.city + ".final.progress.json")
    groups = sorted({original.group(k) for k in client.shared})
    started, calibration_reruns = time.perf_counter(), 0
    if final_progress.exists():
        progress = json.loads(final_progress.read_text())
        payload = _read_tensor(progress["checkpoint"], out, hashes)
        assert payload["city"] == client.city and payload["stage"] == "final"
        model.load_state_dict(payload["model"])
        opt, names = original.optimizer(model, "private_lr2")
        opt.load_state_dict(payload["optimizer"])
        frozen = payload["frozen"]
        loader_generator = torch.Generator(); loader_generator.set_state(payload["loader_generator"])
        restore_rng(_numpy_rng_decode(payload["rng"]))
        completed, coefficients = payload["epoch"], np.asarray(payload["metadata"]["calibration_coefficients"])
        metadata = payload["metadata"]
        log({"stage": "resume_final_epoch", "city": client.city, "epoch": completed})
    else:
        client.broadcast(global_shared)
        local, private = clone_state(model), private_state(model)
        def reset(shared):
            state = copy.deepcopy(local); state.update({k: v.detach().cpu() for k, v in shared.items()})
            model.load_state_dict(state)
            assert exact_equal(private, private_state(model))
        def response(shared):
            reset(shared)
            original.train(model, fit, client.device, cfg["seed"] + 10000, cfg["calibration_epochs"], adapt=True)
            model.eval(); predictions, truths = [], []
            with torch.no_grad():
                for x, y, profile in DataLoader(gate, batch_size=8, shuffle=False):
                    predictions.append((model(x.to(client.device), profile.to(client.device)).cpu() * gate.std + gate.mean).numpy()[..., [0, 2, 5]])
                    truths.append((y * gate.std + gate.mean).numpy()[..., [0, 2, 5]])
            return np.concatenate(predictions), np.concatenate(truths)
        calibration_marker = out / (client.city + ".calibration.attempt.json")
        if calibration_marker.exists():
            marker = json.loads(calibration_marker.read_text()); assert marker["hashes"] == hashes
            calibration_reruns = marker["attempts"]
        save(calibration_marker, {"hashes": hashes, "attempts": calibration_reruns + 1,
             "restart_policy": "whole deterministic calibration phase; previous partial attempt cost explicitly bounded"})
        prediction, truth = response(global_shared)
        mask, error = np.isfinite(truth) & (truth != 0), truth - prediction
        count = mask.sum((0, 1)); assert (count > 0).all()
        scales = np.stack([(np.abs(error) * mask).sum((0, 1)) / count,
                           np.sqrt((error ** 2 * mask).sum((0, 1)) / count)]).clip(1e-6)
        features = []
        for group in groups:
            altered = {k: global_shared[k] + (cfg["local_response_scale"] * last_target_update[k] if original.group(k) == group else 0) for k in client.shared}
            p, observed = response(altered); assert np.array_equal(observed, truth)
            features.append(p - prediction)
        coefficients = original.solve(features, error, mask, scales, 3)
        calibrated = {k: global_shared[k] + cfg["local_response_scale"] * coefficients[groups.index(original.group(k))] * last_target_update[k] for k in client.shared}
        reset(calibrated)
        opt, names = original.optimizer(model, "private_lr2")
        assert sorted(group["lr"] for group in opt.param_groups) == sorted((cfg["head_lr"], cfg["private_lr"], cfg["rest_lr"]))
        frozen = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}
        original.seed(cfg["seed"] + 10000)
        loader_generator = torch.Generator().manual_seed(cfg["seed"] + 10000)
        completed = 0
        metadata = {"epochs": cfg["target_epochs"], "support_windows": len(full), "batches": math.ceil(len(full) / 8) * cfg["target_epochs"],
            "trainable_names": names, "private_reset_before_refit": True, "private_reset_sha256": tree_sha(private),
            "calibrated_shared_sha256": tree_sha(calibrated), "calibration_coefficients": coefficients.tolist(),
            "calibration_groups": groups, "calibration_fits": 4, "calibration_epochs_per_fit": cfg["calibration_epochs"],
            "calibration_fit_batches": math.ceil(len(fit) / 8) * cfg["calibration_epochs"] * 4,
            "calibration_mask": "finite truth!=0; inherited, not chosen by results", "calibration_extra_full_phase_attempts": calibration_reruns,
            "calibration_extra_work_exact_count_known": calibration_reruns == 0,
            "calibration_extra_work_upper_bound_batches": calibration_reruns * math.ceil(len(fit) / 8) * cfg["calibration_epochs"] * 4,
            "native_calibration_target_loss": "ordinary L1", "adaptation_rng_seed": cfg["seed"] + 10000,
            "calibration_elapsed_s_current_successful_attempt": time.perf_counter() - started,
            "checkpoint_interval": "every epoch boundary; interrupted uncommitted epoch deterministically repeated",
            "checkpoint_history": [], "elapsed_s": 0.}
        # Commit calibrated start so interruption in epoch1 resumes identical.
        _commit_final(client, opt, loader_generator, frozen, metadata, 0, out)
    loader = DataLoader(full, batch_size=cfg["target_batch_size"], shuffle=True, num_workers=0, generator=loader_generator)
    for epoch in range(completed, cfg["target_epochs"]):
        epoch_start = time.perf_counter(); before = _optimizer_steps(opt)
        model.train()
        for x, y, profile in loader:
            opt.zero_grad(set_to_none=True)
            loss = (model(x.to(client.device), profile.to(client.device)) - y.to(client.device)).abs().mean()
            assert bool(torch.isfinite(loss)); loss.backward()
            assert bool(torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])))
            opt.step()
        assert _optimizer_steps(opt) == before + len(loader)
        assert all(torch.equal(v, dict(model.named_parameters())[name].detach().cpu()) for name, v in frozen.items())
        _finite_optimizer(opt)
        metadata["elapsed_s"] += time.perf_counter() - epoch_start
        metadata["checkpoint_history"].append({"epoch": epoch + 1, "optimizer_steps": _optimizer_steps(opt),
            "model_sha256": tree_sha(clone_state(model)), "private_sha256": tree_sha(private_state(model)),
            "optimizer_sha256": tree_sha(opt.state_dict()), "global_rng_sha256": tree_sha(capture_rng()),
            "loader_generator_sha256": tree_sha(loader_generator.get_state()), "frozen_unchanged": True})
        _commit_final(client, opt, loader_generator, frozen, metadata, epoch + 1, out)
        if (epoch + 1) % 10 == 0:
            log({"stage": "final_adaptation", "city": client.city, "epoch": epoch + 1, "elapsed_s": time.perf_counter() - started})
    metadata.update(frozen_unchanged=True, completed_epochs=cfg["target_epochs"], optimizer_final_sha256=tree_sha(opt.state_dict()),
                    final_model_sha256=tree_sha(clone_state(model)), final_private_sha256=tree_sha(private_state(model)))
    return metadata


def _commit_final(client, optimizer, generator, frozen, metadata, epoch, out):
    slot = epoch % 2
    path = client.folder / ("final_slot%d.pt" % slot)
    ack = _write_tensor(path, {"hashes": client.hashes, "city": client.city, "stage": "final", "epoch": epoch,
        "model": clone_state(client.model), "optimizer": optimizer.state_dict(), "rng": _numpy_rng_safe(capture_rng()),
        "loader_generator": generator.get_state().clone(), "frozen": frozen, "metadata": metadata,
        "position": {"stage": "final", "epoch": epoch, "batch": 0, "at_epoch_boundary": True},
        "requires_grad": {n: p.requires_grad for n, p in client.model.named_parameters()},
        "model_training_modes": {n: m.training for n, m in client.model.named_modules()}}, out)
    save(out / (client.city + ".final.progress.json"), {"hashes": client.hashes, "stage": "final", "epoch": epoch, "slot": slot, "checkpoint": ack})


@torch.no_grad()
def evaluate(client, test):
    client.model.eval()
    counts = np.zeros((2, 6)); absolute = counts.copy(); squared = counts.copy()
    state_counts = np.zeros((3, 6)); state_absolute = state_counts.copy(); state_squared = state_counts.copy()
    for x, y, profile in DataLoader(test, batch_size=client.cfg["target_batch_size"], shuffle=False):
        truth = (y * test.std + test.mean).double()
        prediction = (client.model(x.to(client.device), profile.to(client.device)).cpu() * test.std + test.mean).double()
        assert bool(torch.isfinite(truth).all()) and bool(torch.isfinite(prediction).all())
        error = prediction - truth
        membership = regime_weights(x)
        for index, valid in enumerate((torch.ones_like(truth, dtype=torch.bool), truth != 0)):
            counts[index] += valid.sum((0, 1)).numpy()
            absolute[index] += (error.abs() * valid).sum((0, 1)).numpy()
            squared[index] += (error.square() * valid).sum((0, 1)).numpy()
        for state in range(3):
            valid = membership[..., state].bool().unsqueeze(-1).expand_as(truth)
            state_counts[state] += valid.sum((0, 1)).numpy()
            state_absolute[state] += (error.abs() * valid).sum((0, 1)).numpy()
            state_squared[state] += (error.square() * valid).sum((0, 1)).numpy()
    assert (counts > 0).all() and np.array_equal(state_counts.sum(0), counts[0])
    def score(n, ae, sse):
        return {"counts": n.astype(int).tolist(), "AE_sum": ae.tolist(), "SSE_sum": sse.tolist(),
                "mae": np.divide(ae, n, out=np.zeros_like(ae), where=n > 0).tolist(),
                "rmse": np.sqrt(np.divide(sse, n, out=np.zeros_like(sse), where=n > 0)).tolist(),
                "empty_state": bool((n == 0).all())}
    scores = {name: score(counts[i], absolute[i], squared[i]) for i, name in enumerate(("primary_unmasked", "auxiliary_masked"))}
    states = {name: score(state_counts[i], state_absolute[i], state_squared[i]) for i, name in enumerate(STATES)}
    assert np.allclose(state_absolute.sum(0), absolute[0], rtol=1e-12, atol=1e-7)
    assert np.allclose(state_squared.sum(0), squared[0], rtol=1e-12, atol=1e-7)
    return scores, {"groups": states, "state_order": list(STATES), "labels_observed_only": True,
                    "same_single_prediction_pass_as_overall": True, "overall": scores["primary_unmasked"]}


def valid_result(path, cfg, hashes, target=None):
    path = Path(path)
    if not path.exists(): return False
    row = json.loads(path.read_text())
    assert row["status"] == "ok" and row["hashes"] == hashes and row["config"] == cfg
    assert target is None or row["target"] == target
    assert row["completed_rounds"] == cfg["rounds"] and row["final_adaptation"]["completed_epochs"] == cfg["target_epochs"]
    assert all(np.isfinite(original.six(value)).all() for value in row["scores"].values())
    assert not any(row["protocol"][key] for key in ("test_used_for_selection", "validation_used_for_selection", "published_values_used_for_selection", "private_parameters_uploaded"))
    for ack in list(row["checkpoints"]["round"]["clients"].values()) + [row["checkpoints"]["round"]["server_checkpoint"], row["checkpoints"]["final"]["checkpoint"]]:
        checkpoint = path.parent / ack["relative_path"]
        assert checkpoint.exists() and sha(checkpoint) == ack["sha256"]
    return True


def _cleanup_inactive(out, progress, final_progress):
    """Only predeclared unreferenced NEW-job scratch slots; active SHA intact."""
    active_paths = {(out / ack["relative_path"]).resolve() for ack in list(progress["clients"].values()) + [progress["server_checkpoint"], final_progress["checkpoint"]]}
    removed, removed_bytes = [], 0
    for active in active_paths:
        name = active.name
        assert name in ("latest_slot0.pt", "latest_slot1.pt", "final_slot0.pt", "final_slot1.pt")
        inactive_name = name.replace("slot0", "slot1") if "slot0" in name else name.replace("slot1", "slot0")
        inactive = active.with_name(inactive_name)
        assert inactive.is_relative_to(out.resolve()) and inactive not in active_paths
        if inactive.exists():
            size = inactive.stat().st_size; digest = sha(inactive)
            inactive.unlink()
            removed.append({"relative_path": str(inactive.relative_to(out.resolve())), "bytes": size, "sha256": digest})
            removed_bytes += size
    return {"predeclared_NEW_job_inactive_temporary_slots_reclaimed": removed,
            "reclaimed_bytes": removed_bytes, "active_committed_checkpoints_retained": True,
            "historical_namespaces_never_touched": True}


def run_job(cfg, output_dir, target, hashes, device="cuda"):
    validate_config(cfg)
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    output = out / (target + ".json")
    with (out / (target + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if valid_result(output, cfg, hashes, target): return json.loads(output.read_text())
        return _run_job(cfg, out, target, hashes, device)


def _run_job(cfg, out, target, hashes, device):
    if str(device).startswith("cuda"): torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    invocation_path = out / (target + ".invocations.json")
    prior_invocations = 0
    if invocation_path.exists():
        previous_invocations = json.loads(invocation_path.read_text())
        assert previous_invocations["hashes"] == hashes
        prior_invocations = previous_invocations["attempts"]
    save(invocation_path, {"hashes": hashes, "attempts": prior_invocations + 1,
         "unknown_interrupted_uncommitted_work_must_not_be_counted_as_zero": True})
    manifest, ds, fit, gate, validation, full, ranges = datasets(target, cfg)
    original.seed(cfg["seed"])
    template = build_model(cfg, "cpu")
    initial = clone_state(template); del template
    shared = {k: v.clone() for k, v in initial.items() if original.is_shared(k, "full_private")}
    clients = {c: Client(c, ds[c], initial, cfg, out / "client_checkpoints" / target / c, hashes, device) for c in CITIES}
    history, target_delta, completed, progress = [], None, 0, None
    commit = out / (target + ".progress.json")
    if commit.exists():
        progress = json.loads(commit.read_text()); assert progress["hashes"] == hashes and progress["target"] == target
        checkpoint = _read_tensor(progress["server_checkpoint"], out, hashes)
        shared, target_delta, history, completed = checkpoint["shared"], checkpoint["last_target_update"], checkpoint["history"], checkpoint["round"]
        for c in CITIES: clients[c].restore(_read_tensor(progress["clients"][c], out, hashes), completed)
        log({"stage": "resume_committed_round", "target": target, "round": completed})
    resumed_from = completed
    for round_id in range(completed + 1, cfg["rounds"] + 1):
        updates, telemetry = {}, {}
        for c in CITIES: updates[c], telemetry[c] = clients[c].update(shared, round_id)
        target_delta = updates[target]
        shared, server = server_aggregate(shared, updates, cfg)
        history.append({"round": round_id, "clients": telemetry, "server": server})
        progress = commit_round(out, target, clients, shared, target_delta, history, hashes, round_id)
        log({"stage": "round_committed", "target": target, "round": round_id, "aggregation": cfg["aggregation"], "elapsed_s": time.perf_counter() - started})
    final = final_adapt(clients[target], shared, target_delta, fit, gate, full, out)
    # FIRST complete target-series load happens only after training is frozen.
    evaluation_path = out / (target + ".evaluation.completed.json")
    model_sha = tree_sha(clone_state(clients[target].model))
    if evaluation_path.exists():
        evaluated = json.loads(evaluation_path.read_text())
        assert evaluated["hashes"] == hashes and evaluated["final_model_sha256"] == model_sha
        scores, state_scores = evaluated["scores"], evaluated["state_scores"]
        bounds = Range(*evaluated["test"]); test_windows = evaluated["test_windows"]
    else:
        attempt_path = out / (target + ".evaluation.attempts.json")
        evaluation_attempts = 0
        if attempt_path.exists():
            attempts = json.loads(attempt_path.read_text()); assert attempts["hashes"] == hashes
            evaluation_attempts = attempts["attempts"]
        save(attempt_path, {"hashes": hashes, "attempts": evaluation_attempts + 1,
             "incomplete_interrupted_pass_may_be_repeated_not_hidden": True})
        data = load_city(manifest, target, verify_hash=True)
        bounds = Range(int(len(data.features) * .8), len(data.features))
        test = TrafficWindows(data, window_starts_inside(bounds, 12, 6), 12, 6, Range(cfg["support_start"], 864))
        scores, state_scores = evaluate(clients[target], test); test_windows = len(test)
        evaluated = {"status": "ok", "hashes": hashes, "final_model_sha256": model_sha,
                     "scores": scores, "state_scores": state_scores, "test": [bounds.start, bounds.end],
                     "test_windows": test_windows, "evaluation_attempts": evaluation_attempts + 1,
                     "completed_single_pass_reused_after_result_commit_interruption": True}
        save(evaluation_path, evaluated)
    final_progress = json.loads((out / (target + ".final.progress.json")).read_text())
    cleanup = _cleanup_inactive(out, progress, final_progress)
    shared_count = sum(v.numel() for v in shared.values())
    total_parameters = sum(p.numel() for p in clients[target].model.parameters())
    local_steps = sum(t["batches"] for h in history for t in h["clients"].values())
    row = {"status": "ok", "target": target, "seed": cfg["seed"], "hashes": hashes, "config": cfg,
        "completed_rounds": len(history), "history": history, "scores": scores, "state_scores": state_scores,
        "ranges": ranges, "target_fit": ranges["target_partitions"]["fit"], "gate": ranges["target_partitions"]["gate"],
        "support_validation": ranges["target_partitions"]["validation"], "target_full_support": [cfg["support_start"], 864],
        "test": [bounds.start, bounds.end], "test_windows": test_windows, "test_stats": [cfg["support_start"], 864],
        "evaluation": {"completed_pass_sha256": sha(evaluation_path), "attempts": evaluated["evaluation_attempts"],
                       "completed_pass_reused_not_predicted_again_after_commit_interruption": True},
        "target_physical_support_days": cfg["target_support_days"][target] * (864 - cfg["support_start"]) / 864,
        "initialization": {"seed": cfg["seed"], "candidate": "trend_innovation",
                           "model_sha256": tree_sha(initial), "shared_sha256": tree_sha({k: initial[k] for k in shared}),
                           "private_sha256": tree_sha({k: v for k, v in initial.items() if k not in shared}),
                           "innovation_weight_initialized_from_original_history_weight": torch.equal(initial["history_embed.0.innovation.weight"], initial["history_embed.0.weight"])},
        "final_adaptation": final, "checkpoints": {"round": progress, "final": final_progress}, "checkpoint_cleanup": cleanup,
        "communication": {"shared_parameters": shared_count, "total_parameters": total_parameters, "quant_bits": 32,
            "update_carriers_per_client_per_round": 1, "uploaded_clients_per_round": 4,
            "upload_bytes": shared_count * 4 * 4 * cfg["rounds"],
            "broadcast_bytes": shared_count * 4 * (4 * cfg["rounds"] + 1),
            "no_optimizer_private_raw_upload": True, "payload_only_excludes_protocol_headers": True,
            "simulator_payload_not_measured_network_latency": True},
        "costs": {"actual_local_optimizer_steps": local_steps, "actual_final_optimizer_steps": final["batches"],
            "native_calibration_optimizer_steps": final["calibration_fit_batches"],
            "invocation_attempts": prior_invocations + 1,
            "uncommitted_interruption_recompute_cost_known": prior_invocations == 0 and final["calibration_extra_work_exact_count_known"],
            "prior_interrupted_invocations": prior_invocations,
            "additional_uncommitted_optimizer_work_upper_bound": prior_invocations * (
                sum(math.ceil(len(d) / 32) * cfg["local_epochs"] for d in ds.values()) +
                final["calibration_fit_batches"] + math.ceil(len(full) / 8)),
            "uncommitted_extra_work_bound_not_claimed_actual_measured_steps": True,
            "round_boundary_resume_repeats_uncommitted_round_not_partial_trajectory": True,
            "retained_checkpoint_bytes": sum(a["bytes"] for a in list(progress["clients"].values()) + [progress["server_checkpoint"], final_progress["checkpoint"]])},
        "protocol": {"test_used_for_selection": False, "validation_used_for_selection": False,
            "published_values_used_for_selection": False, "private_parameters_uploaded": False,
            "same_initial_shared_anchor_all_clients": True, "all_sources": True, "source_window_cap": None,
            "optimizer_persistent_between_rounds": True, "primary_mask": "none", "auxiliary_mask": "truth!=0",
            "target_loss": "ordinary L1", "private_context_all_client_all_forward_paths": "local",
            "source_shared_private_joint_optimization": True, "server_aggregation": cfg["aggregation"],
            "energy_reference_rule": "same-round arithmetic-mean-source FULL per functional block",
            "no_early_stopping": True},
        "elapsed_s": time.perf_counter() - started, "elapsed_s_scope": "current invocation parallel-management wall time",
        "resumed_from_round": resumed_from, "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if str(device).startswith("cuda") else 0.,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20 if str(device).startswith("cuda") else 0.}
    save(out / (target + ".json"), row)
    assert valid_result(out / (target + ".json"), cfg, hashes, target)
    log({"stage": "COMPLETE", "target": target, "seed": cfg["seed"], "source_mode": cfg["source_loss"], "primary": scores["primary_unmasked"]})
    return row
