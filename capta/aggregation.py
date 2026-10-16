"""Shared aggregation retaining aligned and orthogonal source components."""
import numpy as np
import torch
from .utils import group, is_shared
CITIES = ("metr-la", "pems-bay", "chengdu", "shenzhen")

def energy(delta, keys=None):
    return sum(float(delta[k].double().square().sum()) for k in (keys or delta.keys()))


def project(source, target, groups):
    """Split each block into target-parallel and orthogonal components.

    A zero target block defines no direction, so its parallel component is zero.
    This decomposition itself does not decide which component to discard.
    """
    parallel = {}
    for keys in groups:
        den = energy(target, keys)
        dot = sum(float((source[k].double() * target[k].double()).sum()) for k in keys)
        for k in keys:
            parallel[k] = dot / den * target[k] if den > 0 else torch.zeros_like(source[k])
    orth = {k: source[k] - parallel[k] for k in source}
    return parallel, orth


def remove_conflicting_parallel(source, target, groups):
    """Remove only negative target-parallel components, block by block.

    s_safe = s - min(<s,t> / ||t||^2, 0) * t.
    Positive parallel and orthogonal components are retained. With t=0 the
    source block is unchanged. Inputs are never modified in place.
    """
    retained = {}
    for block in groups:
        den = energy(target, block)
        coefficient = min(dot(source, target, block) / den, 0.) if den > 0 else 0.
        for k in block:
            retained[k] = source[k] - coefficient * target[k] if coefficient < 0 else source[k].clone()
    return retained


def match(delta, reference, groups):
    result, scales, statuses, errors = {}, [], [], []
    for keys in groups:
        den, wanted = energy(delta, keys), energy(reference, keys)
        # A fully rejected source correction has no direction to rescale.
        # Keep it zero rather than resurrecting a conflicting update.
        scale = (wanted / den) ** .5 if den > 0 else 0.
        scales.append(scale)
        status = "matched" if den > 0 else "zero_candidate" if wanted == 0 else "zero_candidate_unmatched_reference"
        statuses.append(status)
        for k in keys:
            result[k] = delta[k] * scale
        actual = energy(result, keys)
        error = abs(actual / wanted - 1.) if wanted > 0 else actual
        errors.append(error)
        if status != "zero_candidate_unmatched_reference":
            assert error < 1e-6, error
    return result, dict(scales=scales, block_statuses=statuses,
                        all_blocks_matched=all(e < 1e-6 for e in errors),
                        max_relative_energy_error=max(errors))

def dot(a, b, keys):
    return sum(float((a[k].double()*b[k].double()).sum()) for k in keys)


def pool(vectors, weights):
    return {k: sum(float(w)*v[k] for w, v in zip(weights, vectors)) for k in vectors[0]}


def mgda_weights(vectors, keys):
    gram = np.array([[dot(a, b, keys) for b in vectors] for a in vectors])
    norms = np.sqrt(np.diag(gram)).clip(1e-20)
    gram /= np.outer(norms, norms)
    candidates = []
    for mask in range(1, 1 << len(vectors)):
        ix = [i for i in range(len(vectors)) if mask & (1 << i)]
        g = gram[np.ix_(ix, ix)]
        # Equality-constrained quadratic minimum, with simplex faces enumerated.
        kkt = np.block([[g, np.ones((len(ix), 1))],
                        [np.ones((1, len(ix))), np.zeros((1, 1))]])
        solution = np.linalg.lstsq(kkt, np.r_[np.zeros(len(ix)), 1.], rcond=None)[0][:-1]
        if np.min(solution) < -1e-9:
            continue
        w = np.zeros(len(vectors))
        w[ix] = np.maximum(solution, 0)
        w /= w.sum()
        candidates.append((float(w @ gram @ w), w))
    objective, weights = min(candidates, key=lambda x: x[0])
    gradient = gram @ weights
    active = weights > 1e-8
    assert np.max(np.abs(gradient[active]-objective)) < 1e-7
    assert np.min(gradient-objective) >= -1e-7
    assert objective <= float(np.mean(gram)) + 1e-8
    unit = [{k: v[k]/float(n) for k in keys} for v, n in zip(vectors, norms)]
    return pool(unit, weights), {"weights": weights.tolist(), "unit_gram": gram.tolist(),
                                "objective": objective, "simplex_kkt_passed": True}

def server_aggregate(shared, source_updates, target_update, cfg):
    keys = list(shared)
    assert len(source_updates) == 3 and set(target_update) == set(keys)
    assert all(is_shared(k, "full_private") for k in keys)
    assert all(set(v) == set(keys) for v in source_updates)
    assert all(torch.isfinite(v).all() for d in source_updates+[target_update] for v in d.values())
    groups = sorted({group(k) for k in keys})
    blocks = [[k for k in keys if group(k) == g] for g in groups]
    uniform = {k: sum(v[k] for v in source_updates)/3 for k in keys}
    weighted, simplex = mgda_weights(source_updates, keys)
    reference = remove_conflicting_parallel(uniform, target_update, blocks)
    candidate = remove_conflicting_parallel(weighted, target_update, blocks)
    correction, energy_audit = match(candidate, reference, blocks)
    cosines = [dot(correction, target_update, block)/max(
        (energy(correction, block)*energy(target_update, block))**.5, 1e-20) for block in blocks]
    conflict_error = max(max(-cosine, 0.) for cosine in cosines)
    assert conflict_error < 1e-5
    state = {k: shared[k]+target_update[k]+cfg["beta"]*correction[k] for k in keys}
    assert all(torch.isfinite(v).all() for v in state.values())
    return state, {"simplex": simplex, "energy": energy_audit,
                   "projection_rule": "remove_negative_parallel_v1",
                   "nonconflict_relative_error": conflict_error,
                   "correction_target_cosines": dict(zip(groups, cosines)),
                   "groups": groups, "source_order": [c for c in CITIES if c != cfg["target"]],
                   "target_squared_norm": energy(target_update, keys),
                   "correction_squared_norm": energy(correction, keys)}
