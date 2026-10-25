"""Training, calibration, checkpoint and RNG helpers."""
from __future__ import annotations
import copy
import hashlib
import json
import pickle
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models.residual import adapt_policy

def seed(n):
    random.seed(n);np.random.seed(n);torch.manual_seed(n)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(n)


def is_shared(name,variant='full_private'):
    if variant != 'full_private':
        raise ValueError("CAPTA keeps city-private residual parameters local")
    local_residual=('.private.' in name or name.endswith('.private_scale'))
    return not local_residual


def train(model,ds,device,nseed,epochs,adapt=False):
    if adapt:
        adapt_policy(model,'no_calendar_residual')
    seed(nseed)
    loader=DataLoader(ds,batch_size=8,shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(nseed))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=.0001 if adapt else .001,weight_decay=.0001)
    model.train();losses=[]
    for _ in range(epochs):
        for x,y,profile in loader:
            optimizer.zero_grad(set_to_none=True)
            loss=(model(x.to(device),profile.to(device))-y.to(device)).abs().mean()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();losses.append(float(loss.detach()))
    return losses


def save(out,result):
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(out)

def six(m):return np.array([m[k][h] for k in ('mae','rmse') for h in (0,2,5)])


def group(k):
    if k.startswith(('head.','final_norm.')):return 'prediction'
    if k.startswith('history_embed.') or '.ffn.' in k or '.norm' in k:return 'temporal'
    return 'interaction'

def solve(features,error,mask,scales,target_columns):
    """Fixed constrained linear-response MAE+RMSE objective on gate only."""
    valid=mask.reshape(-1,3);err=error.reshape(-1,3)
    norm=scales[1]
    design=np.stack(features,axis=-1).reshape(-1,3,len(features))/norm[None,:,None]
    err=torch.tensor(err/norm[None,:],dtype=torch.float32)
    design=torch.tensor(design,dtype=torch.float32)
    valid=torch.tensor(valid,dtype=torch.float32);count=valid.sum(0)
    ratios=torch.tensor(scales[0]/norm,dtype=torch.float32)
    weights=torch.zeros(len(features),requires_grad=True)
    opt=torch.optim.Adam([weights],lr=.05)
    for _ in range(100):
        opt.zero_grad(set_to_none=True)
        residual=err-torch.einsum('nhk,k->nh',design,weights)
        mae=(residual.abs()*valid).sum(0)/count/ratios
        rmse=torch.sqrt((residual.square()*valid).sum(0)/count+1e-12)
        loss=.5*(mae+rmse).mean()+.01*weights.square().mean()
        loss.backward();opt.step()
        with torch.no_grad():weights.clamp_(-1,1)
    assert torch.isfinite(weights).all()
    return weights.detach().numpy()

def optimizer(model, recipe='private_lr2'):
    if recipe != 'private_lr2':
        raise ValueError("CAPTA uses the fixed head/private/rest adaptation rates")
    adapt_policy(model, 'no_calendar_residual')
    groups = {'head': [], 'rest': [], 'special': []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith('head.'):
            groups['head'].append(p)
        elif '.private.' in name or name.endswith('.private_scale'):
            groups['special'].append(p)
        else:
            groups['rest'].append(p)
    param_groups = [dict(params=groups['head'], lr=4e-4), dict(params=groups['rest'], lr=1e-4)]
    if groups['special']:
        param_groups.append(dict(params=groups['special'], lr=2e-4, weight_decay=1e-4))
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    return opt, [n for n, p in model.named_parameters() if p.requires_grad]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def log(row):
    print(json.dumps(row), flush=True)

def tree_sha(value):
    digest = hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.dtype, item.shape)).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        else:
            digest.update(pickle.dumps(item, protocol=4))
    visit(value)
    return digest.hexdigest()


def exact_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    return left == right


def capture_rng():
    return {"python": copy.deepcopy(random.getstate()), "numpy": copy.deepcopy(np.random.get_state()),
            "cpu": torch.get_rng_state().clone(),
            "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
