"""Residual-router backbone and adaptation policy."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .base import StrongTransferForecaster, DynamicLatentRouter
CANDIDATES=('old_anchor',)+tuple('no_calendar'+('_residual' if p else '')+('_dynamic' if g else '')+('_kernel' if k else '') for p in (False,True) for g in (False,True) for k in (False,True))

class ZeroCalendar(nn.Module):
    def forward(self,x): return x.new_zeros(x.shape[0],128)

class IncrementalRouter(DynamicLatentRouter):
    def forward(self,x,signature=None):
        z=self.norm1(x); routed=z if signature is None else z+signature
        if self.use_kernel:
            q=F.elu(self.kernel_query(routed))+1; k=F.elu(self.kernel_key(routed))+1
            moment=torch.einsum('bnr,bnd->brd',k,self.value(routed))
            context=torch.einsum('bnr,brd->bnd',q,moment)/torch.einsum('bnr,br->bn',q,k.sum(1)).unsqueeze(-1).clamp_min(1e-6)
        else:
            assignment=torch.softmax(self.assign(routed),-1)
            slots=torch.einsum('bnk,bnd->bkd',assignment,self.value(routed))/assignment.sum(1).clamp_min(1e-6).unsqueeze(-1)
            slots=slots+self.drop(self.role_mixer(slots))
            q=self.query(routed); k=self.role_key(slots); v=self.role_value(slots)
            read=torch.softmax(torch.einsum('bnd,bkd->bnk',q,k)/math.sqrt(q.shape[-1]),-1)
            context=torch.einsum('bnk,bkd->bnd',read,v)
        gate=torch.sigmoid(self.agreement_gate(torch.cat([z,context,z-context,torch.zeros_like(z)],-1))) if self.use_dynamic else torch.sigmoid(self.spatial_scale)
        x=x+gate*self.drop(self.out(context))
        if self.use_private:
            x=x+torch.sigmoid(self.private_scale)*self.private(torch.cat([z,torch.zeros_like(z)],-1))
        return x+self.ffn(self.norm2(x))

class IncrementalForecaster(StrongTransferForecaster):
    def forward(self,data,profile=None):
        speed=data[...,0].transpose(1,2); mean=speed.mean(-1,keepdim=True)
        std=speed.std(-1,keepdim=True,unbiased=False).clamp_min(1e-4)
        normalized=(speed-mean)/std; x=self.history_embed(normalized)
        tod=data[:,-1,:,1].mean(1); dow=data[:,-1,:,2].mean(1)
        calendar=torch.stack([torch.sin(2*math.pi*tod),torch.cos(2*math.pi*tod),torch.sin(2*math.pi*dow),torch.cos(2*math.pi*dow)],-1)
        x=x+self.calendar_embed(calendar).unsqueeze(1)
        if self.change=='add_profile': x=x+torch.sigmoid(self.profile_scale)*self.profile_embed(profile)
        signature=None
        if self.change=='add_state':
            recent=torch.stack([normalized[...,-1],normalized[...,-1]-normalized[...,0],normalized.diff(dim=-1).abs().mean(-1)],-1)
            signature=self.current_traffic_embed(recent)
        for block in self.blocks: x=block(x,signature)
        return self.head(self.final_norm(x))*std+mean

def source_arch(candidate):
    return 'old_anchor' if candidate=='tcc_freeze' else candidate

def build(candidate,horizon=12):
    if candidate not in CANDIDATES: raise ValueError(candidate)
    model=StrongTransferForecaster(variant='dynamic',history=12,horizon=horizon,dim=128,heads=4,ff_dim=256,layers=3,roles=16,dropout=.1,use_profile=False)
    if candidate=='old_anchor': return model
    model.calendar_embed=ZeroCalendar()
    model.__class__=IncrementalForecaster; model.change=candidate
    for index,block in enumerate(model.blocks):
        block.__class__=IncrementalRouter; block.change=candidate
        block.use_private='_residual' in candidate
        block.use_dynamic='_dynamic' in candidate
        block.use_kernel='_kernel' in candidate
        if block.use_private:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+1000+index)
                block.private=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,256),nn.GELU(),nn.Dropout(.1),nn.Linear(256,128))
                block.private_scale=nn.Parameter(torch.tensor(-2.))
        if block.use_dynamic:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+2000+index)
                del block.spatial_scale; block.agreement_gate=nn.Linear(512,1)
                nn.init.zeros_(block.agreement_gate.weight); nn.init.constant_(block.agreement_gate.bias,-2.)
        if block.use_kernel:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed()+3000+index)
                block.kernel_query=nn.Linear(128,16); block.kernel_key=nn.Linear(128,16)
            for name in ('assign','query','role_key','role_value','role_mixer'): delattr(block,name)
    return model

def adapt_policy(model,candidate):
    if candidate=='tcc_freeze':
        for name,p in model.named_parameters(): p.requires_grad=name.startswith(('final_norm.','head.')) or '.norm1.' in name or '.norm2.' in name
        return
    frozen={'assign','value','role_key','role_value','query','role_mixer','out','spatial_scale','gate','agreement_gate','kernel_query','kernel_key'}
    for name,p in model.named_parameters(): p.requires_grad=not bool(set(name.split('.')) & frozen)
