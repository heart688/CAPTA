"""Synthetic CPU checks only; no traffic arrays, server or GPU required."""
import json
import unittest
from pathlib import Path
import tempfile
import numpy as np
import torch
from setting import load_settings, make_job, describe_job
from capta.models import build_model, regime_weights
from capta.models.context import TrendInnovationProjection
from capta.data import Range, CityData, TrafficWindows, window_starts_inside, load_city
from capta.engine import source_objective, private_state, Client, job_config, evaluate
from capta.utils import is_shared, capture_rng, restore_rng
from capta.aggregation import server_aggregate


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_settings_and_half_boundaries(self):
        job = make_job(support="half")
        cfg = job_config(load_settings(), job)
        self.assertEqual((cfg["support_start"], cfg["support_fit_end"], cfg["support_gate_end"]), (432,648,756))
        self.assertEqual(len(window_starts_inside(Range(432,864),12,6)),415)
        self.assertEqual(len(window_starts_inside(Range(0,864),12,6)),847)
        self.assertEqual(describe_job(load_settings(), job)["support_steps"],432)

    def test_model_sensor_count_and_calendar_independence(self):
        torch.manual_seed(0)
        model = build_model().eval()
        for nodes in (3, 7):
            x = torch.randn(2,12,nodes,3)
            with torch.no_grad():
                y = model(x)
                changed = x.clone(); changed[...,1:] = 123
                torch.testing.assert_close(model(changed), y, rtol=0, atol=0)
            self.assertEqual(tuple(y.shape),(2,nodes,6))
            self.assertIsNone(model._client_candidate_hook.context)

    def test_trend_initialization_preserves_linear(self):
        layer = torch.nn.Linear(12,128)
        x = torch.randn(2,7,12)
        y = layer(x)
        branched = TrendInnovationProjection(layer)
        torch.testing.assert_close(branched(x), y, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(branched.innovation.weight,branched.weight,rtol=0,atol=0)

    def test_h2_manual_nested_loss_and_grad(self):
        torch.manual_seed(3)
        x = torch.randn(3,12,7,3)
        y = torch.randn(3,7,6)
        pred = torch.randn_like(y,requires_grad=True)
        got,_ = source_objective(pred,y,x,"h2",torch.Generator().manual_seed(2))
        error=(pred-y).abs().mean(-1); membership=regime_weights(x).bool()
        states=[]
        for s in range(3):
            windows=[error[b][membership[b,:,s]].mean() for b in range(3) if membership[b,:,s].any()]
            if windows: states.append(torch.stack(windows).mean())
        expected=torch.stack(states).mean()
        torch.testing.assert_close(got,expected,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(got,pred,retain_graph=True)[0],torch.autograd.grad(expected,pred)[0],rtol=0,atol=0)

    def test_all_source_modes_are_finite(self):
        x=torch.randn(5,12,7,3); y=torch.randn(5,7,6)
        for mode in ("ordinary","c5","h2","random_c5","pseudo_h2"):
            pred=torch.randn_like(y,requires_grad=True)
            loss,_=source_objective(pred,y,x,mode,torch.Generator().manual_seed(17))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_shared_boundary_and_persistent_client_optimizer(self):
        torch.manual_seed(0)
        features=np.random.default_rng(0).normal(size=(40,7,3)).astype(np.float32)
        city=CityData("shenzhen",features,Path("unused"),144,10)
        ds=TrafficWindows(city,np.arange(5),12,6,Range(0,30))
        model=build_model(); initial={k:v.detach().clone() for k,v in model.state_dict().items()}
        cfg=job_config(load_settings(),make_job())
        with tempfile.TemporaryDirectory() as folder:
            client=Client("shenzhen",ds,initial,cfg,folder,{},device="cpu")
            shared={k:v for k,v in initial.items() if is_shared(k,"full_private")}
            before=private_state(client.model)
            client.broadcast(shared)
            for k,v in before.items(): torch.testing.assert_close(private_state(client.model)[k],v,rtol=0,atol=0)
            opt=client.opt
            client.update(shared,1)
            client.update(shared,2)
            self.assertIs(opt,client.opt)
            self.assertEqual({int(v["step"]) for v in opt.state.values()},{4})
            self.assertTrue(all(is_shared(k,"full_private") for k in client.shared))

    def test_server_private_rejection_and_complementarity(self):
        keys=("history_embed.0.weight","blocks.0.out.weight","head.weight")
        torch.manual_seed(4)
        anchor={k:torch.zeros(11) for k in keys}
        target={k:torch.randn(11) for k in keys}
        sources=[{k:torch.randn(11) for k in keys} for _ in range(3)]
        result,audit=server_aggregate(anchor,sources,target,{"target":"shenzhen","beta":.6})
        self.assertLess(audit["orthogonality_relative_error"],1e-5)
        for k in keys:
            self.assertTrue(torch.isfinite(result[k]).all())
            self.assertAlmostEqual(float(((result[k]-target[k])*target[k]).sum()),0,places=5)
        with self.assertRaises(AssertionError):
            server_aggregate({"blocks.0.private.weight":torch.zeros(2)},[],{}, {"target":"shenzhen","beta":.6})

    def test_rng_restore(self):
        state=capture_rng(); a=torch.rand(5); n=np.random.rand(5)
        restore_rng(state)
        torch.testing.assert_close(torch.rand(5),a,rtol=0,atol=0)
        np.testing.assert_array_equal(np.random.rand(5),n)

    def test_checkpoint_round_restore_next_update(self):
        torch.manual_seed(0)
        features=np.random.default_rng(0).normal(size=(40,3,3)).astype(np.float32)
        data=CityData("shenzhen",features,Path("unused"),144,10)
        ds=TrafficWindows(data,np.arange(3),12,6,Range(0,30))
        initial={k:v.detach().clone() for k,v in build_model().state_dict().items()}
        shared={k:v for k,v in initial.items() if is_shared(k,"full_private")}
        cfg=job_config(load_settings(),make_job())
        with tempfile.TemporaryDirectory() as folder:
            first=Client("shenzhen",ds,initial,cfg,folder,{},"cpu")
            first.update(shared,1)
            path=Path(folder)/"synthetic.pt"
            torch.save(first.snapshot(1),path)
            restored=Client("shenzhen",ds,initial,cfg,folder,{},"cpu")
            restored.restore(torch.load(path,map_location="cpu",weights_only=True),1)
            delta_a,_=first.update(shared,2)
            delta_b,_=restored.update(shared,2)
            for k in delta_a:torch.testing.assert_close(delta_a[k],delta_b[k],rtol=0,atol=0)

    def test_evaluation_pools_squared_errors_not_batch_rmse(self):
        from types import SimpleNamespace
        class FixedData(torch.utils.data.Dataset):
            mean=0.;std=1.
            def __len__(self):return 2
            def __getitem__(self,i):
                return torch.zeros(12,3,3),torch.full((3,6),2. if i==0 else 4.),torch.zeros(3,21)
        class FixedModel(torch.nn.Module):
            def forward(self,x,profile):return torch.ones(x.shape[0],3,6)
        client=SimpleNamespace(model=FixedModel(),cfg={"target_batch_size":1},device="cpu")
        scores,_=evaluate(client,FixedData())
        np.testing.assert_allclose(scores["primary_unmasked"]["mae"],2.)
        np.testing.assert_allclose(scores["primary_unmasked"]["rmse"],np.sqrt(5.))

    def test_series_only_loader_hash_and_training_stats(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            raw=np.ones((40,3,2),dtype=np.float32);raw[20:,:,0]=100
            p=Path(folder)/"dataset.npy";np.save(p,raw)
            manifest={"root":folder,"strict_hash":True,"cities":{"shenzhen":{"series_file":"dataset.npy","expected_shape":[40,3,2],"steps_per_day":144,"interval_minutes":10,"speed_channel":0,"time_of_day_channel":1,"series_sha256":hashlib.sha256(p.read_bytes()).hexdigest()}}}
            city=load_city(manifest,"shenzhen")
            ds=TrafficWindows(city,window_starts_inside(Range(0,20),12,6),12,6,Range(0,20))
            self.assertEqual(ds.mean,1.0)
            self.assertFalse(city.graph_path.exists())
            manifest["cities"]["shenzhen"]["series_sha256"]="0"*64
            with self.assertRaises(ValueError):load_city(manifest,"shenzhen")


if __name__ == "__main__":
    unittest.main()
