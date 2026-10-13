"""One entry point for settings checks, synthetic CPU smoke and CAPTA training."""
import argparse
import hashlib
import json
from pathlib import Path
from setting import CITIES, SOURCE_LOSSES, ROOT, load_settings, make_job, describe_job


def smoke():
    import torch
    from capta.models import build_model
    from capta.engine import source_objective
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = build_model().cpu()
    x = torch.randn(3, 12, 7, 3)
    truth = torch.randn(3, 7, 6)
    pred = model(x)
    assert tuple(pred.shape) == (3, 7, 6)
    loss, _ = source_objective(pred, truth, x, "h2", torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(json.dumps({"synthetic_CPU_smoke": "passed", "output_shape": list(pred.shape),
                      "real_data_loaded": False, "experiment_started": False}))


def main():
    parser = argparse.ArgumentParser(description="CAPTA cross-city traffic forecasting")
    parser.add_argument("--target", choices=CITIES, default="shenzhen")
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--support", choices=("full", "half"), default="full")
    parser.add_argument("--source-loss", choices=SOURCE_LOSSES, default="h2")
    parser.add_argument("--aggregation", choices=("tra", "fedavg", "off"), default="tra")
    parser.add_argument("--context", choices=("local", "zero"), default="local")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "configs/dataset.json")
    parser.add_argument("--data-root", type=Path, help="Override the data directory in the manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true", help="Show settings without loading Torch or data")
    parser.add_argument("--smoke", action="store_true", help="Synthetic CPU forward/backward check only")
    args = parser.parse_args()
    base = load_settings(args.config)
    job = make_job(args.target, args.seed, args.support, args.source_loss, args.aggregation, args.context)
    if args.check:
        print(json.dumps({"job": job, "settings": describe_job(base, job),
                          "training_started": False}, indent=2))
        return
    if args.smoke:
        smoke()
        return
    import torch
    from capta import engine
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --smoke for the synthetic CPU check.")
    torch.set_num_threads(4)
    engine.DATASET_MANIFEST = args.dataset.resolve()
    engine.DATA_ROOT = args.data_root.resolve() if args.data_root else None
    cfg = engine.job_config(base, job)
    hashes = engine.identity(args.config.resolve(), job)
    hashes["public_entry_settings"] = hashlib.sha256((ROOT / "setting.py").read_bytes()).hexdigest()
    out = args.output.resolve() / job["job_id"]
    if out.exists() and not (out / (args.target + ".invocations.json")).exists():
        raise FileExistsError(f"Refusing an unrelated existing output directory: {out}")
    print(f"Run: {job['job_id']}\nOutput: {out}", flush=True)
    engine.run_job(cfg, out, args.target, hashes, device=args.device)


if __name__ == "__main__":
    main()
