# CAPTA

PyTorch code for graph-free cross-city traffic forecasting. This repository contains the CAPTA backbone, source objective, shared-update aggregation, calibration, and target adaptation used in our experiments.

The accompanying paper is being prepared for submission to **The Web Conference (WWW)**. Paper title, author information, and citation will be added when the manuscript is public. This is not an acceptance announcement.

## Installation

Use Python 3.10 on Linux for training. Install the CUDA-enabled PyTorch build appropriate for your machine; our training implementation uses PyTorch 2.1.2 and NumPy 1.26.4.

```bash
conda create -n capta python=3.10
conda activate capta
pip install -r requirements.txt
```

## Data preparation

Data are not included. Obtain the datasets from their original providers and respect their access and redistribution terms. The code expects the processed NumPy arrays described in [data/README.md](data/README.md), not arbitrary raw files.

Place the arrays under `data/<city>/dataset.npy`, or pass `--data-root /path/to/data`. Check the shape, channel definitions, and SHA256 in `configs/dataset.json`. CAPTA does not use an adjacency matrix.

## Quick start

Inspect a setting without loading data or starting training:

```bash
python main.py --target shenzhen --seed 0 --check
```

Check the backbone and source loss on synthetic data, on CPU:

```bash
python main.py --smoke
python -m unittest discover -s tests -v
```

Train CAPTA with full target support:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --target shenzhen --seed 0
```

Supported targets are `metr-la`, `pems-bay`, `chengdu`, and `shenzhen`. The main setting uses seeds 0–4. Run one seed at a time; retain every seed rather than selecting the best result.

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 python main.py --target shenzhen --seed "$seed"
done
```

The result and resumable checkpoints are saved under `outputs/<job_id>/`. Repeating the same command resumes its matching committed checkpoint or reuses a completed result. Changing settings or code requires a new output location; old checkpoints must not be relabeled as matching a new run.

## Settings and controls

The default settings are in [configs/default.json](configs/default.json); [SETTING.md](SETTING.md) explains the splits, budgets, and evaluation. A few examples:

```bash
# Right-aligned half support: [432, 864), with fresh model/optimizer/statistics.
python main.py --target shenzhen --seed 0 --support half

# Source-loss controls; target learning remains ordinary L1.
python main.py --target shenzhen --seed 0 --source-loss ordinary
python main.py --target shenzhen --seed 0 --source-loss c5

# Equal-client shared FedAvg control with the same target adaptation.
python main.py --target shenzhen --seed 0 --aggregation fedavg

# Client-local context control.
python main.py --target shenzhen --seed 0 --context zero
```

Use paired seeds 0/1 for controls. `off` removes the source correction but still executes source training; it is **not** Target-only. This initial release focuses on CAPTA and its matched controls. It does not contain all external baseline implementations or the internal batch scheduler.

## Code

- `capta/models/`: variable-token backbone, private residuals, trend/innovation projection, and observed-window context.
- `capta/engine.py`: source objectives, city clients, calibration, final adaptation, evaluation, and checkpoint transactions.
- `capta/aggregation.py`: minimum-norm source weighting, target-complementary projection, and block energy matching.
- `capta/data.py`: chronological windows and training-range normalization.
- `main.py`, `setting.py`, `configs/`: public entry point and fixed experiment settings.

This is a repackaged implementation, not a byte-identical historical launch environment. `SOURCE_PROVENANCE.json` records the source extraction and integration changes. Parameters, scientific function bodies, training budgets, and evaluation conventions are preserved. Server access records, datasets, checkpoints, and private experiment outputs are not distributed.

## Reporting

Primary metrics are unmasked MAE and RMSE in original speed units, over the complete last 20% of the target series. H1/H3/H6 form the six reported cells. Report all seeds and sample standard deviation (`ddof=1`). Do not average batch RMSE or use test scores to select settings.

The engine's elapsed time is diagnostic, not a CUDA-synchronized efficiency benchmark. Communication fields are logical FP32 shared-parameter payloads, excluding serialization/network overhead. This release does not claim to reproduce the separate 12-run efficiency measurement pipeline or all native pFedCTP baseline instrumentation.

## Citation and license

Citation information will be added with the public manuscript. The project license still needs to be chosen by the authors before publication; no third-party license is inferred from the repository layout reference.
