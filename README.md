<div align="center">

# CAPTA

**Graph-free cross-city traffic forecasting**

PyTorch implementation of the CAPTA model and training protocol.

[Installation](#installation) · [Datasets](#datasets) · [Quick start](#quick-start) · [Configuration](#configuration)

</div>

## Overview

CAPTA (Composition-Aware Personalized Traffic Adaptation) transfers knowledge from data-rich source cities to a target city with limited observations, without requiring an adjacency matrix. This release contains the main CAPTA model, its training and evaluation pipeline, and the default configuration.

TEB balances traffic states and real observation windows during source training. TCA preserves the target update, retains aligned and perpendicular source components, and removes only opposing parallel components. See [training details](docs/settings.md#federated-training).

## Installation

Use **Python 3.10**, **PyTorch 2.1.2**, and **NumPy 1.26.4**. Training is intended for Linux with an NVIDIA GPU.

```bash
git clone https://github.com/heart688/CAPTA.git
cd CAPTA
conda create -n capta python=3.10
conda activate capta
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with your GPU and driver.

## Datasets

Please obtain the datasets directly from their official providers:

- **METR-LA and PEMS-BAY:** the [DCRNN authors' dataset page](https://github.com/liyaguang/DCRNN#data-preparation).
- **DiDi-Chengdu and DiDi-Shenzhen:** request the historical GAIA traffic-index datasets through [DiDi's official data portal](https://outreach.didichuxing.com/opendata/datasets). These benchmark files are not currently listed as direct downloads; see [access notes](docs/data.md#official-sources).

Prepare the local arrays as described in [Data preparation](docs/data.md). Provider downloads are not necessarily in the input format expected by this code; no datasets or raw-data converters are distributed here.

```text
data/
├── metr-la/dataset.npy
├── pems-bay/dataset.npy
├── chengdu/dataset.npy
└── shenzhen/dataset.npy
```

Each array has shape `[time, node, 2]`: traffic speed and time-of-day. CAPTA does not load a road graph. The [dataset manifest](configs/dataset.json) specifies shapes, channels, and checksums for the experimental arrays.

## Quick start

Check the configuration without loading data or starting training:

```bash
python main.py --target shenzhen --seed 0 --check
```

Train and evaluate CAPTA:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --target shenzhen --seed 0
```

Choose `--target` from `metr-la`, `pems-bay`, `chengdu`, and `shenzhen`. The other three cities serve as sources. Use `--data-root /path/to/data` if your arrays are stored elsewhere.

The main experiments use seeds **0–4**. Results and resumable checkpoints are written to `outputs/<job_id>/`. Repeating an unchanged command resumes a matching checkpoint or reuses its completed result; use a new `--output` directory when changing code or settings.

<details>
<summary>CPU sanity checks (no datasets required)</summary>

```bash
python main.py --smoke
python -m unittest discover -s tests -v
```

These use synthetic inputs only and do not start a traffic experiment.

</details>

## Configuration

- [configs/default.json](configs/default.json): model and training settings.
- [configs/dataset.json](configs/dataset.json): local data layout and integrity checks.
- [docs/settings.md](docs/settings.md): data splits, optimization, adaptation, and evaluation details.

The default protocol uses a **12-step lookback**, **6-step forecast**, **20 federated rounds × 2 local epochs**, and **50 final target-adaptation epochs**. Evaluation reports unmasked MAE and RMSE at H1/H3/H6 in original speed units.

Each target supplies **864 support steps**: 3 days for METR-LA/PEMS-BAY or 6 days for Chengdu/Shenzhen. The other three cities use the first 80% of their observations for source training; evaluation uses the target's chronological last 20%.

## Repository structure

```text
CAPTA/
├── capta/             # Model and training implementation
│   ├── models/        # Backbone, private residuals, and local context
│   ├── aggregation.py # Shared-update aggregation
│   ├── data.py        # Windows and normalization
│   └── engine.py      # Training, adaptation, and evaluation
├── configs/           # Model/training settings and data manifest
├── docs/              # Data preparation and protocol details
├── tests/             # Synthetic CPU tests
├── main.py            # Run CAPTA
├── setting.py         # Configuration loading and validation
└── requirements.txt
```

## Citation

Paper and citation information will be added when the manuscript is public.
