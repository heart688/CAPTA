# Experiment settings

[Back to README](../README.md)

Each city is a logical client. A run uses the target city and the other three source cities; raw samples, private residual parameters, and optimizer states remain client-local. The current implementation simulates these clients in one process.

## Backbone

History length is 12 and forecast horizon is 6. Each sensor history becomes a 128-dimensional token. The backbone has three residual-router blocks, 16 latent roles, a 256-dimensional feed-forward layer, and dropout 0.1. No adjacency matrix, sensor-ID embedding, or calendar embedding is used by CAPTA.

The history projection has trend and innovation branches. Innovation weights start as a copy of the history-projection weights, preserving the common initialization. Each block also has a private residual. Its observed-window mean/std context is tanh-bounded and fills the existing private context slots; this context is not uploaded.

## Data splits

Sources use their own chronological first 80%, without a window cap. Full target support is `[0,864)`; half support is `[432,864)`, not the first half. Every 12+6 window must lie entirely within its partition.

Full support uses fit `[0,432)`, calibration gate `[432,648)`, and reserved validation `[648,864)`: 415/199/199 windows. Final adaptation uses all 847 legal support windows.

Half support uses fit `[432,648)`, gate `[648,756)`, and reserved validation `[756,864)`: 199/91/91 windows. Final adaptation uses all 415 legal support windows. The model, optimizer, private state, and statistics are freshly initialized for a half-support run.

864 steps are 3 days for METR-LA/PEMS-BAY (5-minute intervals), and 6 days for Chengdu/Shenzhen (10-minute intervals). Half support is 1.5/3 days respectively.

Source normalization uses source training only. Target FL and calibration use fit statistics; final adaptation and test denormalization use the available support statistics only. Reserved validation is not used for early stopping or hyperparameter selection. The calibration gate fits the prescribed response coefficients; it is not an untouched test set.

## Training

All four clients participate in 20 federated rounds. Each performs 2 local epochs per round, batch size 32, with persistent AdamW: learning rate 0.001, weight decay 0.0001, gradient clipping 5, FP32. Broadcasting shared parameters does not reset client private weights or optimizer state.

The source objective `h2` is a nested mean: node error within each observed state and real window, then nonempty windows within the state, then nonempty states. It is not the sum of two losses. Stable/rise/fall states use the observed lookback's normalized half-window change, with threshold ±0.1; they are not ground-truth congestion labels. Target updates use ordinary normalized L1 over all six horizons.

Shared source updates are normalized and combined by simplex minimum-norm weighting. Within each functional block, only the component parallel and opposite to the target update is removed. Same-direction parallel and orthogonal components are retained:

```text
source_safe = source - min(dot(source, target) / dot(target, target), 0) * target
```

The dot products cover every shared tensor in that functional block, not individual parameters or the entire model at once. When the target block is zero, the source block is unchanged because no target direction is defined.

The retained candidate is then positively rescaled to match the energy of the same-round raw source arithmetic mean after the same negative-parallel filter. This preserves the candidate's retained direction, not its original magnitude. A zero candidate stays zero; if its reference is nonzero, the audit explicitly records the unmatched reference rather than inventing a direction. The server update remains `anchor + target_delta + 0.6 * source_correction`. Private residual weights and their scales are never aggregated.

## Calibration and final adaptation

After round 20, four fit-only probes each run 5 epochs from the same target-private starting state. The gate fits three functional response coefficients with scale 0.3. The inherited finite, nonzero-truth mask applies to this calibration only.

Final adaptation resets the private starting state and trains on all available target support for 50 epochs, batch size 8. Learning rates are 0.0004 for the prediction head, 0.0002 for private parameters, and 0.0001 for the remaining trainable parameters, with the inherited adaptation freezing policy. No early stopping is used.

## Evaluation

The target test interval is the chronological last 20%: `[27417,34272)` for METR-LA, `[41692,52116)` for PEMS-BAY, and `[13824,17280)` for Chengdu/Shenzhen. Test windows are purged inside these intervals. Primary MAE/RMSE include zero-valued observations; masked auxiliary metrics are labeled separately. Error sums and counts are pooled before computing RMSE.

H1/H3/H6 correspond to 5/15/30 minutes for LA/BAY and 10/30/60 minutes for Chengdu/Shenzhen. Report all five main seeds, and paired seeds 0/1 for controls, with sample standard deviation. Historically seen test dates are not a fresh sealed confirmation; the release does not claim such a split.

## Scope

The public entry point reproduces this fixed CAPTA protocol and its source-loss, shared FedAvg, Off, local-context, and support controls. It does not reproduce the additional paper-completion schedules, standalone Target-only, full-model standard baselines, or official native pFedCTP runs. Those algorithms have distinct settings and must not be described as the same CAPTA protocol.

### Optional controls

The README focuses on the main CAPTA run. The existing entry point also supports the following matched controls, using paired seeds 0/1:

```bash
# Right-aligned half support: [432, 864), with a fresh model and optimizer.
python main.py --target shenzhen --seed 0 --support half

# Source objectives; target updates remain ordinary L1.
python main.py --target shenzhen --seed 0 --source-loss ordinary
python main.py --target shenzhen --seed 0 --source-loss c5

# Shared FedAvg and zero local-context controls.
python main.py --target shenzhen --seed 0 --aggregation fedavg
python main.py --target shenzhen --seed 0 --context zero
```

`--aggregation off` removes the source correction but still executes source training; it is not Target-only.

## Release notes

The model and training implementation were repackaged from the experimental code. [source_provenance.json](source_provenance.json) records the extraction, integration changes, and subsequent aggregation revision; this repository is not a byte-identical historical launch environment. Server access records, datasets, checkpoints, and private experiment outputs are not distributed.

**Aggregation revision (2026-10-16):** the current rule is `remove_negative_parallel_v1`. The initial public release removed the full parallel component, including positive alignment. This revision retains positive alignment and perpendicular components, removing only negative alignment; execution and geometry audits use the same revised rule. Existing historical results have not been rerun or relabeled as results of this revision. Code hashes prevent old checkpoints/results from being reused under the new implementation; use a fresh output directory for any future revised run.

The engine's elapsed time is diagnostic, not a CUDA-synchronized efficiency benchmark. Communication fields describe logical FP32 shared-parameter payloads, excluding serialization and network overhead. This release does not include the separate efficiency-measurement pipeline or native pFedCTP instrumentation.

Report every seed and sample standard deviation (`ddof=1`). Do not average batch RMSE or use test scores to select settings.

No project license has been selected yet. Public availability does not itself grant an open-source license; the authors need to choose one separately.
