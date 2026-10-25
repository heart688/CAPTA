# Main CAPTA settings

[Back to README](../README.md)

This release contains the main CAPTA model and its training and evaluation pipeline. Each run uses one target city and the other three cities as sources. Clients are simulated in one process; raw samples, private residual parameters, and optimizer states remain within the logical clients.

## Model

History length is 12 and forecast horizon is 6. Each sensor history becomes a 128-dimensional token. The backbone has three residual-router blocks, 16 latent slots, a 256-dimensional feed-forward layer, and dropout 0.1. CAPTA uses no adjacency matrix, sensor-ID embedding, or calendar embedding.

History is standardized per window and decomposed with a three-step moving average. The trend and innovation projections start with equal weights and learn independently. Each block has a city-private residual conditioned on the observed window's mean and population standard deviation. The statistics are tanh-bounded and repeated into the existing 128-dimensional private context slot.

## Data partitions

Sources use every legal window in their chronological first 80%. Target support is `[0,864)`: 3 days for METR-LA/PEMS-BAY at 5-minute intervals, or 6 days for Chengdu/Shenzhen at 10-minute intervals.

Target support is partitioned into fit `[0,432)`, calibration gate `[432,648)`, and support validation `[648,864)`, giving 415/199/199 legal windows. Each 12+6 window stays within its partition. Final adaptation uses all 847 legal support windows, including windows crossing the internal support partitions.

Source normalization uses source training observations only. Target federated training and calibration use fit statistics. Final adaptation and test denormalization use statistics recomputed from the complete target support. Support validation is not used for early stopping. The calibration gate fits prescribed response coefficients; it is part of the permitted support.

## Federated training

All four clients participate in 20 rounds, with 2 local epochs per round and batch size 32. Persistent AdamW uses learning rate 0.001, weight decay 0.0001, gradient clipping 5, and FP32. Broadcasting shared parameters preserves each client's private weights and optimizer state.

TEB (`h2` internally) averages sensor errors within each observed state and real window, then nonempty windows within each state, then nonempty states. Stable/rise/fall assignments use only the observed lookback's normalized half-window change, with threshold ±0.1. Target clients use ordinary normalized L1 across all six forecast steps.

TCA (`tra` internally) normalizes complete source updates and combines them by simplex minimum-norm weighting. Within each temporal, interaction, or prediction parameter group, it removes only the component parallel and opposite to the target update:

```text
source_safe = source - min(dot(source, target) / dot(target, target), 0) * target
```

The retained candidate is positively rescaled to the norm of the same-round raw source arithmetic mean after the same conditional projection. The server update is `anchor + target_delta + 0.6 * source_correction`. Private residual weights and their scales are never aggregated.

Source normalization uses an L2-norm floor of `1e-20`. As specified in the paper's Appendix D, each functional group's target-update, projected-candidate, and energy-reference squared norms must be strictly greater than `1e-20`; otherwise training stops with `FloatingPointError`. This prevents undefined projection or energy matching.

## Calibration and final adaptation

After round 20, four independent fit-only probes start from the same round-20 private state and train for 5 epochs with fresh AdamW at `1e-4`. The baseline uses the final shared parameters; the other probes add `0.3 * last_target_delta` to one functional group each. Gate predictions fit three response coefficients, constrained to `[-1,1]`, with 100 Adam steps at 0.05 and regularization 0.01. Gate fitting alone uses the finite, nonzero-original-truth mask.

Final adaptation restores the round-20 private state and uses all target support for 50 epochs, batch size 8, with fresh AdamW. Routing parameters stay frozen. Learning rates are `4e-4` for the prediction head, `2e-4` for private parameters, and `1e-4` for other trainable parameters. No early stopping is used.

## Evaluation and running

The target test interval is the chronological last 20%: `[27417,34272)` for METR-LA, `[41692,52116)` for PEMS-BAY, and `[13824,17280)` for Chengdu/Shenzhen. Test windows stay entirely inside these intervals. Primary MAE/RMSE include zero-valued observations; squared errors and counts are pooled before computing RMSE. H1/H3/H6 correspond to 5/15/30 minutes for LA/BAY and 10/30/60 minutes for Chengdu/Shenzhen.

Use seeds 0-4 for the main protocol. Checkpoints include code/config/data identity and optimizer/RNG state. Changing code or settings requires a fresh output directory; old outputs are not silently reused. Engine elapsed time is diagnostic.
