# Experiment setting

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

Shared source updates are normalized and combined by simplex minimum-norm weighting. Within each functional block, their target-collinear component is removed, then energy is matched to the raw source-mean complementary component. The server update is `anchor + target_delta + 0.6 * source_correction`. Private residual weights and their scales are never aggregated.

## Calibration and final adaptation

After round 20, four fit-only probes each run 5 epochs from the same target-private starting state. The gate fits three functional response coefficients with scale 0.3. The inherited finite, nonzero-truth mask applies to this calibration only.

Final adaptation resets the private starting state and trains on all available target support for 50 epochs, batch size 8. Learning rates are 0.0004 for the prediction head, 0.0002 for private parameters, and 0.0001 for the remaining trainable parameters, with the inherited adaptation freezing policy. No early stopping is used.

## Evaluation

The target test interval is the chronological last 20%: `[27417,34272)` for METR-LA, `[41692,52116)` for PEMS-BAY, and `[13824,17280)` for Chengdu/Shenzhen. Test windows are purged inside these intervals. Primary MAE/RMSE include zero-valued observations; masked auxiliary metrics are labeled separately. Error sums and counts are pooled before computing RMSE.

H1/H3/H6 correspond to 5/15/30 minutes for LA/BAY and 10/30/60 minutes for Chengdu/Shenzhen. Report all five main seeds, and paired seeds 0/1 for controls, with sample standard deviation. Historically seen test dates are not a fresh sealed confirmation; the release does not claim such a split.

## Scope

The public entry point reproduces this fixed CAPTA protocol and its source-loss, shared FedAvg, Off, local-context, and support controls. It does not reproduce the additional paper-completion schedules, standalone Target-only, full-model standard baselines, or official native pFedCTP runs. Those algorithms have distinct settings and must not be described as the same CAPTA protocol.
