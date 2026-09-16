# `modeling_Cesar`: Changes

Branched from `modeling-hassan`. These changes fix how models are evaluated and how inputs reach the backbone, and add a paired sweep to measure the effect.

## 1. Deploy threshold chosen on val, not test
- **Change:** New `modeling/metrics.py`. `evaluate_baseline(deploy_threshold=...)` applies a threshold selected on val to the test set. The sweep and `train_vertex` now pass it in.
- **Why:** The old operating point tuned its threshold on the test labels it was scored against. The 0.80 sensitivity floor was therefore always met, and specificity and precision came out optimistic.
- **Benefit:** Test metrics are honest, and a config whose sensitivity actually falls below the floor now shows `deploy_passed_floor=False`. The old test-tuned point is still reported (`op_*`) for comparison with earlier sweeps.

## 2. Per-dataset metrics and a dataset-identity baseline
- **Change:** AUROC, sensitivity, specificity and precision are reported separately for each dataset. `dataset_prior_auroc` reports the AUROC of a score that is just each dataset's train positive rate.
- **Why:** CBIS is about 87% positive and RSNA about 2%, and the images look different (film vs. digital). Pooled AUROC can be earned just by recognizing which dataset an image came from.
- **Benefit:** Shows how much of the headline AUROC is real signal and how much is dataset identity.

## 3. Correct input range for MobileNetV2
- **Change:** New default `input_adapter="rescale_repeat"`: rescale [0,1] to [-1,1], then repeat the grayscale channel three times. The old random 1×1 conv is kept as `"conv1x1"`.
- **Why:** MobileNetV2 was pretrained on [-1,1] inputs. The old conv had no bias, so it could never shift inputs into that range, and a random init could invert channels.
- **Benefit:** The pretrained features see the input distribution they expect. This is likely the largest single accuracy gain, and the model still converts to TF Lite.

## 4. Option to select checkpoints on the deploy objective
- **Change:** A `spec_at_sens` metric (specificity at 0.80 sensitivity) is added, plus a `monitor` setting (`val_auc` by default, or `val_spec_at_sens`). Unknown values fail immediately.
- **Why:** AUROC averages over sensitivity ranges we never operate in.
- **Benefit:** Checkpoints and early stopping can target the metric we actually deploy on.

## 5. Quality-failed images dropped from training
- **Change:** `_build_dataset(drop_failed_quality=True)` is used for training. Rejected images are logged with the reason. Evaluation keeps them so predictions stay aligned row by row with the split table.
- **Why:** Failed images were fed as blank frames that still carried their real label.
- **Benefit:** Removes that label noise.

## 6. Sweep driver
- **Change:** New `input_fixes` sweep comparing three configs on the same job, splits and seed: `legacy`, `rescale`, `rescale_specmon`. Val predictions are computed once and reused for the threshold and calibration. Ranking puts floor-passing configs first, then sorts by deploy specificity. Failed rows keep their config settings. New settings are validated before the dataset build.
- **Benefit:** A trustworthy A/B of these changes, less repeated val preprocessing, and failures that are easier to diagnose.

## Tests
- Added `tests/test_metrics.py`, which needs no TensorFlow.
- Added `tests/test_model_input.py`, which checks the input adapter, the monitor metrics and quality filtering. `build_baseline(weights=None)` lets it run offline.
- Extended `tests/test_sweep_validation.py` with checks for the new settings.
- **Status:** all changed files compile, but the tests have **not been run yet**. The local 64-bit TensorFlow environment was still installing.

## Caveats
- `rescale_repeat` is now the default. Results from the `seeds` or `resolution` sweeps run from this branch are not directly comparable with full-gpu-01/02; use `input_fixes` for that comparison.
- The `legacy` config does not exactly match past runs, because quality-failed training images are now dropped in every config.
