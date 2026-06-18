# Phase 1 BCR Smoke Result

Remote run date: 2026-06-19

## Setup

```text
model: Qwen/Qwen2.5-7B-Instruct
method: Wanda(C4)
sparsity: 30% unstructured
calibration: C4 streaming
eval data: HH-RLHF, 50 examples
reference: Qwen/Qwen2.5-7B base
metric: base-reference-normalized BCR
```

## Result

```json
{
  "actual_sparsity": 0.2999573783745782,
  "bcr@0": 0.1111111111111111,
  "bcr@q25": 0.05,
  "bcr@q50": 0.0,
  "bcr@q75": 0.0,
  "coverage@0": 0.54,
  "coverage@q25": 0.4,
  "coverage@q50": 0.26,
  "coverage@q75": 0.14,
  "mean_delta_dense": 0.03750720431071156,
  "mean_delta_pruned": 0.02954568813332231,
  "mean_margin_drop": 0.007961516177389258,
  "preference_accuracy_dense": 0.54,
  "preference_accuracy_pruned": 0.54,
  "num_examples": 50
}
```

## Interpretation

The BCR pipeline is operational. On this tiny smoke set, 30% Wanda shows a small
boundary-crossing signal (`BCR@q25 = 0.05`) and a positive mean margin drop.
This is not a Phase 1 decision result because `n=50` is too small. Proceed to
the 1000-example BCR band at 30%, 40%, and 50%.

