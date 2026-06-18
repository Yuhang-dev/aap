# Phase 1 BCR Band Result

Remote run date: 2026-06-19

## Setup

```text
model: Qwen/Qwen2.5-7B-Instruct
method: Wanda(C4)
sparsity: 30%, 40%, 50% unstructured
calibration: C4 streaming
eval data: HH-RLHF, 1000 examples
reference: Qwen/Qwen2.5-7B base
metric: base-reference-normalized BCR
saved checkpoints: outputs/phase1/pruned_models/
```

Dense-reference coverage was identical across pruning points:

```text
Coverage@0   = 0.542
Coverage@q25 = 0.406
Coverage@q50 = 0.271
Coverage@q75 = 0.136
```

## Results

| sparsity | actual | PPL | BCR@0 | BCR@q25 | BCR@q50 | BCR@q75 | mean margin drop | pref acc dense | pref acc pruned |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.299957 | 7.711824 | 0.092251 | 0.027094 | 0.014760 | 0.000000 | 0.001515 | 0.542 | 0.549 |
| 40% | 0.399872 | 8.112684 | 0.158672 | 0.091133 | 0.062731 | 0.007353 | 0.007667 | 0.542 | 0.530 |
| 50% | 0.500000 | 9.198289 | 0.243542 | 0.184729 | 0.151292 | 0.066176 | 0.030553 | 0.542 | 0.544 |

Dense PPL baseline:

```text
7.458973
```

## Interpretation

The BCR signal is monotonic with sparsity:

```text
BCR@q25: 30% = 2.7%, 40% = 9.1%, 50% = 18.5%
BCR@0:   30% = 9.2%, 40% = 15.9%, 50% = 24.4%
```

Current operating-point reading:

```text
30%: weak alignment damage, strong QA/PPL preservation.
40%: best candidate for "QA still mostly preserved, alignment measurably damaged".
50%: strong boundary-crossing point, but PPL degradation is larger; use as collapse/stress point.
```

This is not yet the full Phase 1 gate. Phase 1 still needs:

- QA benchmarks beyond PPL: ARC-Challenge, HellaSwag, WinoGrande, MMLU.
- Alignment guardrails beyond BCR: at minimum XSTest-FPR for over-refusal, plus IFEval and TruthfulQA; HarmBench/AdvBench if feasible.

## Saved Checkpoints

```text
outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p30
outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p40
outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p50
```

Reuse these checkpoints for the remaining Phase 1 evaluations. Do not repeat
Wanda pruning for the same seed/calibration unless the saved checkpoint is
deleted or found corrupt.

