# Project Notes

This project follows `alignment_aware_pruning_execution_spec_1.md`.

## Method Boundaries

- `M-grad-SNIP` uses `|theta_i * grad_align_i|`.
- `M-grad-OBS` uses task-compensated alignment saliency with `H_T^-1`.
- `M-ACA` only changes calibration data for Wanda/SparseGPT-style scores.

Do not describe `M-ACA` as a bilevel surrogate. It has no `g_align`.

## Gradient Objective

For gradient saliencies, `L_align` must have nonzero gradient at dense weights.
Use NLL of chosen responses or negative reward. Do not use KL-to-dense for
scoring because the gradient is zero at the dense model.

## Required Alignment Eval Guardrail

Every alignment evaluation must include XSTest false-positive rate to detect
over-refusal.

