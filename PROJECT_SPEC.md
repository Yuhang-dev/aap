# Alignment-Aware Pruning Execution Plan

This repository implements the staged plan from
`alignment_aware_pruning_execution_spec_1.md`.

## Core Claim Under Test

Standard pruning can preserve general QA metrics while damaging preference
alignment. The task-optimal pruning mask and the alignment-optimal pruning mask
may be genuinely different because task-loss compensation can move weights in a
direction that harms alignment.

## Method Objects

Keep these methods distinct:

| tag | meaning |
|---|---|
| `M-grad-SNIP` | alignment-gradient SNIP, `abs(theta_i * grad_align_i)` |
| `M-grad-OBS` | task-compensated alignment saliency using `H_T^-1 grad_align` |
| `M-ACA` | mixed calibration data for Wanda/SparseGPT activation or Hessian scores |

`M-ACA` is not a bilevel surrogate and must not be described as one.

## Gates

1. Phase 0: analytical toy gap. Decide whether the bilevel / `M-grad-OBS`
   theory branch is alive.
2. Phase 1: establish a sparsity band where QA remains roughly flat while at
   least one alignment axis degrades.
3. Phase 2: disentangle alignment-specific calibration from plain instruction
   domain matching.
4. Phase 3: compare methods on the alignment-QA Pareto frontier with seeds.
5. Phase 4: optional full bilevel mask learning, only if earlier gates justify
   the extra complexity.

## Immediate Milestone

Implement and run Phase 0. Do not start the large GPU matrix until the Phase 0
gate is read.

The default Phase 0 gate uses a task-budgeted comparison: among masks whose
block-OBS task loss is within 5% of the greedy-OBS task loss, choose the mask
with the best compensation-only alignment change. This isolates the theoretical
claim that task-optimal OBS compensation can move remaining weights in an
alignment-damaging direction.
