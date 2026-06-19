# Phase 2 Key Experiments

## Purpose

Phase 1 established the main phenomenon at 30% sparsity: BCR rises while QA and
IFEval remain comparatively preserved. Phase 2 tests whether this is genuinely
alignment-specific and whether alignment-aware calibration can beat task-only
Wanda at higher sparsity.

## Experiment A: Alignment-Specificity

Question: is any gain from preference data just instruction-domain matching, or
does the preferred response side matter?

At 30% sparsity, compare:

| method | calibration source | role |
|---|---|---|
| `wanda_0p30` | C4 | task-only baseline from Phase 1 |
| `maca_pair_0p30` | HH prompt + chosen/rejected | instruction-domain control |
| `maca_chosen_0p30` | HH prompt + chosen | alignment calibration |
| `maca_rejected_0p30` | HH prompt + rejected | negative alignment control |

Support criterion: `maca_chosen_0p30` should reduce BCR more than both
`maca_pair_0p30` and `maca_rejected_0p30`, not merely improve over C4.

## Experiment B: High-Sparsity Alignment-Aware vs Task-Only

Question: at high sparsity, does alignment-aware calibration improve the BCR/QA
Pareto frontier relative to task-only Wanda?

At 40% and 50% sparsity, compare:

| method | calibration source | role |
|---|---|---|
| `wanda_0p40`, `wanda_0p50` | C4 | task-only baselines from Phase 1 |
| `maca_chosen_0p40`, `maca_chosen_0p50` | HH prompt + chosen | alignment calibration |

Primary readout is BCR reduction versus C4 at the same sparsity. PPL is checked
immediately. QA is run for `maca_chosen_*` after BCR/PPL indicates the variant is
worth confirming.

## Commands

Smoke:

```bash
bash remote/run_phase2_key_smoke.sh
```

Full BCR/PPL:

```bash
bash remote/run_phase2_key_experiments.sh
```

Optional QA confirmation:

```bash
bash remote/run_phase2_qa_confirm.sh
```

Summaries:

```bash
bash remote/summarize_phase2_key.sh
```

Outputs:

```text
outputs/phase2/phase2_key_summary.csv
outputs/phase2/phase2_key_summary.json
```
