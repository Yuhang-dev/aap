# Phase 1 Current Decision

Phase 1 has established a preliminary PPL-vs-BCR pattern for
`Qwen/Qwen2.5-7B-Instruct` under Wanda(C4):

```text
PPL stays close to dense through 30%, worsens moderately at 40%, and degrades
more visibly at 50%.

BCR rises monotonically from 30% to 50%, with the strongest signal at 50%.
```

## Candidate Operating Band

Use the following points for the next evaluations:

```text
40% unstructured: primary phenomenon candidate.
50% unstructured: stronger collapse / stress point.
30% unstructured: low-damage control.
```

## Not Yet Claimed

Do not yet claim the full Phase 1 phenomenon. PPL+BCR is enough to motivate
continuation, but the Phase 1 gate requires QA and alignment axes beyond these:

```text
QA: ARC-Challenge, HellaSwag, WinoGrande, MMLU.
Alignment: XSTest-FPR mandatory, IFEval, TruthfulQA, HarmBench or AdvBench.
```

## Next Step

Run QA evaluation on dense, 30%, 40%, and 50% checkpoints. If 40% or 50% keeps
QA within the configured tolerance, run XSTest-FPR and the rest of the alignment
suite on the same checkpoints.

Before running QA, check whether `lm_eval` is available in the remote `pbp`
environment:

```bash
bash remote/run_phase1_eval_preflight.sh
cat outputs/phase1/eval_preflight.json
```

If `lm_eval_available` is false, install the HuggingFace backend:

```bash
bash remote/install_lm_eval.sh
cat outputs/phase1/eval_preflight.json
```

The installer keeps pip cache and any source checkout under `/root/autodl-tmp`.
If the configured package mirror is missing dependencies such as `evaluate`, the
installer retries with the official PyPI index.
It also installs IFEval optional dependencies (`langdetect`, `immutabledict`)
and stores NLTK data under `/root/autodl-tmp/nltk_data`.

After `lm_eval_available=true`, run a small QA smoke:

```bash
bash remote/run_phase1_qa_smoke.sh
```

If the smoke passes, run the core QA confirmation:

```bash
bash remote/run_phase1_qa_core.sh
bash remote/summarize_phase1_qa.sh
cat outputs/phase1/qa_core_summary.csv
```

The core QA tasks are:

```text
arc_challenge, hellaswag, winogrande, mmlu
```

They are run on dense, 30%, 40%, and 50% checkpoints.

The QA scripts use a fixed `--batch_size 64`, based on the successful smoke
auto-batch probe. If a full run OOMs on long examples, lower this to `32`.

After QA, run the mandatory over-refusal guardrail:

```bash
bash remote/run_phase1_xstest_smoke.sh
bash remote/run_phase1_xstest_core.sh
```

The XSTest runner reports `xstest_fpr` on safe prompts and
`unsafe_refusal_rate` on unsafe contrast prompts using a string refusal
heuristic. This is the first-pass guardrail; inspect saved JSONL responses if
the heuristic looks suspicious.

The runner loads XSTest from `Paul/XSTest` on Hugging Face, with `walledai/XSTest`
as a fallback mirror.

Export a manual review sheet after XSTest core:

```bash
bash remote/export_phase1_xstest_review.sh
```

This creates:

```text
outputs/phase1/xstest_core/xstest_manual_review.csv
outputs/phase1/xstest_core/xstest_manual_review_summary.json
```

The CSV focuses on safe prompts that newly become refusals under pruning, safe
prompts flagged as refusals by the heuristic, unsafe prompts that lose refusal,
and random calibration examples. Fill `manual_refusal` and `manual_notes` during
review.

Then run the auxiliary alignment checks:

```bash
bash remote/run_phase1_alignment_aux.sh
```

This runs `ifeval` and `truthfulqa_mc1,truthfulqa_mc2` on dense, 30%, 40%, and
50% checkpoints. Treat these as control axes: if BCR rises while IFEval and
TruthfulQA remain comparatively stable, the evidence favors
preference-boundary-specific degradation rather than generic instruction
following or truthfulness collapse.

After the full Phase 1 suite finishes, create the integrated summary table:

```bash
bash remote/summarize_phase1_all.sh
```

This writes:

```text
outputs/phase1/phase1_full_summary.csv
outputs/phase1/phase1_full_summary.json
```
