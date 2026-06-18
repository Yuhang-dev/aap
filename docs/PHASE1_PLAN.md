# Phase 1 Plan

Phase 1 is the first GPU phase.

## Objective

Find a sparsity band where standard task-only Wanda pruning preserves general QA
approximately while at least one alignment axis degrades materially.

## First Target

Use the cached remote model first:

```text
Qwen/Qwen2.5-7B-Instruct
```

Base model cache for BCR/reference work:

```text
Qwen/Qwen2.5-7B
```

## Preflight

```bash
cd /root/autodl-tmp/aap
conda activate pbp
git pull --ff-only
bash remote/run_phase1_preflight.sh
cat outputs/phase1_preflight.json
```

Do not start the pruning/eval matrix unless:

- `ready_for_phase1_gpu` is true;
- `torch_cuda.cuda_available` is true;
- `nvidia-smi` sees the 96 GB GPU;
- both Qwen2.5-7B base and instruct caches have complete weight files.

Audit model cache completeness directly:

```bash
python scripts/audit_hf_model_cache.py
```

Current preflight status:

```text
ready_for_phase1_gpu: true
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 94.97 GiB
python: /root/miniconda3/envs/pbp/bin/python
cached: Qwen/Qwen2.5-7B, Qwen/Qwen2.5-7B-Instruct
```

## Wanda Setup

The official Wanda repository is used as an external dependency and pinned to:

```text
https://github.com/locuslab/wanda.git
8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6
```

All downloads and generated artifacts must stay under the data disk:

```text
/root/autodl-tmp
```

The setup script clones Wanda to:

```text
/root/autodl-tmp/aap/external/wanda
```

Hugging Face model and dataset downloads use:

```text
HF_HOME=/root/autodl-tmp/hf_cache
HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
```

Set it up on the remote machine:

```bash
bash remote/setup_wanda.sh
```

## First GPU Smoke

Run one small Wanda job before the full sweep:

```bash
bash remote/run_phase1_wanda_smoke.sh
cat outputs/phase1/wanda_smoke_qwen2p5_7b_10p.json
```

The smoke uses `nsamples=8` and `seqlen=2048`, prunes 10% unstructured weights,
and evaluates WikiText-2 perplexity in the same process. It does not save a full
pruned 7B model.

## Cleanup Failed Dataset Cache

If an old failed loader created redundant dataset cache files, inspect first:

```bash
bash remote/cleanup_failed_dataset_cache.sh
```

Then delete only the listed dataset-builder candidates:

```bash
bash remote/cleanup_failed_dataset_cache.sh --delete
```

This script refuses to clean outside `/root/autodl-tmp` and does not touch
Hugging Face model caches.

## Sweep

Initial Phase 1 matrix:

```text
model: Qwen/Qwen2.5-7B-Instruct
method: Wanda(C4)
sparsities: 10%, 20%, 30%, 40%, 50%, 60%, 70%, plus 2:4
```

Required eval axes:

```text
QA: WikiText-2 PPL, ARC-Challenge, HellaSwag, WinoGrande, MMLU
Alignment: BCR, HarmBench or AdvBench, XSTest-FPR, IFEval, TruthfulQA
```

AlpacaEval is deferred until Phase 3 decision points.

Run the first PPL-only sweep:

```bash
bash remote/run_phase1_wanda_ppl_sweep.sh
cat outputs/phase1/wanda_ppl_sweep_summary.csv
```

This produces one JSON per sparsity under:

```text
outputs/phase1/wanda_ppl_sweep/
```

The script skips existing JSON outputs so it can be resumed after interruption.

The sweep includes a dense baseline:

```text
outputs/phase1/wanda_ppl_sweep/qwen2p5_7b_dense.json
```

Use this as the reference for the Phase 1 QA preservation gate.

## BCR Alignment Probe

After the PPL sweep, run a small BCR smoke:

```bash
bash remote/run_phase1_bcr_smoke.sh
cat outputs/phase1/bcr/wanda_0p30_metrics_smoke.json
```

If the smoke passes, run BCR on the PPL-selected band:

```bash
bash remote/run_phase1_bcr_band.sh
cat outputs/phase1/bcr/wanda_0p30_metrics.json
cat outputs/phase1/bcr/wanda_0p40_metrics.json
cat outputs/phase1/bcr/wanda_0p50_metrics.json
```

This BCR probe is not the full Phase 1 alignment suite, but it is the cheapest
way to check whether preference boundary crossing appears in the likely
operating band.

Current BCR band result:

```text
30%: BCR@q25 = 0.027, PPL = 7.712
40%: BCR@q25 = 0.091, PPL = 8.113
50%: BCR@q25 = 0.185, PPL = 9.198
```

The next Phase 1 step is QA benchmark confirmation on dense and the saved
30/40/50 checkpoints. Do not advance to Phase 2 until QA and over-refusal
guardrails are checked.

The BCR band runner saves pruned checkpoints under:

```text
outputs/phase1/pruned_models/
```

Each 7B bf16 pruned checkpoint is roughly the size of the dense model. This is
intentional for Phase 1: pruning is deterministic for a fixed seed/calibration
sample, and saved checkpoints can be reused for XSTest-FPR, IFEval, TruthfulQA,
and HarmBench/AdvBench instead of repeating Wanda.
