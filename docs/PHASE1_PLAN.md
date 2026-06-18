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
- both Qwen2.5-7B base and instruct caches are present.

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

