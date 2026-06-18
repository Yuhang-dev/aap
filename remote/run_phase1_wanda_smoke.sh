#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

python scripts/run_phase1_wanda_once.py \
  --wanda-dir external/wanda \
  --model Qwen/Qwen2.5-7B-Instruct \
  --cache-dir "$HF_HUB_CACHE" \
  --dtype bfloat16 \
  --seed 0 \
  --nsamples 8 \
  --seqlen 2048 \
  --sparsity-ratio 0.10 \
  --sparsity-type unstructured \
  --prune-method wanda \
  --eval-ppl \
  --out outputs/phase1/wanda_smoke_qwen2p5_7b_10p.json

