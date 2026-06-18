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
export OMP_NUM_THREADS=8

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

mkdir -p outputs/phase1/wanda_ppl_sweep

dense_out="outputs/phase1/wanda_ppl_sweep/qwen2p5_7b_dense.json"
if [[ ! -f "$dense_out" ]]; then
  python scripts/run_phase1_wanda_once.py \
    --wanda-dir external/wanda \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir "$HF_HUB_CACHE" \
    --dtype bfloat16 \
    --seed 0 \
    --nsamples 128 \
    --seqlen 2048 \
    --sparsity-ratio 0.0 \
    --sparsity-type unstructured \
    --prune-method wanda \
    --eval-ppl \
    --out "$dense_out"
else
  echo "skip existing $dense_out"
fi

for sparsity in 0.10 0.20 0.30 0.40 0.50 0.60 0.70; do
  tag="${sparsity/./p}"
  out="outputs/phase1/wanda_ppl_sweep/qwen2p5_7b_wanda_unstructured_${tag}.json"
  if [[ -f "$out" ]]; then
    echo "skip existing $out"
    continue
  fi
  python scripts/run_phase1_wanda_once.py \
    --wanda-dir external/wanda \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir "$HF_HUB_CACHE" \
    --dtype bfloat16 \
    --seed 0 \
    --nsamples 128 \
    --seqlen 2048 \
    --sparsity-ratio "$sparsity" \
    --sparsity-type unstructured \
    --prune-method wanda \
    --eval-ppl \
    --out "$out"
done

out_24="outputs/phase1/wanda_ppl_sweep/qwen2p5_7b_wanda_2to4.json"
if [[ ! -f "$out_24" ]]; then
  python scripts/run_phase1_wanda_once.py \
    --wanda-dir external/wanda \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cache-dir "$HF_HUB_CACHE" \
    --dtype bfloat16 \
    --seed 0 \
    --nsamples 128 \
    --seqlen 2048 \
    --sparsity-ratio 0.50 \
    --sparsity-type 2:4 \
    --prune-method wanda \
    --eval-ppl \
    --out "$out_24"
else
  echo "skip existing $out_24"
fi

python scripts/summarize_phase1_wanda_sweep.py \
  --input-dir outputs/phase1/wanda_ppl_sweep \
  --out outputs/phase1/wanda_ppl_sweep_summary.csv
