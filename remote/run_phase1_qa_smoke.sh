#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DISK/pip_cache}"
export LM_HARNESS_CACHE_PATH="${LM_HARNESS_CACHE_PATH:-$AAP_ROOT/outputs/lm_eval_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS=8

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

mkdir -p outputs/phase1/qa_smoke outputs/lm_eval_cache

run_eval() {
  local name="$1"
  local pretrained="$2"
  local out_dir="outputs/phase1/qa_smoke/${name}"
  if [[ -d "$out_dir" ]]; then
    echo "skip existing $out_dir"
    return 0
  fi
  python -m lm_eval run \
    --model hf \
    --model_args "pretrained=${pretrained},dtype=bfloat16,cache_dir=${HF_HUB_CACHE}" \
    --tasks arc_challenge,hellaswag,winogrande,mmlu \
    --num_fewshot 0 \
    --batch_size auto:4 \
    --device cuda:0 \
    --limit 20 \
    --trust_remote_code \
    --output_path "$out_dir"
}

run_eval "dense_limit20" "Qwen/Qwen2.5-7B-Instruct"
run_eval "wanda_0p40_limit20" "outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p40"

