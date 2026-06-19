#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
export LM_HARNESS_CACHE_PATH="${LM_HARNESS_CACHE_PATH:-$AAP_ROOT/outputs/lm_eval_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export AAP_LOCAL_FILES_ONLY="${AAP_LOCAL_FILES_ONLY:-1}"

PHASE2_ROOT="${PHASE2_ROOT:-outputs/phase2}"
BATCH_SIZE="${BATCH_SIZE:-64}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$PHASE2_ROOT/qa_core" outputs/lm_eval_cache

run_eval() {
  local tag="$1"
  local model="$PHASE2_ROOT/pruned_models/qwen2p5_7b_${tag}"
  local out_dir="$PHASE2_ROOT/qa_core/${tag}"
  local existing_result
  existing_result="$(find "$out_dir" -type f -name 'results_*.json' -print -quit 2>/dev/null || true)"
  if [[ -n "$existing_result" ]]; then
    echo "skip existing $existing_result"
    return 0
  fi
  if [[ ! -d "$model" ]]; then
    echo "missing pruned model: $model" >&2
    return 1
  fi

  python -m lm_eval run \
    --model hf \
    --model_args "pretrained=${model},dtype=bfloat16,cache_dir=${HF_HUB_CACHE}" \
    --tasks arc_challenge,hellaswag,winogrande,mmlu \
    --num_fewshot 0 \
    --batch_size "$BATCH_SIZE" \
    --device cuda:0 \
    --trust_remote_code \
    --output_path "$out_dir"
}

run_eval maca_chosen_0p30
run_eval maca_chosen_0p40
run_eval maca_chosen_0p50

bash remote/summarize_phase2_key.sh
