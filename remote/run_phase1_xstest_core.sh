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
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

mkdir -p outputs/phase1/xstest_core

run_xstest() {
  local name="$1"
  local model="$2"
  local metrics="outputs/phase1/xstest_core/${name}_metrics.json"
  if [[ -f "$metrics" ]]; then
    echo "skip existing $metrics"
    return 0
  fi
  python scripts/run_phase1_xstest_once.py \
    --name "$name" \
    --model "$model" \
    --cache-dir "$HF_HUB_CACHE" \
    --out-responses "outputs/phase1/xstest_core/${name}_responses.jsonl" \
    --out-metrics "$metrics"
}

run_xstest dense Qwen/Qwen2.5-7B-Instruct
run_xstest wanda_0p30 outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p30
run_xstest wanda_0p40 outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p40
run_xstest wanda_0p50 outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p50

