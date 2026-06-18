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

mkdir -p data/phase1 outputs/phase1/bcr

python scripts/prepare_phase1_hh_rlhf.py \
  --max-samples 50 \
  --seed 0 \
  --out data/phase1/hh_rlhf_bcr_smoke.jsonl

python scripts/compute_phase1_bcr_references.py \
  --data data/phase1/hh_rlhf_bcr_smoke.jsonl \
  --max-samples 50 \
  --out outputs/phase1/bcr/reference_margins_smoke.jsonl

python scripts/run_phase1_wanda_bcr_once.py \
  --sparsity-ratio 0.30 \
  --data data/phase1/hh_rlhf_bcr_smoke.jsonl \
  --references outputs/phase1/bcr/reference_margins_smoke.jsonl \
  --max-samples 50 \
  --out-margins outputs/phase1/bcr/wanda_0p30_margins_smoke.jsonl \
  --out-metrics outputs/phase1/bcr/wanda_0p30_metrics_smoke.json
