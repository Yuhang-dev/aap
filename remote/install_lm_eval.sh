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
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed

mkdir -p "$PIP_CACHE_DIR" "$AAP_ROOT/external"

set +e
python -m pip install "lm_eval[hf]"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "pip install lm_eval[hf] failed; falling back to official GitHub repo"
  lm_eval_dir="$AAP_ROOT/external/lm-evaluation-harness"
  if [[ ! -d "$lm_eval_dir/.git" ]]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$lm_eval_dir"
  else
    git -C "$lm_eval_dir" pull --ff-only
  fi
  python -m pip install -e "$lm_eval_dir[hf]" --no-build-isolation
fi

python scripts/phase1_eval_preflight.py

