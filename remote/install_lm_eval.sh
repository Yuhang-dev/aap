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
export NLTK_DATA="${NLTK_DATA:-$DATA_DISK/nltk_data}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed

mkdir -p "$PIP_CACHE_DIR" "$NLTK_DATA" "$AAP_ROOT/external"

install_with_current_index() {
  python -m pip install "$@"
}

install_with_pypi_index() {
  python -m pip install -i https://pypi.org/simple "$@"
}

set +e
install_with_current_index "lm_eval[hf]"
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "pip install lm_eval[hf] failed on the configured index; retrying with official PyPI"
  set +e
  install_with_pypi_index "lm_eval[hf]"
  status=$?
  set -e
fi

if [[ "$status" -ne 0 ]]; then
  echo "pip install lm_eval[hf] failed; falling back to official GitHub repo"
  lm_eval_dir="$AAP_ROOT/external/lm-evaluation-harness"
  if [[ ! -d "$lm_eval_dir/.git" ]]; then
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$lm_eval_dir"
  else
    git -C "$lm_eval_dir" pull --ff-only
  fi
  set +e
  install_with_current_index -e "$lm_eval_dir[hf]" --no-build-isolation
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "editable install failed on the configured index; retrying editable install with official PyPI"
    install_with_pypi_index -e "$lm_eval_dir[hf]" --no-build-isolation
  fi
fi

set +e
install_with_current_index langdetect immutabledict
status=$?
set -e

if [[ "$status" -ne 0 ]]; then
  echo "optional IFEval dependency install failed on the configured index; retrying with official PyPI"
  install_with_pypi_index langdetect immutabledict
fi

python - <<'PY'
import os

import nltk

download_dir = os.environ.get("NLTK_DATA")
if download_dir:
    nltk.download("punkt_tab", download_dir=download_dir, quiet=True)

import langdetect  # noqa: F401
import immutabledict  # noqa: F401
from lm_eval.tasks.ifeval import instructions, instructions_registry, utils  # noqa: F401

print("IFEval optional dependencies are importable")
PY

python scripts/phase1_eval_preflight.py
