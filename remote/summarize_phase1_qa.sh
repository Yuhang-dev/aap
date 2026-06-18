#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

python scripts/summarize_lm_eval_results.py \
  --input-dir outputs/phase1/qa_core \
  --out outputs/phase1/qa_core_summary.csv

