#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

python scripts/summarize_phase1_all.py \
  --phase1-dir outputs/phase1 \
  --out-csv outputs/phase1/phase1_full_summary.csv \
  --out-json outputs/phase1/phase1_full_summary.json

cat outputs/phase1/phase1_full_summary.csv
