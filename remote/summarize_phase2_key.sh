#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"
PHASE2_ROOT="${PHASE2_ROOT:-outputs/phase2}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

if [[ ! -f outputs/phase1/phase1_full_summary.json ]]; then
  bash remote/summarize_phase1_all.sh
fi

python scripts/summarize_phase2_key.py \
  --phase1-summary outputs/phase1/phase1_full_summary.json \
  --phase2-dir "$PHASE2_ROOT" \
  --out-csv "$PHASE2_ROOT/phase2_key_summary.csv" \
  --out-json "$PHASE2_ROOT/phase2_key_summary.json"

cat "$PHASE2_ROOT/phase2_key_summary.csv"
