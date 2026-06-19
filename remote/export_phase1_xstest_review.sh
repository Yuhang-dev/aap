#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

python scripts/export_xstest_review_sheet.py \
  --dense-responses outputs/phase1/xstest_core/dense_responses.jsonl \
  --compare wanda_0p30=outputs/phase1/xstest_core/wanda_0p30_responses.jsonl \
  --compare wanda_0p40=outputs/phase1/xstest_core/wanda_0p40_responses.jsonl \
  --compare wanda_0p50=outputs/phase1/xstest_core/wanda_0p50_responses.jsonl \
  --max-per-bucket 80 \
  --seed 0 \
  --out-csv outputs/phase1/xstest_core/xstest_manual_review.csv \
  --out-summary outputs/phase1/xstest_core/xstest_manual_review_summary.json

