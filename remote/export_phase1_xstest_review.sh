#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

required_files=(
  outputs/phase1/xstest_core/dense_responses.jsonl
  outputs/phase1/xstest_core/wanda_0p30_responses.jsonl
  outputs/phase1/xstest_core/wanda_0p40_responses.jsonl
  outputs/phase1/xstest_core/wanda_0p50_responses.jsonl
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required XSTest response file: $path" >&2
    echo "run first: bash remote/run_phase1_xstest_core.sh" >&2
    exit 1
  fi
done

python scripts/export_xstest_review_sheet.py \
  --dense-responses outputs/phase1/xstest_core/dense_responses.jsonl \
  --compare wanda_0p30=outputs/phase1/xstest_core/wanda_0p30_responses.jsonl \
  --compare wanda_0p40=outputs/phase1/xstest_core/wanda_0p40_responses.jsonl \
  --compare wanda_0p50=outputs/phase1/xstest_core/wanda_0p50_responses.jsonl \
  --max-per-bucket 80 \
  --seed 0 \
  --out-csv outputs/phase1/xstest_core/xstest_manual_review.csv \
  --out-summary outputs/phase1/xstest_core/xstest_manual_review_summary.json
