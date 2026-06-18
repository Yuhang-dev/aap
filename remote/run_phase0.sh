#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/aap
source /root/autodl-tmp/aap/remote/common.sh
activate_pbp_if_needed
export PYTHONPATH="/root/autodl-tmp/aap/src:${PYTHONPATH:-}"
python scripts/run_phase0_toy_gap.py \
  --config configs/phase0_toy_gap.yaml \
  --out-dir outputs/phase0_toy_gap
