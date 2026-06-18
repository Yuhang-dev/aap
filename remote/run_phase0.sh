#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/aap
conda activate pbp
python scripts/run_phase0_toy_gap.py \
  --config configs/phase0_toy_gap.yaml \
  --out-dir outputs/phase0_toy_gap

