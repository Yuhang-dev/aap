#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/aap
conda activate pbp
python -m pip install -e .

