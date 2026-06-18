#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/aap
source /root/autodl-tmp/aap/remote/common.sh
activate_pbp_if_needed
python -m pip install -e . --no-build-isolation
