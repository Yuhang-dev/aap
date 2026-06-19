#!/usr/bin/env bash
set -euo pipefail

export PHASE2_ROOT="${PHASE2_ROOT:-outputs/phase2_smoke}"
export PHASE2_NSAMPLES="${PHASE2_NSAMPLES:-8}"
export PHASE2_MAX_SAMPLES="${PHASE2_MAX_SAMPLES:-50}"
export PHASE2_PPL_MAX_SAMPLES="${PHASE2_PPL_MAX_SAMPLES:-2}"
export PHASE2_KEEP_MODELS="${PHASE2_KEEP_MODELS:-none}"

bash remote/run_phase2_key_experiments.sh
