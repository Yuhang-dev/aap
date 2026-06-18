#!/usr/bin/env bash
set -euo pipefail

echo "This script is a draft. Run remote/run_phase1_eval_preflight.sh first."
echo "It expects EleutherAI lm-evaluation-harness to be installed as 'lm_eval'."
exit 1

# Example once lm_eval availability is confirmed:
#
# lm_eval \
#   --model hf \
#   --model_args pretrained=outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p40,dtype=bfloat16,parallelize=True \
#   --tasks arc_challenge,hellaswag,winogrande,mmlu \
#   --batch_size auto \
#   --output_path outputs/phase1/lm_eval/qwen2p5_7b_wanda_0p40.json

