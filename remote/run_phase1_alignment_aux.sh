#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
export LM_HARNESS_CACHE_PATH="${LM_HARNESS_CACHE_PATH:-$AAP_ROOT/outputs/lm_eval_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

BATCH_SIZE="${BATCH_SIZE:-64}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

python -c "import lm_eval" >/dev/null 2>&1 || {
  echo "lm_eval is not importable. Run: bash remote/install_lm_eval.sh" >&2
  exit 1
}

mkdir -p outputs/phase1/ifeval outputs/phase1/truthfulqa outputs/lm_eval_cache

model_for_name() {
  case "$1" in
    dense)
      echo "Qwen/Qwen2.5-7B-Instruct"
      ;;
    wanda_0p30)
      echo "outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p30"
      ;;
    wanda_0p40)
      echo "outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p40"
      ;;
    wanda_0p50)
      echo "outputs/phase1/pruned_models/qwen2p5_7b_wanda_unstructured_0p50"
      ;;
    *)
      echo "unknown model name: $1" >&2
      return 1
      ;;
  esac
}

run_eval() {
  local suite="$1"
  local name="$2"
  local tasks="$3"
  local out_dir="outputs/phase1/${suite}/${name}"
  local model
  model="$(model_for_name "$name")"

  if [[ -d "$out_dir" ]]; then
    echo "skip existing $out_dir"
    return 0
  fi

  python -m lm_eval run \
    --model hf \
    --model_args "pretrained=${model},dtype=bfloat16,cache_dir=${HF_HUB_CACHE}" \
    --tasks "$tasks" \
    --num_fewshot 0 \
    --batch_size "$BATCH_SIZE" \
    --device cuda:0 \
    --trust_remote_code \
    --output_path "$out_dir"
}

for name in dense wanda_0p30 wanda_0p40 wanda_0p50; do
  run_eval ifeval "$name" ifeval
done

for name in dense wanda_0p30 wanda_0p40 wanda_0p50; do
  run_eval truthfulqa "$name" truthfulqa_mc1,truthfulqa_mc2
done
