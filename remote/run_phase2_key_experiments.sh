#!/usr/bin/env bash
set -euo pipefail

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export AAP_ROOT="${AAP_ROOT:-$DATA_DISK/aap}"
export HF_HOME="${HF_HOME:-$DATA_DISK/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$DATA_DISK/torch_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export AAP_LOCAL_FILES_ONLY="${AAP_LOCAL_FILES_ONLY:-1}"

PHASE2_ROOT="${PHASE2_ROOT:-outputs/phase2}"
PHASE2_NSAMPLES="${PHASE2_NSAMPLES:-128}"
PHASE2_MAX_SAMPLES="${PHASE2_MAX_SAMPLES:-1000}"
PHASE2_PPL_MAX_SAMPLES="${PHASE2_PPL_MAX_SAMPLES:-16}"
PHASE2_KEEP_MODELS="${PHASE2_KEEP_MODELS:-chosen}"

cd "$AAP_ROOT"
source "$AAP_ROOT/remote/common.sh"
configure_hf_transfer_env
activate_pbp_if_needed
export PYTHONPATH="$AAP_ROOT/src:${PYTHONPATH:-}"

mkdir -p data/phase1 "$PHASE2_ROOT/bcr" "$PHASE2_ROOT/margins" "$PHASE2_ROOT/ppl" "$PHASE2_ROOT/pruned_models"

if [[ ! -f data/phase1/hh_rlhf_bcr_eval.jsonl ]]; then
  python scripts/prepare_phase1_hh_rlhf.py \
    --max-samples 1000 \
    --seed 0 \
    --out data/phase1/hh_rlhf_bcr_eval.jsonl
fi

if [[ ! -f outputs/phase1/bcr/reference_margins.jsonl ]]; then
  python scripts/compute_phase1_bcr_references.py \
    --data data/phase1/hh_rlhf_bcr_eval.jsonl \
    --max-samples 1000 \
    --out outputs/phase1/bcr/reference_margins.jsonl
fi

run_variant() {
  local tag="$1"
  local calibration_source="$2"
  local sparsity="$3"
  local model_dir="$PHASE2_ROOT/pruned_models/qwen2p5_7b_${tag}"
  local metrics="$PHASE2_ROOT/bcr/${tag}_metrics.json"
  local margins="$PHASE2_ROOT/margins/${tag}_margins.jsonl"
  local ppl="$PHASE2_ROOT/ppl/${tag}.json"

  if [[ -f "$metrics" ]]; then
    echo "skip existing $metrics"
  else
    local model_args=(--save-pruned-model "$model_dir")
    if [[ -d "$model_dir" ]]; then
      model_args=(--pruned-model "$model_dir")
    fi

    python scripts/run_phase1_wanda_bcr_once.py \
      --wanda-dir external/wanda \
      --model Qwen/Qwen2.5-7B-Instruct \
      --cache-dir "$HF_HUB_CACHE" \
      --dtype bfloat16 \
      --seed 0 \
      --nsamples "$PHASE2_NSAMPLES" \
      --seqlen 2048 \
      --sparsity-ratio "$sparsity" \
      --sparsity-type unstructured \
      --calibration-source "$calibration_source" \
      --calibration-data data/phase1/hh_rlhf_bcr_eval.jsonl \
      --data data/phase1/hh_rlhf_bcr_eval.jsonl \
      --references outputs/phase1/bcr/reference_margins.jsonl \
      --max-samples "$PHASE2_MAX_SAMPLES" \
      "${model_args[@]}" \
      --out-margins "$margins" \
      --out-metrics "$metrics"
  fi

  if [[ -f "$ppl" ]]; then
    echo "skip existing $ppl"
  else
    python scripts/run_phase1_wanda_once.py \
      --wanda-dir external/wanda \
      --model "$model_dir" \
      --cache-dir "$HF_HUB_CACHE" \
      --dtype bfloat16 \
      --seed 0 \
      --nsamples "$PHASE2_NSAMPLES" \
      --seqlen 2048 \
      --sparsity-ratio 0.0 \
      --sparsity-type unstructured \
      --prune-method wanda \
      --eval-ppl \
      --ppl-max-samples "$PHASE2_PPL_MAX_SAMPLES" \
      --out "$ppl"
  fi

  case "$PHASE2_KEEP_MODELS" in
    all)
      ;;
    chosen)
      if [[ "$tag" != maca_chosen_* ]]; then
        echo "delete control checkpoint after metrics/PPL: $model_dir"
        rm -rf "$model_dir"
      fi
      ;;
    none)
      echo "delete checkpoint after metrics/PPL: $model_dir"
      rm -rf "$model_dir"
      ;;
    *)
      echo "unknown PHASE2_KEEP_MODELS=$PHASE2_KEEP_MODELS; expected all/chosen/none" >&2
      exit 1
      ;;
  esac
}

# Experiment A: alignment-specificity versus instruction-domain matching.
run_variant maca_pair_0p30 hh_pair 0.30
run_variant maca_chosen_0p30 hh_chosen 0.30
run_variant maca_rejected_0p30 hh_rejected 0.30

# Experiment B: high-sparsity alignment-aware calibration versus task-only C4 Wanda.
run_variant maca_pair_0p40 hh_pair 0.40
run_variant maca_chosen_0p40 hh_chosen 0.40
run_variant maca_chosen_0p50 hh_chosen 0.50

bash remote/summarize_phase2_key.sh
