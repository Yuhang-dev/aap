from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aap.bcr import compute_bcr_metrics, read_jsonl, write_json
from aap.logprobs import score_response_logprob
from aap.preference_data import read_preference_jsonl, write_jsonl
from aap.wanda_adapter import (
    WandaRunConfig,
    add_wanda_to_path,
    check_sparsity,
    load_model_and_tokenizer,
    prune_wanda_aap,
    torch_dtype_from_name,
)


def model_device(model):
    import torch

    if hasattr(model, "hf_device_map") and "lm_head" in model.hf_device_map:
        return torch.device(model.hf_device_map["lm_head"])
    return next(model.parameters()).device


def load_saved_model_and_tokenizer(model_path: str, cache_dir: str | None, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {
        "torch_dtype": torch_dtype_from_name(dtype),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    tok_kwargs = {"use_fast": False}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Wanda pruning BCR evaluation.")
    parser.add_argument("--wanda-dir", default="external/wanda")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--sparsity-ratio", type=float, required=True)
    parser.add_argument("--sparsity-type", default="unstructured", choices=["unstructured", "2:4", "4:8"])
    parser.add_argument(
        "--calibration-source",
        default="c4",
        choices=["c4", "hh_chosen", "hh_rejected", "hh_pair"],
        help="Calibration source for Wanda/M-ACA pruning.",
    )
    parser.add_argument("--calibration-data", default=None, help="Preference JSONL used by HH calibration sources.")
    parser.add_argument("--data", default="data/phase1/hh_rlhf_bcr_eval.jsonl")
    parser.add_argument("--references", default="outputs/phase1/bcr/reference_margins.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--pruned-model", default=None, help="Load an existing pruned model instead of pruning.")
    parser.add_argument("--save-pruned-model", default=None, help="Save the pruned model for reuse.")
    parser.add_argument("--out-margins", required=True)
    parser.add_argument("--out-metrics", required=True)
    args = parser.parse_args()

    add_wanda_to_path(args.wanda_dir)
    records = read_preference_jsonl(args.data, max_samples=args.max_samples)
    reference_rows = read_jsonl(args.references)
    reference_by_id = {str(row["id"]): row for row in reference_rows}

    if args.pruned_model:
        model, tokenizer = load_saved_model_and_tokenizer(args.pruned_model, args.cache_dir, args.dtype)
        actual_sparsity = check_sparsity(model)
    else:
        config = WandaRunConfig(
            wanda_dir=Path(args.wanda_dir),
            model=args.model,
            cache_dir=args.cache_dir,
            dtype=args.dtype,
            seed=args.seed,
            nsamples=args.nsamples,
            seqlen=args.seqlen,
            sparsity_ratio=args.sparsity_ratio,
            sparsity_type=args.sparsity_type,
            prune_method="wanda",
            use_variant=False,
            eval_ppl=False,
            ppl_max_samples=None,
            save_model=None,
            out=Path(args.out_metrics),
            calibration_source=args.calibration_source,
            calibration_data=Path(args.calibration_data) if args.calibration_data else None,
        )
        model, tokenizer = load_model_and_tokenizer(config)

        if args.sparsity_type != "unstructured":
            prune_n, prune_m = [int(part) for part in args.sparsity_type.split(":")]
        else:
            prune_n, prune_m = 0, 0
        if args.sparsity_ratio:
            prune_args = type("Args", (), {})()
            prune_args.seed = args.seed
            prune_args.nsamples = args.nsamples
            prune_args.sparsity_ratio = args.sparsity_ratio
            prune_args.sparsity_type = args.sparsity_type
            prune_args.prune_method = "wanda"
            prune_args.cache_dir = args.cache_dir
            prune_args.use_variant = False
            prune_args.calibration_source = args.calibration_source
            prune_args.calibration_data = args.calibration_data
            prune_wanda_aap(prune_args, model, tokenizer, model_device(model), prune_n=prune_n, prune_m=prune_m)

        actual_sparsity = check_sparsity(model)
        if args.save_pruned_model:
            save_path = Path(args.save_pruned_model)
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
    device = model_device(model)
    rows = []
    for idx, record in enumerate(records):
        if idx % 25 == 0:
            print(f"pruned scoring {idx}/{len(records)}")
        ref = reference_by_id[record.id]
        chosen = score_response_logprob(
            model,
            tokenizer,
            record.prompt,
            record.chosen,
            device,
            max_length=args.max_length,
        )
        rejected = score_response_logprob(
            model,
            tokenizer,
            record.prompt,
            record.rejected,
            device,
            max_length=args.max_length,
        )
        delta_pruned = (
            chosen.length_normalized_logprob
            - float(ref["base_chosen"])
            - rejected.length_normalized_logprob
            + float(ref["base_rejected"])
        )
        rows.append(
            {
                "id": record.id,
                "pruned_chosen": chosen.length_normalized_logprob,
                "pruned_rejected": rejected.length_normalized_logprob,
                "pruned_chosen_tokens": chosen.num_response_tokens,
                "pruned_rejected_tokens": rejected.num_response_tokens,
                "delta_pruned": delta_pruned,
            }
        )

    write_jsonl(args.out_margins, rows)
    selected_refs = [reference_by_id[record.id] for record in records]
    metrics = compute_bcr_metrics(selected_refs, rows)
    metrics.update(
        {
            "model": args.model,
            "method": "wanda",
            "sparsity_type": args.sparsity_type,
            "sparsity_ratio_target": args.sparsity_ratio,
            "actual_sparsity": actual_sparsity,
            "num_eval_examples": len(records),
            "pruned_model": args.pruned_model,
            "saved_pruned_model": args.save_pruned_model,
            "calibration_source": args.calibration_source,
            "calibration_data": args.calibration_data,
        }
    )
    write_json(args.out_metrics, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
