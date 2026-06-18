from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aap.logprobs import score_response_logprob
from aap.preference_data import read_preference_jsonl, write_jsonl


def torch_dtype_from_name(name: str):
    import torch

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype: {name}")


def model_device(model):
    import torch

    if hasattr(model, "hf_device_map") and "lm_head" in model.hf_device_map:
        return torch.device(model.hf_device_map["lm_head"])
    return next(model.parameters()).device


def load_model(model_id: str, cache_dir: str | None, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {
        "torch_dtype": torch_dtype_from_name(dtype),
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    tok_kwargs = {"use_fast": False}
    if cache_dir:
        tok_kwargs["cache_dir"] = cache_dir
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kwargs)
    return model, tokenizer


def score_records(model, tokenizer, records, max_length: int, prefix: str) -> dict[str, dict]:
    device = model_device(model)
    output = {}
    for idx, record in enumerate(records):
        if idx % 25 == 0:
            print(f"{prefix} scoring {idx}/{len(records)}")
        chosen = score_response_logprob(
            model,
            tokenizer,
            record.prompt,
            record.chosen,
            device,
            max_length=max_length,
        )
        rejected = score_response_logprob(
            model,
            tokenizer,
            record.prompt,
            record.rejected,
            device,
            max_length=max_length,
        )
        output[record.id] = {
            f"{prefix}_chosen": chosen.length_normalized_logprob,
            f"{prefix}_rejected": rejected.length_normalized_logprob,
            f"{prefix}_chosen_tokens": chosen.num_response_tokens,
            f"{prefix}_rejected_tokens": rejected.num_response_tokens,
        }
    return output


def unload_model(model) -> None:
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute base and dense margins for Phase 1 BCR.")
    parser.add_argument("--data", default="data/phase1/hh_rlhf_bcr_eval.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--dense-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--out", default="outputs/phase1/bcr/reference_margins.jsonl")
    args = parser.parse_args()

    records = read_preference_jsonl(args.data, max_samples=args.max_samples)

    base_model, base_tokenizer = load_model(args.base_model, args.cache_dir, args.dtype)
    base_scores = score_records(base_model, base_tokenizer, records, args.max_length, "base")
    unload_model(base_model)

    dense_model, dense_tokenizer = load_model(args.dense_model, args.cache_dir, args.dtype)
    dense_scores = score_records(dense_model, dense_tokenizer, records, args.max_length, "dense")
    unload_model(dense_model)

    rows = []
    for record in records:
        base = base_scores[record.id]
        dense = dense_scores[record.id]
        delta_dense = (
            dense["dense_chosen"]
            - base["base_chosen"]
            - dense["dense_rejected"]
            + base["base_rejected"]
        )
        rows.append(
            {
                "id": record.id,
                **base,
                **dense,
                "delta_dense": delta_dense,
            }
        )
    write_jsonl(args.out, rows)
    print(json.dumps({"out": args.out, "num_examples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

