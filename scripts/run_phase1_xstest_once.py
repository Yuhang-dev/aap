from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aap.refusal import is_refusal
from aap.wanda_adapter import torch_dtype_from_name


PROMPT_COLUMNS = ["prompt", "instruction", "text", "input"]
LABEL_COLUMNS = ["label", "safety_label", "safe_label", "is_safe", "type", "prompt_type", "category"]


def infer_prompt(row: dict[str, Any]) -> str:
    for column in PROMPT_COLUMNS:
        if column in row and isinstance(row[column], str) and row[column].strip():
            return row[column]
    for key, value in row.items():
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"could not infer prompt column from keys={list(row)}")


def infer_is_safe(row: dict[str, Any]) -> bool:
    for column in LABEL_COLUMNS:
        if column not in row:
            continue
        value = row[column]
        if isinstance(value, bool):
            return value
        text = str(value).lower().strip()
        if text in {"safe", "benign", "allowed", "0"}:
            return True
        if text in {"unsafe", "harmful", "disallowed", "1"}:
            return False
        if "unsafe" in text or "harmful" in text or "contrast" in text:
            return False
        if "safe" in text or "benign" in text:
            return True
    raise ValueError(f"could not infer safe/unsafe label from row={row}")


def infer_prompt_type(row: dict[str, Any]) -> str:
    for column in ["type", "prompt_type", "category", "subcategory"]:
        if column in row:
            return str(row[column])
    return "unknown"


def load_xstest_dataset():
    from datasets import load_dataset

    errors = {}
    for dataset_id in ["Paul/XSTest", "walledai/XSTest"]:
        for split in ["test", "train"]:
            try:
                return dataset_id, split, load_dataset(dataset_id, split=split)
            except Exception as exc:
                errors[f"{dataset_id}:{split}"] = repr(exc)
    raise RuntimeError(f"could not load XSTest from known dataset IDs: {errors}")


def load_xstest(max_samples: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_id, split, dataset = load_xstest_dataset()
    rows = []
    schema = {
        "dataset_id": dataset_id,
        "split": split,
        "columns": list(dataset.column_names),
        "num_total": len(dataset),
    }
    for idx, row in enumerate(dataset):
        if max_samples is not None and len(rows) >= max_samples:
            break
        prompt = infer_prompt(row)
        is_safe = infer_is_safe(row)
        prompt_type = infer_prompt_type(row)
        rows.append(
            {
                "id": str(row.get("id", idx)),
                "prompt": prompt,
                "is_safe": is_safe,
                "prompt_type": prompt_type,
                "raw": dict(row),
            }
        )
    schema["num_loaded"] = len(rows)
    schema["safe_count"] = sum(row["is_safe"] for row in rows)
    schema["unsafe_count"] = sum(not row["is_safe"] for row in rows)
    schema["prompt_type_counts"] = dict(Counter(row["prompt_type"] for row in rows))
    return rows, schema


def format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_model_and_tokenizer(model_path: str, cache_dir: str | None, dtype: str):
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
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    formatted = format_prompt(tokenizer, prompt)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    safe_rows = [row for row in rows if row["is_safe"]]
    unsafe_rows = [row for row in rows if not row["is_safe"]]
    metrics: dict[str, Any] = {
        "num_examples": len(rows),
        "safe_count": len(safe_rows),
        "unsafe_count": len(unsafe_rows),
        "xstest_fpr": sum(row["refusal"] for row in safe_rows) / len(safe_rows) if safe_rows else None,
        "unsafe_refusal_rate": sum(row["refusal"] for row in unsafe_rows) / len(unsafe_rows) if unsafe_rows else None,
        "overall_refusal_rate": sum(row["refusal"] for row in rows) / len(rows) if rows else None,
    }
    by_type = defaultdict(list)
    for row in rows:
        by_type[row["prompt_type"]].append(row)
    metrics["by_prompt_type"] = {
        prompt_type: {
            "count": len(items),
            "safe_count": sum(item["is_safe"] for item in items),
            "unsafe_count": sum(not item["is_safe"] for item in items),
            "refusal_rate": sum(item["refusal"] for item in items) / len(items),
        }
        for prompt_type, items in sorted(by_type.items())
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XSTest refusal/FPR evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out-responses", required=True)
    parser.add_argument("--out-metrics", required=True)
    args = parser.parse_args()

    rows, schema = load_xstest(max_samples=args.max_samples)
    model, tokenizer = load_model_and_tokenizer(args.model, args.cache_dir, args.dtype)

    outputs = []
    for idx, row in enumerate(rows):
        if idx % 25 == 0:
            print(f"{args.name} XSTest {idx}/{len(rows)}")
        response = generate_response(model, tokenizer, row["prompt"], args.max_new_tokens)
        refusal = is_refusal(response)
        outputs.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "is_safe": row["is_safe"],
                "prompt_type": row["prompt_type"],
                "response": response,
                "refusal": refusal,
            }
        )

    metrics = compute_metrics(outputs)
    metrics.update(
        {
            "name": args.name,
            "model": args.model,
            "dtype": args.dtype,
            "max_samples": args.max_samples,
            "max_new_tokens": args.max_new_tokens,
            "dataset_schema": schema,
            "refusal_classifier": "string_heuristic_v1",
        }
    )

    out_responses = Path(args.out_responses)
    out_responses.parent.mkdir(parents=True, exist_ok=True)
    with out_responses.open("w", encoding="utf-8") as f:
        for row in outputs:
            json.dump(row, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
