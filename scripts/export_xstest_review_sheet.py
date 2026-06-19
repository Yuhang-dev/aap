from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_comparison(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("comparison must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("comparison must be NAME=PATH")
    return name.strip(), Path(path)


def response_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in rows}


def add_review_row(
    bucket: str,
    comparison: str,
    dense: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "bucket": bucket,
        "comparison": comparison,
        "id": dense["id"],
        "prompt_type": dense.get("prompt_type", ""),
        "is_safe": dense.get("is_safe", ""),
        "dense_refusal": dense.get("refusal", ""),
        "candidate_refusal": candidate.get("refusal", ""),
        "manual_refusal": "",
        "manual_notes": "",
        "prompt": dense.get("prompt", ""),
        "dense_response": dense.get("response", ""),
        "candidate_response": candidate.get("response", ""),
    }


def take_sample(rows: list[dict[str, Any]], max_items: int, rng: random.Random) -> list[dict[str, Any]]:
    if len(rows) <= max_items:
        return rows
    return rng.sample(rows, max_items)


def build_review_rows(
    dense_rows: list[dict[str, Any]],
    comparison_name: str,
    comparison_rows: list[dict[str, Any]],
    max_per_bucket: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dense_by_id = response_map(dense_rows)
    candidate_by_id = response_map(comparison_rows)
    common_ids = sorted(set(dense_by_id) & set(candidate_by_id))

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_id in common_ids:
        dense = dense_by_id[item_id]
        candidate = candidate_by_id[item_id]
        is_safe = bool(dense.get("is_safe"))
        dense_refusal = bool(dense.get("refusal"))
        candidate_refusal = bool(candidate.get("refusal"))

        if is_safe and (not dense_refusal) and candidate_refusal:
            buckets["safe_new_refusal"].append(
                add_review_row("safe_new_refusal", comparison_name, dense, candidate)
            )
        if is_safe and candidate_refusal:
            buckets["safe_candidate_refusal"].append(
                add_review_row("safe_candidate_refusal", comparison_name, dense, candidate)
            )
        if (not is_safe) and dense_refusal and (not candidate_refusal):
            buckets["unsafe_lost_refusal"].append(
                add_review_row("unsafe_lost_refusal", comparison_name, dense, candidate)
            )
        if is_safe and (not candidate_refusal):
            buckets["random_safe_nonrefusal"].append(
                add_review_row("random_safe_nonrefusal", comparison_name, dense, candidate)
            )
        if not is_safe:
            buckets["random_unsafe"].append(
                add_review_row("random_unsafe", comparison_name, dense, candidate)
            )

    review_rows: list[dict[str, Any]] = []
    bucket_counts = {}
    for bucket_name in [
        "safe_new_refusal",
        "safe_candidate_refusal",
        "unsafe_lost_refusal",
        "random_safe_nonrefusal",
        "random_unsafe",
    ]:
        bucket_rows = buckets[bucket_name]
        bucket_counts[bucket_name] = len(bucket_rows)
        review_rows.extend(take_sample(bucket_rows, max_per_bucket, rng))

    summary = {
        "comparison": comparison_name,
        "common_examples": len(common_ids),
        "bucket_counts_before_sampling": bucket_counts,
        "sampled_rows": len(review_rows),
    }
    return review_rows, summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "bucket",
        "comparison",
        "id",
        "prompt_type",
        "is_safe",
        "dense_refusal",
        "candidate_refusal",
        "manual_refusal",
        "manual_notes",
        "prompt",
        "dense_response",
        "candidate_response",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export XSTest response pairs for manual refusal review.")
    parser.add_argument("--dense-responses", required=True)
    parser.add_argument(
        "--compare",
        action="append",
        type=parse_comparison,
        required=True,
        help="Comparison response JSONL as NAME=PATH. Can be repeated.",
    )
    parser.add_argument("--max-per-bucket", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-summary", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dense_rows = read_jsonl(args.dense_responses)

    all_review_rows: list[dict[str, Any]] = []
    summaries = []
    for comparison_name, comparison_path in args.compare:
        comparison_rows = read_jsonl(comparison_path)
        review_rows, summary = build_review_rows(
            dense_rows,
            comparison_name,
            comparison_rows,
            max_per_bucket=args.max_per_bucket,
            rng=rng,
        )
        all_review_rows.extend(review_rows)
        summaries.append(summary)

    write_csv(args.out_csv, all_review_rows)
    summary_payload = {
        "dense_responses": args.dense_responses,
        "max_per_bucket": args.max_per_bucket,
        "seed": args.seed,
        "comparisons": summaries,
        "total_review_rows": len(all_review_rows),
        "review_csv": args.out_csv,
    }
    out_summary = Path(args.out_summary)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

