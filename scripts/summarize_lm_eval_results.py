from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def find_result_jsons(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "results" in path.name.lower())


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_metric(metrics: dict[str, Any], candidates: list[str]) -> Any:
    for name in candidates:
        if name in metrics:
            return metrics[name]
    return None


def flatten_results(path: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    results = payload.get("results", {})
    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "source_file": str(path),
                "run_name": path.parent.name,
                "task": task,
                "acc": first_metric(metrics, ["acc,none", "acc", "acc_norm,none", "acc_norm"]),
                "acc_norm": first_metric(metrics, ["acc_norm,none", "acc_norm"]),
                "exact_match": first_metric(metrics, ["exact_match,strict-match", "exact_match"]),
                "stderr": first_metric(metrics, ["acc_stderr,none", "acc_stderr", "acc_norm_stderr,none"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize lm-eval JSON results into CSV.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = []
    for path in find_result_jsons(Path(args.input_dir)):
        rows.extend(flatten_results(path, load_json(path)))
    if not rows:
        raise SystemExit(f"no lm-eval result JSONs found under {args.input_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_name", "task", "acc", "acc_norm", "exact_match", "stderr", "source_file"]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

