from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def dense_thresholds(dense_margins: list[float]) -> dict[str, float]:
    positives = [value for value in dense_margins if value > 0.0]
    thresholds = {"0": 0.0}
    if positives:
        arr = np.array(positives, dtype=float)
        thresholds.update(
            {
                "q25": float(np.quantile(arr, 0.25)),
                "q50": float(np.quantile(arr, 0.50)),
                "q75": float(np.quantile(arr, 0.75)),
            }
        )
    else:
        thresholds.update({"q25": 0.0, "q50": 0.0, "q75": 0.0})
    return thresholds


def compute_bcr_metrics(reference_rows: list[dict[str, Any]], pruned_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pruned_by_id = {str(row["id"]): row for row in pruned_rows}
    dense = []
    pruned = []
    drops = []
    for ref in reference_rows:
        row = pruned_by_id[str(ref["id"])]
        dense_margin = float(ref["delta_dense"])
        pruned_margin = float(row["delta_pruned"])
        dense.append(dense_margin)
        pruned.append(pruned_margin)
        drops.append(dense_margin - pruned_margin)

    thresholds = dense_thresholds(dense)
    metrics: dict[str, Any] = {
        "num_examples": len(dense),
        "preference_accuracy_dense": sum(value > 0 for value in dense) / len(dense),
        "preference_accuracy_pruned": sum(value > 0 for value in pruned) / len(pruned),
        "mean_delta_dense": mean(dense),
        "mean_delta_pruned": mean(pruned),
        "mean_margin_drop": mean(drops),
        "thresholds": thresholds,
    }

    for name, threshold in thresholds.items():
        eligible = [idx for idx, value in enumerate(dense) if value > threshold]
        metrics[f"coverage@{name}"] = len(eligible) / len(dense)
        if eligible:
            crossings = sum(pruned[idx] < 0.0 for idx in eligible)
            metrics[f"bcr@{name}"] = crossings / len(eligible)
        else:
            metrics[f"bcr@{name}"] = None
    return metrics

