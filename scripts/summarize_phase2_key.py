from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


BASELINE_BY_SPARSITY = {
    "0p30": "wanda_0p30",
    "0p40": "wanda_0p40",
    "0p50": "wanda_0p50",
}

QA_TASKS = {
    "qa_arc_challenge": ("arc_challenge", ["acc_norm,none", "acc_norm", "acc,none", "acc"]),
    "qa_hellaswag": ("hellaswag", ["acc_norm,none", "acc_norm", "acc,none", "acc"]),
    "qa_winogrande": ("winogrande", ["acc,none", "acc"]),
    "qa_mmlu": ("mmlu", ["acc,none", "acc"]),
}

FIELDS = [
    "experiment",
    "method",
    "role",
    "sparsity_tag",
    "sparsity_ratio_target",
    "actual_sparsity",
    "calibration_source",
    "wikitext2_ppl",
    "ppl_delta_vs_c4",
    "qa_avg4",
    "qa_avg4_delta_vs_c4_pp",
    "bcr_at_0",
    "bcr_at_0_delta_vs_c4_pp",
    "bcr_at_q25",
    "mean_margin_drop",
    "preference_accuracy_pruned",
    "status",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_result_json(root: Path) -> Path | None:
    files = sorted(root.rglob("results_*.json"))
    return files[-1] if files else None


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def pp_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return 100.0 * (value - baseline)


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def metric_from_task(payload: dict[str, Any] | None, task: str, candidates: list[str]) -> float | None:
    if not payload:
        return None
    metrics = payload.get("results", {}).get(task, {})
    if not isinstance(metrics, dict):
        return None
    for key in candidates:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def qa_avg_from_result(path: Path | None) -> float | None:
    payload = load_json(path) if path else None
    values = []
    for task, candidates in QA_TASKS.values():
        value = metric_from_task(payload, task, candidates)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if len(values) == len(QA_TASKS) else None


def load_phase1_baselines(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not payload:
        raise SystemExit(f"missing Phase 1 summary JSON: {path}")
    return {row["model"]: row for row in payload["rows"]}


def parse_tag(path: Path) -> tuple[str, str, str]:
    # Example: maca_chosen_0p40_metrics.json
    stem = path.name.removesuffix("_metrics.json")
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"unexpected metrics filename: {path}")
    sparsity_tag = parts[-1]
    method = "_".join(parts[:-1])
    return stem, method, sparsity_tag


def role_for_method(method: str) -> str:
    if method == "maca_chosen":
        return "alignment_calibration"
    if method == "maca_pair":
        return "instruction_domain_control"
    if method == "maca_rejected":
        return "negative_alignment_control"
    return "unknown"


def experiment_for(method: str, sparsity_tag: str) -> str:
    if sparsity_tag == "0p30":
        return "alignment_specificity"
    if method == "maca_chosen" and sparsity_tag in {"0p40", "0p50"}:
        return "high_sparsity_alignment_aware_vs_task_only"
    return "supporting_control"


def collect_rows(phase1: dict[str, dict[str, Any]], phase2_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows = []
    sources: dict[str, dict[str, str]] = {}
    for bcr_path in sorted((phase2_dir / "bcr").glob("*_metrics.json")):
        tag, method, sparsity_tag = parse_tag(bcr_path)
        baseline_name = BASELINE_BY_SPARSITY.get(sparsity_tag)
        if not baseline_name:
            continue
        baseline = phase1[baseline_name]
        bcr = load_json(bcr_path) or {}
        ppl_path = phase2_dir / "ppl" / f"{tag}.json"
        ppl = load_json(ppl_path) or {}
        qa_path = latest_result_json(phase2_dir / "qa_core" / tag)
        qa_avg = qa_avg_from_result(qa_path)

        row = {
            "experiment": experiment_for(method, sparsity_tag),
            "method": method,
            "role": role_for_method(method),
            "sparsity_tag": sparsity_tag,
            "sparsity_ratio_target": as_float(bcr.get("sparsity_ratio_target")),
            "actual_sparsity": as_float(bcr.get("actual_sparsity")),
            "calibration_source": bcr.get("calibration_source"),
            "wikitext2_ppl": as_float(ppl.get("wikitext2_ppl")),
            "ppl_delta_vs_c4": delta(as_float(ppl.get("wikitext2_ppl")), baseline.get("wikitext2_ppl")),
            "qa_avg4": qa_avg,
            "qa_avg4_delta_vs_c4_pp": pp_delta(qa_avg, baseline.get("qa_avg4")),
            "bcr_at_0": as_float(bcr.get("bcr@0")),
            "bcr_at_0_delta_vs_c4_pp": pp_delta(as_float(bcr.get("bcr@0")), baseline.get("bcr_at_0")),
            "bcr_at_q25": as_float(bcr.get("bcr@q25")),
            "mean_margin_drop": as_float(bcr.get("mean_margin_drop")),
            "preference_accuracy_pruned": as_float(bcr.get("preference_accuracy_pruned")),
            "status": "qa_complete" if qa_avg is not None else "qa_pending",
        }
        rows.append(row)
        sources[tag] = {
            "bcr": str(bcr_path),
            "ppl": str(ppl_path) if ppl_path.exists() else "",
            "qa": str(qa_path) if qa_path else "",
            "phase1_baseline": baseline_name,
        }
    return rows, sources


def decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["method"], row["sparsity_tag"]): row for row in rows}
    spec_chosen = by_key.get(("maca_chosen", "0p30"))
    spec_pair = by_key.get(("maca_pair", "0p30"))
    spec_rejected = by_key.get(("maca_rejected", "0p30"))
    high = [by_key.get(("maca_chosen", tag)) for tag in ["0p40", "0p50"]]

    specificity_support = None
    if spec_chosen and spec_pair:
        chosen_gain = -(spec_chosen.get("bcr_at_0_delta_vs_c4_pp") or 0.0)
        pair_gain = -(spec_pair.get("bcr_at_0_delta_vs_c4_pp") or 0.0)
        specificity_support = chosen_gain > pair_gain + 1.0
        if spec_rejected:
            rejected_gain = -(spec_rejected.get("bcr_at_0_delta_vs_c4_pp") or 0.0)
            specificity_support = specificity_support and chosen_gain > rejected_gain + 1.0

    high_sparsity_support = []
    for row in high:
        if not row:
            continue
        bcr_improves = (row.get("bcr_at_0_delta_vs_c4_pp") or 0.0) < -2.0
        qa_ok = row.get("qa_avg4_delta_vs_c4_pp") is None or row["qa_avg4_delta_vs_c4_pp"] >= -1.0
        ppl_ok = row.get("ppl_delta_vs_c4") is None or row["ppl_delta_vs_c4"] <= 0.5
        high_sparsity_support.append(
            {
                "method": row["method"],
                "sparsity_tag": row["sparsity_tag"],
                "bcr_improves": bcr_improves,
                "qa_ok_or_pending": qa_ok,
                "ppl_ok_or_pending": ppl_ok,
                "supported": bcr_improves and qa_ok and ppl_ok,
            }
        )

    return {
        "alignment_specificity_supported": specificity_support,
        "high_sparsity_alignment_aware_checks": high_sparsity_support,
        "rules": {
            "specificity": "maca_chosen BCR gain must exceed maca_pair and maca_rejected gains by >1 percentage point at 30%.",
            "high_sparsity": "maca_chosen must reduce BCR@0 by >2 pp vs C4 task-only with <=0.5 PPL increase and <=1 pp QA drop, when QA is available.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 key experiments.")
    parser.add_argument("--phase1-summary", default="outputs/phase1/phase1_full_summary.json")
    parser.add_argument("--phase2-dir", default="outputs/phase2")
    parser.add_argument("--out-csv", default="outputs/phase2/phase2_key_summary.csv")
    parser.add_argument("--out-json", default="outputs/phase2/phase2_key_summary.json")
    args = parser.parse_args()

    phase1 = load_phase1_baselines(Path(args.phase1_summary))
    rows, sources = collect_rows(phase1, Path(args.phase2_dir))
    if not rows:
        raise SystemExit(f"no Phase 2 metrics found under {args.phase2_dir}/bcr")
    verdict = decision(rows)

    write_csv(Path(args.out_csv), rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps({"rows": rows, "sources": sources, "phase2_decision": verdict}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
