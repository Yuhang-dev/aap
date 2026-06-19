from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODEL_ORDER = ["dense", "wanda_0p30", "wanda_0p40", "wanda_0p50"]

PPL_FILES = {
    "dense": "qwen2p5_7b_dense.json",
    "wanda_0p30": "qwen2p5_7b_wanda_unstructured_0p30.json",
    "wanda_0p40": "qwen2p5_7b_wanda_unstructured_0p40.json",
    "wanda_0p50": "qwen2p5_7b_wanda_unstructured_0p50.json",
}

QA_TASKS = {
    "qa_arc_challenge": ("arc_challenge", ["acc_norm,none", "acc_norm", "acc,none", "acc"]),
    "qa_hellaswag": ("hellaswag", ["acc_norm,none", "acc_norm", "acc,none", "acc"]),
    "qa_winogrande": ("winogrande", ["acc,none", "acc"]),
    "qa_mmlu": ("mmlu", ["acc,none", "acc"]),
}

SUMMARY_FIELDS = [
    "model",
    "role",
    "sparsity_ratio_target",
    "actual_sparsity",
    "wikitext2_ppl",
    "wikitext2_ppl_delta",
    "qa_arc_challenge",
    "qa_hellaswag",
    "qa_winogrande",
    "qa_mmlu",
    "qa_avg4",
    "qa_avg4_delta_pp",
    "bcr_at_0",
    "bcr_at_q25",
    "bcr_at_q50",
    "bcr_at_q75",
    "mean_margin_drop",
    "preference_accuracy_dense",
    "preference_accuracy_pruned",
    "xstest_fpr",
    "xstest_fpr_delta_pp",
    "unsafe_refusal_rate",
    "unsafe_refusal_rate_delta_pp",
    "ifeval_prompt_strict",
    "ifeval_prompt_strict_delta_pp",
    "ifeval_inst_strict",
    "ifeval_inst_strict_delta_pp",
    "truthfulqa_mc1",
    "truthfulqa_mc1_delta_pp",
    "truthfulqa_mc2",
    "truthfulqa_mc2_delta_pp",
]


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_result_json(root: Path) -> Path | None:
    files = sorted(root.rglob("results_*.json"))
    return files[-1] if files else None


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


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def delta_pp(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return 100.0 * (value - baseline)


def delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def role_for_model(model: str) -> str:
    return {
        "dense": "baseline",
        "wanda_0p30": "primary_phenomenon",
        "wanda_0p40": "edge_region",
        "wanda_0p50": "stress_region",
    }[model]


def build_row(root: Path, model: str) -> tuple[dict[str, Any], dict[str, str]]:
    row: dict[str, Any] = {"model": model, "role": role_for_model(model)}
    sources: dict[str, str] = {}

    ppl_path = root / "wanda_ppl_sweep" / PPL_FILES[model]
    ppl = load_json(ppl_path)
    if ppl:
        sources["ppl"] = str(ppl_path)
        row["sparsity_ratio_target"] = as_float(ppl.get("sparsity_ratio_target"))
        row["actual_sparsity"] = as_float(ppl.get("actual_sparsity"))
        row["wikitext2_ppl"] = as_float(ppl.get("wikitext2_ppl"))
    elif model == "dense":
        row["sparsity_ratio_target"] = 0.0
        row["actual_sparsity"] = 0.0

    qa_path = latest_result_json(root / "qa_core" / model)
    qa_payload = load_json(qa_path) if qa_path else None
    if qa_path:
        sources["qa"] = str(qa_path)
    qa_values = []
    for field, (task, candidates) in QA_TASKS.items():
        value = metric_from_task(qa_payload, task, candidates)
        row[field] = value
        if value is not None:
            qa_values.append(value)
    row["qa_avg4"] = sum(qa_values) / len(qa_values) if len(qa_values) == len(QA_TASKS) else None

    if model != "dense":
        bcr_path = root / "bcr" / f"{model}_metrics.json"
        bcr = load_json(bcr_path)
        if bcr:
            sources["bcr"] = str(bcr_path)
            row["bcr_at_0"] = as_float(bcr.get("bcr@0"))
            row["bcr_at_q25"] = as_float(bcr.get("bcr@q25"))
            row["bcr_at_q50"] = as_float(bcr.get("bcr@q50"))
            row["bcr_at_q75"] = as_float(bcr.get("bcr@q75"))
            row["mean_margin_drop"] = as_float(bcr.get("mean_margin_drop"))
            row["preference_accuracy_dense"] = as_float(bcr.get("preference_accuracy_dense"))
            row["preference_accuracy_pruned"] = as_float(bcr.get("preference_accuracy_pruned"))

    xstest_path = root / "xstest_core" / f"{model}_metrics.json"
    xstest = load_json(xstest_path)
    if xstest:
        sources["xstest"] = str(xstest_path)
        row["xstest_fpr"] = as_float(xstest.get("xstest_fpr"))
        row["unsafe_refusal_rate"] = as_float(xstest.get("unsafe_refusal_rate"))

    ifeval_path = latest_result_json(root / "ifeval" / model)
    ifeval = load_json(ifeval_path) if ifeval_path else None
    if ifeval_path:
        sources["ifeval"] = str(ifeval_path)
    row["ifeval_prompt_strict"] = metric_from_task(
        ifeval, "ifeval", ["prompt_level_strict_acc,none", "prompt_level_strict_acc"]
    )
    row["ifeval_inst_strict"] = metric_from_task(
        ifeval, "ifeval", ["inst_level_strict_acc,none", "inst_level_strict_acc"]
    )

    truthfulqa_path = latest_result_json(root / "truthfulqa" / model)
    truthfulqa = load_json(truthfulqa_path) if truthfulqa_path else None
    if truthfulqa_path:
        sources["truthfulqa"] = str(truthfulqa_path)
    row["truthfulqa_mc1"] = metric_from_task(truthfulqa, "truthfulqa_mc1", ["acc,none", "acc"])
    row["truthfulqa_mc2"] = metric_from_task(truthfulqa, "truthfulqa_mc2", ["acc,none", "acc"])

    return row, sources


def add_baseline_deltas(rows: list[dict[str, Any]]) -> None:
    baseline = next(row for row in rows if row["model"] == "dense")
    for row in rows:
        row["wikitext2_ppl_delta"] = delta(row.get("wikitext2_ppl"), baseline.get("wikitext2_ppl"))
        row["qa_avg4_delta_pp"] = delta_pp(row.get("qa_avg4"), baseline.get("qa_avg4"))
        row["xstest_fpr_delta_pp"] = delta_pp(row.get("xstest_fpr"), baseline.get("xstest_fpr"))
        row["unsafe_refusal_rate_delta_pp"] = delta_pp(
            row.get("unsafe_refusal_rate"), baseline.get("unsafe_refusal_rate")
        )
        row["ifeval_prompt_strict_delta_pp"] = delta_pp(
            row.get("ifeval_prompt_strict"), baseline.get("ifeval_prompt_strict")
        )
        row["ifeval_inst_strict_delta_pp"] = delta_pp(
            row.get("ifeval_inst_strict"), baseline.get("ifeval_inst_strict")
        )
        row["truthfulqa_mc1_delta_pp"] = delta_pp(row.get("truthfulqa_mc1"), baseline.get("truthfulqa_mc1"))
        row["truthfulqa_mc2_delta_pp"] = delta_pp(row.get("truthfulqa_mc2"), baseline.get("truthfulqa_mc2"))


def phase1_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for row in rows:
        if row["model"] == "dense":
            continue
        bcr = row.get("bcr_at_0")
        qa_delta = row.get("qa_avg4_delta_pp")
        ifeval_delta = row.get("ifeval_prompt_strict_delta_pp")
        if (
            bcr is not None
            and qa_delta is not None
            and ifeval_delta is not None
            and bcr >= 0.05
            and qa_delta >= -2.0
            and ifeval_delta >= -2.0
        ):
            candidates.append(row["model"])

    return {
        "primary_model": candidates[0] if candidates else None,
        "supported_claim": (
            "At the primary sparsity point, preference-boundary damage is measurable "
            "while QA and instruction following are comparatively preserved."
        ),
        "candidate_rule": {
            "bcr_at_0_min": 0.05,
            "qa_avg4_delta_pp_min": -2.0,
            "ifeval_prompt_strict_delta_pp_min": -2.0,
        },
        "candidates": candidates,
        "stress_models": [row["model"] for row in rows if row["model"] in {"wanda_0p40", "wanda_0p50"}],
        "next_step": "Phase 2: test alignment-specificity against domain/task controls.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SUMMARY_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all Phase 1 evaluation axes.")
    parser.add_argument("--phase1-dir", default="outputs/phase1")
    parser.add_argument("--out-csv", default="outputs/phase1/phase1_full_summary.csv")
    parser.add_argument("--out-json", default="outputs/phase1/phase1_full_summary.json")
    args = parser.parse_args()

    root = Path(args.phase1_dir)
    rows = []
    sources = {}
    for model in MODEL_ORDER:
        row, model_sources = build_row(root, model)
        rows.append(row)
        sources[model] = model_sources
    add_baseline_deltas(rows)

    decision = phase1_decision(rows)
    write_csv(Path(args.out_csv), rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps({"rows": rows, "sources": sources, "phase1_decision": decision}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_json}")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
