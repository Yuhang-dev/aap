from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 1 Wanda sweep JSON outputs.")
    parser.add_argument("--input-dir", default="outputs/phase1/wanda_ppl_sweep")
    parser.add_argument("--out", default="outputs/phase1/wanda_ppl_sweep_summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    rows = []
    for path in sorted(input_dir.glob("*.json")):
        payload = load_json(path)
        rows.append(
            {
                "file": str(path),
                "model": payload.get("model"),
                "prune_method": payload.get("prune_method"),
                "sparsity_type": payload.get("sparsity_type"),
                "sparsity_ratio_target": payload.get("sparsity_ratio_target"),
                "actual_sparsity": payload.get("actual_sparsity"),
                "wikitext2_ppl": payload.get("wikitext2_ppl"),
                "nsamples": payload.get("nsamples"),
                "ppl_max_samples": payload.get("ppl_max_samples"),
                "seed": payload.get("seed"),
                "runtime_seconds": payload.get("runtime_seconds"),
            }
        )

    if not rows:
        raise SystemExit(f"no JSON outputs found under {input_dir}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "model",
        "prune_method",
        "sparsity_type",
        "sparsity_ratio_target",
        "actual_sparsity",
        "wikitext2_ppl",
        "nsamples",
        "ppl_max_samples",
        "seed",
        "runtime_seconds",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out}")
    for row in rows:
        print(
            f"{row['sparsity_type']} {row['sparsity_ratio_target']}: "
            f"actual={row['actual_sparsity']} ppl={row['wikitext2_ppl']}"
        )


if __name__ == "__main__":
    main()

