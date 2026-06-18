from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aap.preference_data import normalize_hh_pair, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HH-RLHF preference pairs for Phase 1 BCR.")
    parser.add_argument("--dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/phase1/hh_rlhf_bcr_eval.jsonl")
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    records = []
    for idx in indices:
        if len(records) >= args.max_samples:
            break
        try:
            records.append(asdict(normalize_hh_pair(dataset[idx], idx)))
        except Exception as exc:
            print(f"skip idx={idx}: {exc}")

    if len(records) != args.max_samples:
        raise SystemExit(f"prepared {len(records)} records, expected {args.max_samples}")
    write_jsonl(args.out, records)
    print(f"wrote {args.out} records={len(records)}")


if __name__ == "__main__":
    main()

