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

from aap.wanda_adapter import WandaRunConfig, run_wanda


def optional_path(value: str | None) -> Path | None:
    if value is None or value == "":
        return None
    return Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Phase 1 Wanda pruning/PPL job.")
    parser.add_argument("--wanda-dir", default="external/wanda")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--sparsity-ratio", type=float, required=True)
    parser.add_argument("--sparsity-type", default="unstructured", choices=["unstructured", "2:4", "4:8"])
    parser.add_argument("--prune-method", default="wanda", choices=["wanda", "magnitude", "sparsegpt"])
    parser.add_argument("--use-variant", action="store_true")
    parser.add_argument("--eval-ppl", action="store_true")
    parser.add_argument("--save-model", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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
        prune_method=args.prune_method,
        use_variant=args.use_variant,
        eval_ppl=args.eval_ppl,
        save_model=optional_path(args.save_model),
        out=Path(args.out),
    )
    result = run_wanda(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

