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

from aap.hf_cache import audit_model_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local Hugging Face model cache completeness.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
    )
    parser.add_argument("--hub-cache", default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    reports = [audit_model_cache(model, hub_cache=args.hub_cache) for model in args.models]
    payload = {"hub_cache": args.hub_cache, "models": reports}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

