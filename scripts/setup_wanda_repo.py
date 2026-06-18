from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aap.external import WANDA_COMMIT, ensure_wanda_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone or update the pinned official Wanda repo.")
    parser.add_argument("--target-dir", default="external/wanda")
    parser.add_argument("--commit", default=WANDA_COMMIT)
    args = parser.parse_args()

    target = ensure_wanda_repo(args.target_dir, commit=args.commit)
    print(f"wanda_dir={target}")
    print(f"commit={args.commit}")


if __name__ == "__main__":
    main()

