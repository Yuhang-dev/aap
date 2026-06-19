from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    report = {
        "python": sys.executable,
        "lm_eval_available": module_available("lm_eval"),
        "accelerate_available": module_available("accelerate"),
        "transformers_available": module_available("transformers"),
        "datasets_available": module_available("datasets"),
        "langdetect_available": module_available("langdetect"),
        "immutabledict_available": module_available("immutabledict"),
        "lm_eval_cli": shutil.which("lm_eval"),
        "lm_eval_module": shutil.which("lm-eval"),
    }
    out = Path("outputs/phase1/eval_preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
