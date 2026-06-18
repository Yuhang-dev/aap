from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def run_command(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": f"{command[0]} not found",
        }
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "available": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def torch_cuda_report() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"importable": False, "error": repr(exc)}

    report: dict[str, Any] = {
        "importable": True,
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "devices": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            report["devices"].append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    return report


def disk_report(path: str) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
    }


def model_cache_report(model_id: str) -> dict[str, Any]:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hub = Path(os.environ.get("HF_HUB_CACHE", str(hf_home / "hub")))
    cache_name = "models--" + model_id.replace("/", "--")
    model_dir = hub / cache_name
    snapshots = model_dir / "snapshots"
    snapshot_dirs = sorted([p.name for p in snapshots.iterdir()]) if snapshots.exists() else []
    return {
        "model_id": model_id,
        "hub_cache": str(hub),
        "cache_dir": str(model_dir),
        "cache_dir_exists": model_dir.exists(),
        "snapshot_count": len(snapshot_dirs),
        "snapshots": snapshot_dirs[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 1 GPU/model-cache readiness.")
    parser.add_argument("--out", default="outputs/phase1_preflight.json")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
    )
    args = parser.parse_args()

    data_disk = os.environ.get("DATA_DISK", "/root/autodl-tmp")
    report: dict[str, Any] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "environment": {
            "DATA_DISK": os.environ.get("DATA_DISK"),
            "AAP_ROOT": os.environ.get("AAP_ROOT"),
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "TORCH_HOME": os.environ.get("TORCH_HOME"),
        },
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "numpy": package_version("numpy"),
            "yaml": package_version("yaml"),
            "matplotlib": package_version("matplotlib"),
        },
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "torch_cuda": torch_cuda_report(),
        "disk": {
            "data_disk": disk_report(data_disk),
            "cwd": disk_report(str(Path.cwd())),
        },
        "model_caches": [model_cache_report(model_id) for model_id in args.models],
    }

    checks = {
        "python_is_pbp": "/envs/pbp/bin/python" in sys.executable,
        "cuda_available": bool(report["torch_cuda"].get("cuda_available")),
        "nvidia_smi_ok": report["nvidia_smi"]["returncode"] == 0,
        "all_model_caches_present": all(item["cache_dir_exists"] for item in report["model_caches"]),
    }
    report["checks"] = checks
    report["ready_for_phase1_gpu"] = all(checks.values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps({"checks": checks, "ready_for_phase1_gpu": report["ready_for_phase1_gpu"]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

