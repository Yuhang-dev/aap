from __future__ import annotations

import os
from pathlib import Path
from typing import Any


WEIGHT_SUFFIXES = (".safetensors", ".bin")


def model_cache_dir(model_id: str, hub_cache: str | Path | None = None) -> Path:
    if hub_cache is None:
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hub_cache = os.environ.get("HF_HUB_CACHE", str(hf_home / "hub"))
    return Path(hub_cache) / ("models--" + model_id.replace("/", "--"))


def file_size_following_symlink(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def audit_model_cache(model_id: str, hub_cache: str | Path | None = None) -> dict[str, Any]:
    cache_dir = model_cache_dir(model_id, hub_cache)
    snapshots_dir = cache_dir / "snapshots"
    snapshot_dirs = sorted([p for p in snapshots_dir.iterdir() if p.is_dir()]) if snapshots_dir.exists() else []

    snapshot_reports = []
    total_weight_bytes = 0
    total_weight_files = 0
    broken_paths: list[str] = []
    for snapshot in snapshot_dirs:
        files = [p for p in snapshot.rglob("*") if p.is_file() or p.is_symlink()]
        weight_files = [p for p in files if p.name.endswith(WEIGHT_SUFFIXES)]
        broken = [p for p in files if p.is_symlink() and not p.exists()]
        weight_bytes = sum(file_size_following_symlink(p) for p in weight_files)
        total_weight_bytes += weight_bytes
        total_weight_files += len(weight_files)
        broken_paths.extend(str(p) for p in broken)
        snapshot_reports.append(
            {
                "snapshot": snapshot.name,
                "file_count": len(files),
                "weight_file_count": len(weight_files),
                "weight_total_gb": round(weight_bytes / (1024**3), 3),
                "weight_files": sorted(p.name for p in weight_files),
                "broken_symlink_count": len(broken),
            }
        )

    return {
        "model_id": model_id,
        "cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "snapshot_count": len(snapshot_dirs),
        "snapshots": snapshot_reports,
        "weight_file_count": total_weight_files,
        "weight_total_gb": round(total_weight_bytes / (1024**3), 3),
        "broken_symlink_count": len(broken_paths),
        "broken_symlinks": broken_paths[:20],
        "looks_complete": cache_dir.exists()
        and len(snapshot_dirs) > 0
        and total_weight_files > 0
        and not broken_paths,
    }

