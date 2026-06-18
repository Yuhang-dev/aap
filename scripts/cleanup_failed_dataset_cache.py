from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


TARGET_MARKERS = (
    "allenai___c4",
    "Salesforce___wikitext",
    "wikitext",
)


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def candidate_dirs(cache_dir: Path) -> list[Path]:
    candidates: set[Path] = set()
    if not cache_dir.exists():
        return []

    for child in cache_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if any(marker in name for marker in TARGET_MARKERS):
            candidates.add(child)

    downloads = cache_dir / "downloads"
    if downloads.exists():
        for lock_or_incomplete in downloads.glob("*.lock"):
            candidates.add(lock_or_incomplete)
        for incomplete in downloads.glob("*.incomplete"):
            candidates.add(incomplete)

    return sorted(candidates, key=lambda p: str(p))


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def main() -> None:
    default_cache = os.environ.get(
        "HF_DATASETS_CACHE",
        str(Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "datasets"),
    )
    parser = argparse.ArgumentParser(description="Clean failed Phase 1 dataset builder caches.")
    parser.add_argument("--cache-dir", default=default_cache)
    parser.add_argument("--delete", action="store_true", help="Actually delete candidates. Default is dry-run.")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    expected_prefix = Path("/root/autodl-tmp").resolve()
    if expected_prefix not in [cache_dir, *cache_dir.parents]:
        raise SystemExit(f"refusing to clean outside /root/autodl-tmp: {cache_dir}")

    candidates = candidate_dirs(cache_dir)
    if not candidates:
        print(f"No cleanup candidates found under {cache_dir}")
        return

    print(f"cache_dir={cache_dir}")
    print(f"mode={'delete' if args.delete else 'dry-run'}")
    total = 0
    for path in candidates:
        size = dir_size(path) if path.is_dir() else path.stat().st_size
        total += size
        print(f"{format_bytes(size)}\t{path}")
        if args.delete:
            remove_path(path)
    print(f"total_candidate_size={format_bytes(total)}")


if __name__ == "__main__":
    main()

