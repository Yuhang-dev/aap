from __future__ import annotations

import subprocess
from pathlib import Path


WANDA_REPO_URL = "https://github.com/locuslab/wanda.git"
WANDA_COMMIT = "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6"


def run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def ensure_wanda_repo(target_dir: str | Path, commit: str = WANDA_COMMIT) -> Path:
    target = Path(target_dir)
    if target.exists() and not (target / ".git").exists():
        raise ValueError(f"target exists but is not a git repository: {target}")

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run_git(["clone", WANDA_REPO_URL, str(target)])

    run_git(["fetch", "origin"], cwd=target)
    run_git(["checkout", commit], cwd=target)
    actual = run_git(["rev-parse", "HEAD"], cwd=target)
    if actual != commit:
        raise RuntimeError(f"Wanda checkout mismatch: expected {commit}, got {actual}")
    return target

