from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aap.io import ensure_dir, read_yaml, write_json, write_matrix_csv
from aap.phase0 import summarize_gate, sweep_toy_gap


def parse_float_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [float(item) for item in value]


def parse_int_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return [int(item) for item in value]


def point_rows(points):
    for point in points:
        yield {
            "rho": point.rho,
            "angle_degrees": point.angle_degrees,
            "mean_gap": point.mean_gap,
            "mean_total_gap": point.mean_total_gap,
            "mean_compensation_gap": point.mean_compensation_gap,
            "median_gap": point.median_gap,
            "mean_relative_gap": point.mean_relative_gap,
            "mask_difference_rate": point.mask_difference_rate,
            "substantial_gap_rate": point.substantial_gap_rate,
            "seeds": point.seeds,
        }


def write_heatmap_csv(path: Path, points, value_name: str) -> None:
    rows = [
        {
            "rho": point.rho,
            "angle_degrees": point.angle_degrees,
            value_name: getattr(point, value_name),
        }
        for point in points
    ]
    write_matrix_csv(path, rows, ["rho", "angle_degrees", value_name])


def plot_gap_heatmap(path: Path, points, value_name: str, title: str) -> None:
    rhos = sorted({point.rho for point in points})
    angles = sorted({point.angle_degrees for point in points})
    grid = np.full((len(angles), len(rhos)), np.nan)
    for point in points:
        row = angles.index(point.angle_degrees)
        col = rhos.index(point.rho)
        grid[row, col] = getattr(point, value_name)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(rhos)), [f"{rho:.2f}" for rho in rhos])
    ax.set_yticks(range(len(angles)), [f"{angle:.0f}" for angle in angles])
    ax.set_xlabel("coupling rho")
    ax.set_ylabel("angle from low-curvature direction")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=value_name)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 0 toy OBS/alignment mask-gap sweep.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = read_yaml(args.config)
    out_dir = ensure_dir(args.out_dir)

    dim = int(config.get("dim", 12))
    k = int(config.get("k", 3))
    beta = float(config.get("beta", 0.1))
    condition_number = float(config.get("condition_number", 20.0))
    substantial_gap = float(config.get("substantial_gap", 0.02))
    gate_min_gap = float(config.get("gate_min_gap", 0.02))
    gate_min_mask_diff_rate = float(config.get("gate_min_mask_diff_rate", 0.25))
    selection_mode = str(config.get("selection_mode", "task_budget"))
    alignment_metric = str(config.get("alignment_metric", "compensation"))
    task_budget_multiplier = float(config.get("task_budget_multiplier", 1.05))
    rhos = parse_float_list(config.get("rhos"), "rhos")
    angles = parse_float_list(config.get("angles_degrees"), "angles_degrees")
    seeds = parse_int_list(config.get("seeds"), "seeds")

    points = sweep_toy_gap(
        dim=dim,
        k=k,
        rhos=rhos,
        angles_degrees=angles,
        seeds=seeds,
        beta=beta,
        condition_number=condition_number,
        substantial_gap=substantial_gap,
        selection_mode=selection_mode,
        alignment_metric=alignment_metric,
        task_budget_multiplier=task_budget_multiplier,
    )
    summary = summarize_gate(
        points,
        min_gap=gate_min_gap,
        min_mask_diff_rate=gate_min_mask_diff_rate,
    )
    summary["config"] = {
        "dim": dim,
        "k": k,
        "beta": beta,
        "condition_number": condition_number,
        "substantial_gap": substantial_gap,
        "selection_mode": selection_mode,
        "alignment_metric": alignment_metric,
        "task_budget_multiplier": task_budget_multiplier,
        "rhos": rhos,
        "angles_degrees": angles,
        "seeds": seeds,
    }

    write_matrix_csv(
        out_dir / "phase0_points.csv",
        point_rows(points),
        [
            "rho",
            "angle_degrees",
            "mean_gap",
            "mean_total_gap",
            "mean_compensation_gap",
            "median_gap",
            "mean_relative_gap",
            "mask_difference_rate",
            "substantial_gap_rate",
            "seeds",
        ],
    )
    write_heatmap_csv(out_dir / "gap_heatmap.csv", points, "mean_gap")
    write_heatmap_csv(out_dir / "total_gap_heatmap.csv", points, "mean_total_gap")
    write_heatmap_csv(out_dir / "compensation_gap_heatmap.csv", points, "mean_compensation_gap")
    write_heatmap_csv(out_dir / "mask_difference_heatmap.csv", points, "mask_difference_rate")
    write_json(out_dir / "phase0_summary.json", summary)
    plot_gap_heatmap(out_dir / "gap_heatmap.png", points, "mean_gap", "Phase 0 Alignment Gap")
    plot_gap_heatmap(
        out_dir / "mask_difference_heatmap.png",
        points,
        "mask_difference_rate",
        "Phase 0 Mask Difference Rate",
    )

    print(f"wrote {out_dir / 'phase0_summary.json'}")
    print(f"theory_branch_alive={summary['theory_branch_alive']}")


if __name__ == "__main__":
    main()
