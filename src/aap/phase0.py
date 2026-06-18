from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import cos, sin
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MaskEvaluation:
    removed: tuple[int, ...]
    task_loss: float
    alignment_change: float
    compensation_alignment_change: float
    objective: float


@dataclass(frozen=True)
class SweepPoint:
    rho: float
    angle_degrees: float
    mean_gap: float
    mean_total_gap: float
    mean_compensation_gap: float
    median_gap: float
    mean_relative_gap: float
    mask_difference_rate: float
    substantial_gap_rate: float
    seeds: int


def make_coupled_hessian(dim: int, rho: float, condition_number: float = 20.0) -> np.ndarray:
    """Create a positive definite Hessian with tunable dense coupling.

    rho=0 gives a diagonal Hessian. Larger rho rotates the eigenspace away from
    the coordinate axes while preserving positive eigenvalues.
    """

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    if dim < 2:
        raise ValueError("dim must be at least 2")
    if condition_number < 1.0:
        raise ValueError("condition_number must be >= 1")

    eigvals = np.geomspace(1.0, condition_number, dim)
    diagonal = np.diag(eigvals)
    if rho == 0.0:
        return diagonal

    centered = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (dim, dim), dtype=float)
    q, _ = np.linalg.qr(centered)
    hessian = q @ diagonal @ q.T
    return 0.5 * (hessian + hessian.T)


def make_alignment_gradient(
    hessian: np.ndarray,
    angle_degrees: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct g_align at a controlled angle from the lowest-curvature eigenvector."""

    eigvals, eigvecs = np.linalg.eigh(hessian)
    low = eigvecs[:, np.argmin(eigvals)]
    random_dir = rng.normal(size=hessian.shape[0])
    random_dir = random_dir - low * float(random_dir @ low)
    norm = np.linalg.norm(random_dir)
    if norm < 1e-12:
        random_dir = eigvecs[:, np.argmax(eigvals)]
    else:
        random_dir = random_dir / norm
    angle = np.deg2rad(angle_degrees)
    grad = cos(angle) * low + sin(angle) * random_dir
    grad_norm = np.linalg.norm(grad)
    if grad_norm == 0.0:
        raise ValueError("constructed zero alignment gradient")
    return grad / grad_norm


def block_obs_solution(theta: np.ndarray, h_inv: np.ndarray, removed: Iterable[int]) -> np.ndarray:
    removed_idx = tuple(int(i) for i in removed)
    if not removed_idx:
        return theta.copy()

    h_ss = h_inv[np.ix_(removed_idx, removed_idx)]
    rhs = theta[list(removed_idx)]
    lagrange = np.linalg.solve(h_ss, rhs)
    correction = h_inv[:, removed_idx] @ lagrange
    updated = theta - correction
    updated[list(removed_idx)] = 0.0
    return updated


def evaluate_removed_set(
    theta: np.ndarray,
    hessian: np.ndarray,
    h_inv: np.ndarray,
    g_align: np.ndarray,
    removed: Iterable[int],
    beta: float,
    objective_alignment_metric: str = "total",
) -> MaskEvaluation:
    removed_tuple = tuple(sorted(int(i) for i in removed))
    theta_star = block_obs_solution(theta, h_inv, removed_tuple)
    delta = theta_star - theta
    compensation_delta = delta.copy()
    compensation_delta[list(removed_tuple)] = 0.0
    task_loss = 0.5 * float(delta @ hessian @ delta)
    alignment_change = float(g_align @ delta)
    compensation_alignment_change = float(g_align @ compensation_delta)
    objective = alignment_metric_value(
        alignment_change,
        compensation_alignment_change,
        objective_alignment_metric,
    ) + beta * task_loss
    return MaskEvaluation(
        removed=removed_tuple,
        task_loss=task_loss,
        alignment_change=alignment_change,
        compensation_alignment_change=compensation_alignment_change,
        objective=objective,
    )


def alignment_metric_value(
    alignment_change: float,
    compensation_alignment_change: float,
    metric: str,
) -> float:
    if metric == "total":
        return alignment_change
    if metric == "compensation":
        return compensation_alignment_change
    raise ValueError(f"unknown alignment metric: {metric}")


def evaluation_alignment_metric(evaluation: MaskEvaluation, metric: str) -> float:
    return alignment_metric_value(
        evaluation.alignment_change,
        evaluation.compensation_alignment_change,
        metric,
    )


def greedy_obs_mask(theta: np.ndarray, h_inv: np.ndarray, k: int) -> tuple[int, ...]:
    if not 1 <= k <= theta.shape[0]:
        raise ValueError("k must be in [1, dim]")
    single_task = theta * theta / (2.0 * np.diag(h_inv))
    return tuple(sorted(np.argsort(single_task)[:k].astype(int).tolist()))


def optimal_alignment_mask(
    theta: np.ndarray,
    hessian: np.ndarray,
    h_inv: np.ndarray,
    g_align: np.ndarray,
    k: int,
    beta: float,
    alignment_metric: str,
) -> MaskEvaluation:
    best: MaskEvaluation | None = None
    for removed in combinations(range(theta.shape[0]), k):
        score = evaluate_removed_set(
            theta,
            hessian,
            h_inv,
            g_align,
            removed,
            beta,
            objective_alignment_metric=alignment_metric,
        )
        if best is None or score.objective < best.objective:
            best = score
    if best is None:
        raise RuntimeError("no candidate masks evaluated")
    return best


def optimal_alignment_mask_with_task_budget(
    theta: np.ndarray,
    hessian: np.ndarray,
    h_inv: np.ndarray,
    g_align: np.ndarray,
    k: int,
    beta: float,
    alignment_metric: str,
    task_budget: float,
) -> MaskEvaluation:
    best: MaskEvaluation | None = None
    for removed in combinations(range(theta.shape[0]), k):
        score = evaluate_removed_set(
            theta,
            hessian,
            h_inv,
            g_align,
            removed,
            beta,
            objective_alignment_metric=alignment_metric,
        )
        if score.task_loss > task_budget:
            continue
        if best is None:
            best = score
            continue
        score_metric = evaluation_alignment_metric(score, alignment_metric)
        best_metric = evaluation_alignment_metric(best, alignment_metric)
        if (score_metric, score.task_loss) < (best_metric, best.task_loss):
            best = score
    if best is None:
        raise RuntimeError("no candidate masks inside task budget")
    return best


def run_single_trial(
    dim: int,
    k: int,
    rho: float,
    angle_degrees: float,
    beta: float,
    condition_number: float,
    selection_mode: str,
    alignment_metric: str,
    task_budget_multiplier: float,
    rng: np.random.Generator,
) -> tuple[MaskEvaluation, MaskEvaluation]:
    hessian = make_coupled_hessian(dim=dim, rho=rho, condition_number=condition_number)
    h_inv = np.linalg.inv(hessian)
    theta = rng.normal(size=dim)
    theta = theta / max(np.linalg.norm(theta), 1e-12)
    g_align = make_alignment_gradient(hessian, angle_degrees, rng)

    greedy_removed = greedy_obs_mask(theta, h_inv, k)
    greedy_eval = evaluate_removed_set(
        theta,
        hessian,
        h_inv,
        g_align,
        greedy_removed,
        beta,
        objective_alignment_metric=alignment_metric,
    )
    if selection_mode == "beta":
        aware_eval = optimal_alignment_mask(
            theta,
            hessian,
            h_inv,
            g_align,
            k,
            beta,
            alignment_metric=alignment_metric,
        )
    elif selection_mode == "task_budget":
        aware_eval = optimal_alignment_mask_with_task_budget(
            theta,
            hessian,
            h_inv,
            g_align,
            k,
            beta,
            alignment_metric=alignment_metric,
            task_budget=greedy_eval.task_loss * task_budget_multiplier + 1e-12,
        )
    else:
        raise ValueError(f"unknown selection mode: {selection_mode}")
    return greedy_eval, aware_eval


def sweep_toy_gap(
    dim: int,
    k: int,
    rhos: Iterable[float],
    angles_degrees: Iterable[float],
    seeds: Iterable[int],
    beta: float,
    condition_number: float,
    substantial_gap: float,
    selection_mode: str,
    alignment_metric: str,
    task_budget_multiplier: float,
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for rho in rhos:
        for angle in angles_degrees:
            gaps: list[float] = []
            total_gaps: list[float] = []
            compensation_gaps: list[float] = []
            relative_gaps: list[float] = []
            mask_diffs = 0
            substantial = 0
            seed_count = 0
            for seed in seeds:
                rng = np.random.default_rng(seed)
                greedy_eval, aware_eval = run_single_trial(
                    dim=dim,
                    k=k,
                    rho=float(rho),
                    angle_degrees=float(angle),
                    beta=beta,
                    condition_number=condition_number,
                    selection_mode=selection_mode,
                    alignment_metric=alignment_metric,
                    task_budget_multiplier=task_budget_multiplier,
                    rng=rng,
                )
                total_gap = greedy_eval.alignment_change - aware_eval.alignment_change
                compensation_gap = (
                    greedy_eval.compensation_alignment_change
                    - aware_eval.compensation_alignment_change
                )
                gap = evaluation_alignment_metric(greedy_eval, alignment_metric) - evaluation_alignment_metric(
                    aware_eval,
                    alignment_metric,
                )
                denominator = abs(evaluation_alignment_metric(greedy_eval, alignment_metric)) + 1e-12
                gaps.append(gap)
                total_gaps.append(total_gap)
                compensation_gaps.append(compensation_gap)
                relative_gaps.append(gap / denominator)
                mask_diffs += int(greedy_eval.removed != aware_eval.removed)
                substantial += int(gap > substantial_gap)
                seed_count += 1

            points.append(
                SweepPoint(
                    rho=float(rho),
                    angle_degrees=float(angle),
                    mean_gap=float(np.mean(gaps)),
                    mean_total_gap=float(np.mean(total_gaps)),
                    mean_compensation_gap=float(np.mean(compensation_gaps)),
                    median_gap=float(np.median(gaps)),
                    mean_relative_gap=float(np.mean(relative_gaps)),
                    mask_difference_rate=mask_diffs / seed_count,
                    substantial_gap_rate=substantial / seed_count,
                    seeds=seed_count,
                )
            )
    return points


def summarize_gate(points: list[SweepPoint], min_gap: float, min_mask_diff_rate: float) -> dict[str, object]:
    if not points:
        raise ValueError("cannot summarize empty sweep")

    max_gap_point = max(points, key=lambda p: p.mean_gap)
    max_diff_point = max(points, key=lambda p: p.mask_difference_rate)
    mean_gap = float(np.mean([p.mean_gap for p in points]))
    mean_diff = float(np.mean([p.mask_difference_rate for p in points]))
    alive_points = [
        p
        for p in points
        if p.mean_gap >= min_gap and p.mask_difference_rate >= min_mask_diff_rate
    ]

    return {
        "theory_branch_alive": bool(alive_points),
        "decision_rule": {
            "min_gap": min_gap,
            "min_mask_difference_rate": min_mask_diff_rate,
        },
        "num_alive_grid_points": len(alive_points),
        "num_grid_points": len(points),
        "mean_gap_over_grid": mean_gap,
        "mean_mask_difference_rate_over_grid": mean_diff,
        "max_gap_point": max_gap_point.__dict__,
        "max_mask_difference_point": max_diff_point.__dict__,
        "interpretation": (
            "Gap is substantial in at least one coupling/angle region; keep the "
            "bilevel/M-grad-OBS branch alive as an existence result."
            if alive_points
            else "Gap did not pass the configured threshold; do not rely on the "
            "bilevel mask-gap narrative without changing the evidence."
        ),
    }
