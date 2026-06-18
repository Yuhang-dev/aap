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
    objective: float


@dataclass(frozen=True)
class SweepPoint:
    rho: float
    angle_degrees: float
    mean_gap: float
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
) -> MaskEvaluation:
    removed_tuple = tuple(sorted(int(i) for i in removed))
    theta_star = block_obs_solution(theta, h_inv, removed_tuple)
    delta = theta_star - theta
    task_loss = 0.5 * float(delta @ hessian @ delta)
    alignment_change = float(g_align @ delta)
    objective = alignment_change + beta * task_loss
    return MaskEvaluation(
        removed=removed_tuple,
        task_loss=task_loss,
        alignment_change=alignment_change,
        objective=objective,
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
) -> MaskEvaluation:
    best: MaskEvaluation | None = None
    for removed in combinations(range(theta.shape[0]), k):
        score = evaluate_removed_set(theta, hessian, h_inv, g_align, removed, beta)
        if best is None or score.objective < best.objective:
            best = score
    if best is None:
        raise RuntimeError("no candidate masks evaluated")
    return best


def run_single_trial(
    dim: int,
    k: int,
    rho: float,
    angle_degrees: float,
    beta: float,
    condition_number: float,
    rng: np.random.Generator,
) -> tuple[MaskEvaluation, MaskEvaluation]:
    hessian = make_coupled_hessian(dim=dim, rho=rho, condition_number=condition_number)
    h_inv = np.linalg.inv(hessian)
    theta = rng.normal(size=dim)
    theta = theta / max(np.linalg.norm(theta), 1e-12)
    g_align = make_alignment_gradient(hessian, angle_degrees, rng)

    greedy_removed = greedy_obs_mask(theta, h_inv, k)
    greedy_eval = evaluate_removed_set(theta, hessian, h_inv, g_align, greedy_removed, beta)
    aware_eval = optimal_alignment_mask(theta, hessian, h_inv, g_align, k, beta)
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
) -> list[SweepPoint]:
    points: list[SweepPoint] = []
    for rho in rhos:
        for angle in angles_degrees:
            gaps: list[float] = []
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
                    rng=rng,
                )
                gap = greedy_eval.alignment_change - aware_eval.alignment_change
                denominator = abs(greedy_eval.alignment_change) + 1e-12
                gaps.append(gap)
                relative_gaps.append(gap / denominator)
                mask_diffs += int(greedy_eval.removed != aware_eval.removed)
                substantial += int(gap > substantial_gap)
                seed_count += 1

            points.append(
                SweepPoint(
                    rho=float(rho),
                    angle_degrees=float(angle),
                    mean_gap=float(np.mean(gaps)),
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

