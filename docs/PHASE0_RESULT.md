# Phase 0 Result

Remote run date: 2026-06-19

## Gate Output

```json
{
  "theory_branch_alive": true,
  "num_alive_grid_points": "8 / 49",
  "mean_gap_over_grid": 0.0012235970770096655,
  "mean_mask_difference_rate_over_grid": 0.08928571428571429,
  "max_gap_point": {
    "rho": 0.85,
    "angle_degrees": 0.0,
    "mean_gap": 0.00838507116040027,
    "mean_compensation_gap": 0.00838507116040027,
    "mean_total_gap": 0.010729460360020943,
    "mask_difference_rate": 0.25,
    "substantial_gap_rate": 0.25,
    "seeds": 8
  }
}
```

## Interpretation

The strict task-budgeted gate passes, but weakly. The bilevel / `M-grad-OBS`
theory branch remains alive as an existence result, especially at high coupling.
Do not frame Phase 0 as evidence that the gap is broad or large across the
whole grid.

Proceed to Phase 1 to establish whether the empirical phenomenon exists on
7B-scale instruction models.

