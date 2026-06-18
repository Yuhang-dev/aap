# Remote Commands

Remote project directory:

```bash
/root/autodl-tmp/aap
```

## Bootstrap

```bash
cd /root/autodl-tmp
git clone <repo-url> aap
cd /root/autodl-tmp/aap
conda activate pbp
python -m pip install -e . --no-build-isolation
```

If copying files manually instead of using git, place this project at
`/root/autodl-tmp/aap` before running commands.

For Phase 0, editable install is optional. If the package mirror cannot resolve
build dependencies, use `PYTHONPATH` as shown below.

## Phase 0 Toy Gap

```bash
cd /root/autodl-tmp/aap
conda activate pbp
export PYTHONPATH=/root/autodl-tmp/aap/src:${PYTHONPATH:-}
python scripts/run_phase0_toy_gap.py \
  --config configs/phase0_toy_gap.yaml \
  --out-dir outputs/phase0_toy_gap
```

Outputs:

```text
outputs/phase0_toy_gap/phase0_summary.json
outputs/phase0_toy_gap/phase0_points.csv
outputs/phase0_toy_gap/gap_heatmap.csv
outputs/phase0_toy_gap/gap_heatmap.png
outputs/phase0_toy_gap/mask_difference_heatmap.csv
outputs/phase0_toy_gap/mask_difference_heatmap.png
```

## Phase 1 Placeholder

Do not start Phase 1 until Phase 0 gate is read. The first Phase 1 model should
be:

```text
Qwen/Qwen2.5-7B-Instruct
```

The remote cache already includes:

```text
Qwen/Qwen2.5-7B
Qwen/Qwen2.5-7B-Instruct
```
