# Alignment-Aware Pruning

Executable code for the staged Alignment-Aware Pruning research plan.

The project is organized around gated phases. Do not run later experimental
matrices until the earlier gate has been checked.

## Remote Location

Use this directory on the remote machine:

```bash
/root/autodl-tmp/aap
```

The remote environment already has Qwen2.5 7B base and instruct model caches
under the configured Hugging Face cache.

## Local Policy

Local execution is limited to key code inspection and static syntax checks.
Do not load large models, download datasets, or run GPU experiments locally.

Allowed local check:

```bash
python -m compileall src scripts
```

## Setup

Editable install is optional for Phase 0. If the package index cannot provide
build dependencies, run with `PYTHONPATH=src` instead.

```bash
cd /root/autodl-tmp/aap
python -m pip install -e . --no-build-isolation
```

## Phase 0: Toy Gap Gate

Run the analytical CPU-only gate:

```bash
export PYTHONPATH=/root/autodl-tmp/aap/src:${PYTHONPATH:-}
python scripts/run_phase0_toy_gap.py \
  --config configs/phase0_toy_gap.yaml \
  --out-dir outputs/phase0_toy_gap
```

Expected artifacts:

```text
outputs/phase0_toy_gap/
  phase0_summary.json
  gap_heatmap.csv
  mask_difference_heatmap.csv
  gap_heatmap.png
```

Decision rule:

- If the set-level alignment gap is substantial at non-trivial coupling, keep
  the bilevel / M-grad-OBS theory branch alive.
- If the gap is near zero across the sweep, drop the bilevel framing and treat
  M-ACA as an empirical calibration-distribution method only.

## Next Phase

After Phase 0 passes, Phase 1 should establish the operating sparsity band on
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. The first remote target is
Qwen2.5-7B-Instruct because it is already cached remotely.
