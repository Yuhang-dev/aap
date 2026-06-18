# Phase 1 Wanda Smoke Result

Remote run date: 2026-06-19

## Setup

```text
model: Qwen/Qwen2.5-7B-Instruct
method: Wanda
sparsity: 10% unstructured
calibration: C4 streaming
nsamples: 8
seqlen: 2048
dtype: bfloat16
ppl eval: WikiText-2, max 16 segments
save_model: false
```

## Result

```json
{
  "actual_sparsity": 0.09991475674915636,
  "dtype": "bfloat16",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "nsamples": 8,
  "ppl_max_samples": 16,
  "prune_method": "wanda",
  "runtime_seconds": 13.713129043579102,
  "saved_model": null,
  "seed": 0,
  "seqlen": 2048,
  "sparsity_ratio_target": 0.1,
  "sparsity_type": "unstructured",
  "wanda_dir": "external/wanda",
  "wikitext2_ppl": 6.968544006347656
}
```

## Interpretation

The Phase 1 Wanda path is operational on Qwen2.5-7B-Instruct. The adapter can:

- load the cached 7B model;
- stream C4 calibration samples;
- apply unstructured Wanda masks;
- verify actual sparsity;
- run the AAP WikiText-2 PPL path.

Proceed to the Phase 1 sparsity sweep.

The full sweep must include a dense PPL baseline before comparing pruning
points.
