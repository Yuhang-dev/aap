# Phase 1 Wanda PPL Sweep Result

Remote run date: 2026-06-19

## Setup

```text
model: Qwen/Qwen2.5-7B-Instruct
method: Wanda(C4)
calibration: C4 streaming
nsamples: 128
seqlen: 2048
dtype: bfloat16
eval: WikiText-2 PPL
save_model: false
```

## Results

Dense baseline:

```text
PPL = 7.458973407745361
```

| sparsity type | target | actual | PPL | delta PPL | relative PPL increase |
|---|---:|---:|---:|---:|---:|
| dense | 0.00 | 0.000000 | 7.458973 | 0.000000 | 0.0% |
| unstructured | 0.10 | 0.099915 | 7.466560 | 0.007586 | 0.1% |
| unstructured | 0.20 | 0.199830 | 7.525525 | 0.066552 | 0.9% |
| unstructured | 0.30 | 0.299957 | 7.711824 | 0.252851 | 3.4% |
| unstructured | 0.40 | 0.399872 | 8.112684 | 0.653711 | 8.8% |
| unstructured | 0.50 | 0.500000 | 9.198289 | 1.739316 | 23.3% |
| unstructured | 0.60 | 0.599915 | 14.274823 | 6.815850 | 91.4% |
| unstructured | 0.70 | 0.699830 | 73.226486 | 65.767513 | 881.7% |
| 2:4 | 0.50 | 0.500000 | 15.758570 | 8.299596 | 111.3% |

## Interpretation

PPL-only candidate band:

```text
primary: 30%
stress/edge: 40%
boundary-collapse probe: 50%
```

Do not use 60%, 70%, or 2:4 as evidence for "QA preserved" on PPL. They may
still be useful stress points, but they are outside the likely operating regime
for the Phase 1 phenomenon claim.

Next step: run cheap alignment evaluation via HH-RLHF BCR for 30%, 40%, and
50%, then decide which sparsity points deserve the full alignment suite
(XSTest-FPR, IFEval, TruthfulQA, HarmBench/AdvBench).

