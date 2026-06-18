# Remote Environment Target

Target path:

```text
/root/autodl-tmp/aap
```

Known environment from the handoff:

```text
conda env: pbp
python: 3.11.15
torch: 2.12.0+cu130
transformers: 5.12.1
datasets: 5.0.0
HF_HOME: /root/autodl-tmp/hf_cache
```

Cached models include:

```text
Qwen/Qwen2.5-7B
Qwen/Qwen2.5-7B-Instruct
```

Phase 0 is CPU-only and does not depend on Torch, CUDA, model caches, or
datasets.

All new downloads should remain under `/root/autodl-tmp`:

```text
project: /root/autodl-tmp/aap
external repos: /root/autodl-tmp/aap/external
HF_HOME: /root/autodl-tmp/hf_cache
HF_HUB_CACHE: /root/autodl-tmp/hf_cache/hub
HF_DATASETS_CACHE: /root/autodl-tmp/hf_cache/datasets
TORCH_HOME: /root/autodl-tmp/torch_cache
HF_XET_CACHE: /root/autodl-tmp/hf_cache/xet
```
