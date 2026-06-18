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

