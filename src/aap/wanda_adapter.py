from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def add_wanda_to_path(wanda_dir: str | Path) -> Path:
    root = Path(wanda_dir).resolve()
    if not (root / "lib").exists():
        raise FileNotFoundError(f"Wanda lib directory not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


@dataclass(frozen=True)
class WandaRunConfig:
    wanda_dir: Path
    model: str
    cache_dir: str | None
    dtype: str
    seed: int
    nsamples: int
    seqlen: int
    sparsity_ratio: float
    sparsity_type: str
    prune_method: str
    use_variant: bool
    eval_ppl: bool
    ppl_max_samples: int | None
    save_model: Path | None
    out: Path
    calibration_source: str = "c4"
    calibration_data: Path | None = None


def torch_dtype_from_name(name: str):
    import torch

    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def move_to_device(value: Any, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def find_linear_layers(module):
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return {"": module}
    result = {}
    for child_name, child in module.named_children():
        for name, layer in find_linear_layers(child).items():
            full_name = child_name if name == "" else f"{child_name}.{name}"
            result[full_name] = layer
    return result


def check_sparsity(model) -> float:
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    zero_count = 0
    param_count = 0
    for idx, layer in enumerate(layers):
        layer_zero = 0
        layer_params = 0
        for _, linear in find_linear_layers(layer).items():
            weights = linear.weight.data
            current_zero = int((weights == 0).sum().item())
            current_params = weights.numel()
            zero_count += current_zero
            param_count += current_params
            layer_zero += current_zero
            layer_params += current_params
        print(f"layer {idx} sparsity {layer_zero / layer_params:.6f}")
    model.config.use_cache = use_cache
    return zero_count / param_count


class CalibrationCatcherException(Exception):
    pass


def prepare_calibration_input(model, dataloader, device, nsamples: int):
    import torch
    import torch.nn as nn

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    cache: dict[str, Any] = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] >= nsamples:
                raise CalibrationCatcherException
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = dict(kwargs)
            raise CalibrationCatcherException

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        if cache["i"] >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except CalibrationCatcherException:
            pass
    layers[0] = layers[0].module

    if cache["i"] != nsamples:
        raise RuntimeError(f"captured {cache['i']} calibration samples, expected {nsamples}")

    outs = torch.zeros_like(inps)
    model.config.use_cache = use_cache
    return inps, outs, cache["layer_kwargs"]


def get_c4_calibration_loader(nsamples: int, seed: int, seqlen: int, tokenizer):
    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/c4",
        "en",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(seed=seed, buffer_size=max(1000, nsamples * 32))

    rng = random.Random(seed)
    loader = []
    attempts = 0
    max_attempts = max(10000, nsamples * 1000)
    for row in dataset:
        attempts += 1
        if attempts > max_attempts:
            break
        encoded = tokenizer(row["text"], return_tensors="pt")
        if encoded.input_ids.shape[1] <= seqlen:
            continue
        start = rng.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        end = start + seqlen
        inp = encoded.input_ids[:, start:end]
        target = inp.clone()
        target[:, :-1] = -100
        loader.append((inp, target))
        if len(loader) == nsamples:
            return loader

    if len(loader) != nsamples:
        raise RuntimeError(
            f"collected {len(loader)} C4 calibration samples after {attempts} attempts, "
            f"expected {nsamples}"
        )
    return loader


def build_segment_loader_from_texts(texts: list[str], nsamples: int, seed: int, seqlen: int, tokenizer):
    import torch

    rng = random.Random(seed)
    shuffled = [text for text in texts if text.strip()]
    if not shuffled:
        raise RuntimeError("no non-empty calibration texts")
    rng.shuffle(shuffled)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer has neither eos_token_id nor pad_token_id")

    required_tokens = nsamples * seqlen + seqlen
    token_ids: list[int] = []
    passes = 0
    while len(token_ids) < required_tokens and passes < max(8, nsamples):
        passes += 1
        for text in shuffled:
            encoded = tokenizer(text, add_special_tokens=False).input_ids
            if encoded:
                token_ids.extend(encoded)
                token_ids.append(int(eos_id))
            if len(token_ids) >= required_tokens:
                break

    if len(token_ids) < seqlen:
        raise RuntimeError(f"preference calibration text is too short: {len(token_ids)} tokens < seqlen={seqlen}")

    max_start = len(token_ids) - seqlen
    if max_start < nsamples:
        starts = [0 for _ in range(nsamples)]
    else:
        starts = [rng.randint(0, max_start) for _ in range(nsamples)]

    loader = []
    token_tensor = torch.tensor(token_ids, dtype=torch.long)
    for start in starts:
        inp = token_tensor[start : start + seqlen].unsqueeze(0)
        target = inp.clone()
        target[:, :-1] = -100
        loader.append((inp, target))
    return loader


def get_preference_calibration_loader(
    nsamples: int,
    seed: int,
    seqlen: int,
    tokenizer,
    data_path: str | Path,
    source: str,
):
    from aap.preference_data import read_preference_jsonl

    records = read_preference_jsonl(data_path)
    texts: list[str] = []
    for record in records:
        if source == "hh_chosen":
            texts.append(record.prompt + record.chosen)
        elif source == "hh_rejected":
            texts.append(record.prompt + record.rejected)
        elif source == "hh_pair":
            texts.append(record.prompt + record.chosen)
            texts.append(record.prompt + record.rejected)
        else:
            raise ValueError(f"unsupported preference calibration source: {source}")
    return build_segment_loader_from_texts(texts, nsamples, seed, seqlen, tokenizer)


def get_calibration_loader(args, nsamples: int, seed: int, seqlen: int, tokenizer):
    source = getattr(args, "calibration_source", "c4")
    if source == "c4":
        return get_c4_calibration_loader(
            nsamples=nsamples,
            seed=seed,
            seqlen=seqlen,
            tokenizer=tokenizer,
        )

    data_path = getattr(args, "calibration_data", None)
    if not data_path:
        raise ValueError(f"calibration source {source!r} requires --calibration-data")
    return get_preference_calibration_loader(
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
        data_path=data_path,
        source=source,
    )


def load_wikitext2_test_encoding(tokenizer, cache_dir: str | None = None):
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    parquet_path = hf_hub_download(
        repo_id="Salesforce/wikitext",
        filename="wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    dataset = load_dataset("parquet", data_files={"test": parquet_path}, split="test")
    text = "\n\n".join(dataset["text"])
    return tokenizer(text, return_tensors="pt")


def eval_wikitext2_ppl_aap(model, tokenizer, device, cache_dir: str | None = None, max_samples: int | None = None) -> float:
    import torch
    import torch.nn as nn

    print("evaluating on Salesforce/wikitext wikitext-2-raw-v1 test")
    testenc = load_wikitext2_test_encoding(tokenizer, cache_dir=cache_dir).input_ids
    nsamples = testenc.numel() // model.seqlen
    if max_samples is not None:
        nsamples = min(nsamples, max_samples)
    if nsamples <= 0:
        raise RuntimeError("WikiText-2 test encoding is shorter than one evaluation segment")

    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    with torch.no_grad():
        for idx in range(nsamples):
            if idx % 25 == 0:
                print(f"ppl sample {idx}/{nsamples}")
            inputs = testenc[:, idx * model.seqlen : (idx + 1) * model.seqlen].to(device)
            logits = model(inputs).logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    torch.cuda.empty_cache()
    return float(ppl.item())


def prune_wanda_aap(args, model, tokenizer, device, prune_n: int = 0, prune_m: int = 0) -> None:
    import torch

    from lib.layerwrapper import WrappedGPT

    use_cache = model.config.use_cache
    model.config.use_cache = False

    calibration_source = getattr(args, "calibration_source", "c4")
    print(f"loading {calibration_source} calibration data")
    dataloader = get_calibration_loader(
        args,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")

    with torch.no_grad():
        inps, outs, layer_kwargs = prepare_calibration_input(
            model,
            dataloader,
            device,
            nsamples=args.nsamples,
        )

    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        if hasattr(model, "hf_device_map") and f"model.layers.{layer_idx}" in model.hf_device_map:
            device = model.hf_device_map[f"model.layers.{layer_idx}"]
            inps = inps.to(device)
            outs = outs.to(device)
            layer_kwargs = move_to_device(layer_kwargs, device)

        subset = find_linear_layers(layer)
        wrapped_layers = {name: WrappedGPT(linear) for name, linear in subset.items()}

        def add_batch(name):
            def hook(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return hook

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]
        for sample_idx in range(args.nsamples):
            with torch.no_grad():
                outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)[0]
        for handle in handles:
            handle.remove()

        for name, linear in subset.items():
            print(f"pruning layer {layer_idx} name {name}")
            weight_metric = torch.abs(linear.weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            weight_mask = torch.zeros_like(weight_metric, dtype=torch.bool)
            if prune_n:
                for col in range(weight_metric.shape[1]):
                    if col % prune_m == 0:
                        block = weight_metric[:, col : col + prune_m].float()
                        weight_mask.scatter_(
                            1,
                            col + torch.topk(block, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sorted_metric = torch.sort(weight_metric, dim=-1, stable=True)
                if args.use_variant:
                    raise NotImplementedError("Wanda variant search is not enabled in the AAP adapter")
                indices = sorted_metric[1][:, : int(weight_metric.shape[1] * args.sparsity_ratio)]
                weight_mask.scatter_(1, indices, True)

            linear.weight.data[weight_mask] = 0

        for sample_idx in range(args.nsamples):
            with torch.no_grad():
                outs[sample_idx] = layer(inps[sample_idx].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def load_model_and_tokenizer(config: WandaRunConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype_from_name(config.dtype)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    if config.cache_dir:
        model_kwargs["cache_dir"] = config.cache_dir

    print(f"loading model {config.model}")
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    max_position_embeddings = int(getattr(model.config, "max_position_embeddings", config.seqlen))
    model.seqlen = min(max_position_embeddings, config.seqlen)
    model.eval()

    tokenizer_kwargs: dict[str, Any] = {"use_fast": False}
    if config.cache_dir:
        tokenizer_kwargs["cache_dir"] = config.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(config.model, **tokenizer_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_wanda(config: WandaRunConfig) -> dict[str, Any]:
    import numpy as np
    import torch

    add_wanda_to_path(config.wanda_dir)
    from lib.prune import prune_magnitude, prune_sparsegpt

    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)

    if config.sparsity_type != "unstructured":
        if config.sparsity_ratio != 0.5:
            raise ValueError("N:M sparsity requires sparsity_ratio=0.5")
        prune_n, prune_m = [int(part) for part in config.sparsity_type.split(":")]
    else:
        prune_n, prune_m = 0, 0

    model, tokenizer = load_model_and_tokenizer(config)
    device = torch.device("cuda:0")
    if hasattr(model, "hf_device_map") and "lm_head" in model.hf_device_map:
        device = model.hf_device_map["lm_head"]
    print(f"using device {device}")
    print(f"effective seqlen {model.seqlen}")

    args = type("Args", (), {})()
    args.seed = config.seed
    args.nsamples = config.nsamples
    args.sparsity_ratio = config.sparsity_ratio
    args.sparsity_type = config.sparsity_type
    args.prune_method = config.prune_method
    args.cache_dir = config.cache_dir
    args.use_variant = config.use_variant
    args.calibration_source = config.calibration_source
    args.calibration_data = str(config.calibration_data) if config.calibration_data else None

    started = time.time()
    if config.sparsity_ratio:
        print("pruning starts")
        if config.prune_method == "wanda":
            prune_wanda_aap(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif config.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif config.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        else:
            raise ValueError(f"unsupported prune_method: {config.prune_method}")

    actual_sparsity = check_sparsity(model)
    ppl = None
    if config.eval_ppl:
        ppl = float(
            eval_wikitext2_ppl_aap(
                model,
                tokenizer,
                device,
                cache_dir=config.cache_dir,
                max_samples=config.ppl_max_samples,
            )
        )

    if config.save_model:
        config.save_model.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(config.save_model)
        tokenizer.save_pretrained(config.save_model)

    result = {
        "model": config.model,
        "dtype": config.dtype,
        "seed": config.seed,
        "nsamples": config.nsamples,
        "seqlen": model.seqlen,
        "sparsity_ratio_target": config.sparsity_ratio,
        "sparsity_type": config.sparsity_type,
        "prune_method": config.prune_method,
        "actual_sparsity": actual_sparsity,
        "wikitext2_ppl": ppl,
        "ppl_max_samples": config.ppl_max_samples,
        "runtime_seconds": time.time() - started,
        "saved_model": str(config.save_model) if config.save_model else None,
        "wanda_dir": str(config.wanda_dir),
        "calibration_source": config.calibration_source,
        "calibration_data": str(config.calibration_data) if config.calibration_data else None,
    }
    config.out.parent.mkdir(parents=True, exist_ok=True)
    with config.out.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    return result
