from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseLogprob:
    sum_logprob: float
    num_response_tokens: int
    length_normalized_logprob: float


def truncate_for_response_scoring(
    prompt_ids: list[int],
    response_ids: list[int],
    max_length: int,
) -> tuple[list[int], int]:
    full_ids = prompt_ids + response_ids
    prompt_len = len(prompt_ids)
    if len(full_ids) <= max_length:
        return full_ids, prompt_len

    overflow = len(full_ids) - max_length
    if overflow < prompt_len:
        return full_ids[overflow:], prompt_len - overflow

    response_keep = response_ids[-max_length:]
    return response_keep, 0


def score_response_logprob(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device,
    max_length: int = 4096,
) -> ResponseLogprob:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    if not response_ids:
        raise ValueError("empty response after tokenization")

    input_ids, prompt_len = truncate_for_response_scoring(prompt_ids, response_ids, max_length)
    if len(input_ids) < 2:
        raise ValueError("sequence too short for log-probability scoring")

    start = max(prompt_len, 1)
    if start >= len(input_ids):
        raise ValueError("no response tokens remain after truncation")

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tensor).logits[0, :-1, :]
        labels = input_tensor[0, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    response_log_probs = token_log_probs[start - 1 :]
    sum_logprob = float(response_log_probs.sum().item())
    num_tokens = int(response_log_probs.numel())
    return ResponseLogprob(
        sum_logprob=sum_logprob,
        num_response_tokens=num_tokens,
        length_normalized_logprob=sum_logprob / num_tokens,
    )

