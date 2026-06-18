from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PreferenceRecord:
    id: str
    prompt: str
    chosen: str
    rejected: str
    source: str


def split_hh_transcript(text: str) -> tuple[str, str]:
    markers = ["\n\nAssistant:", "\nAssistant:", "Assistant:"]
    for marker in markers:
        idx = text.rfind(marker)
        if idx >= 0:
            prompt = text[: idx + len(marker)]
            response = text[idx + len(marker) :]
            if prompt.strip() and response.strip():
                return prompt, response
    raise ValueError("could not split HH-RLHF transcript on final Assistant marker")


def normalize_hh_pair(row: dict, idx: int) -> PreferenceRecord:
    chosen_prompt, chosen_response = split_hh_transcript(str(row["chosen"]))
    rejected_prompt, rejected_response = split_hh_transcript(str(row["rejected"]))
    if chosen_prompt.strip() != rejected_prompt.strip():
        # Keep the chosen prompt as canonical; this is rare and usually reflects
        # whitespace differences in the source transcript.
        rejected_response = str(row["rejected"])[len(chosen_prompt) :] if str(row["rejected"]).startswith(chosen_prompt) else rejected_response
    return PreferenceRecord(
        id=str(row.get("id", f"hh_rlhf_{idx}")),
        prompt=chosen_prompt,
        chosen=chosen_response,
        rejected=rejected_response,
        source="Anthropic/hh-rlhf",
    )


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def read_preference_jsonl(path: str | Path, max_samples: int | None = None) -> list[PreferenceRecord]:
    records: list[PreferenceRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(records) >= max_samples:
                break
            payload = json.loads(line)
            records.append(
                PreferenceRecord(
                    id=str(payload["id"]),
                    prompt=str(payload["prompt"]),
                    chosen=str(payload["chosen"]),
                    rejected=str(payload["rejected"]),
                    source=str(payload.get("source", "unknown")),
                )
            )
    return records

