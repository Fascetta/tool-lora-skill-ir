"""Deterministic lexical/token-sequence separator pointer."""
from __future__ import annotations

def pointer_index(tokenizer, observed: str, candidates: tuple[str, ...]) -> int:
    observed_ids = tuple(tokenizer(observed, add_special_tokens=False)["input_ids"])
    matches = [i for i, value in enumerate(candidates) if tuple(tokenizer(value, add_special_tokens=False)["input_ids"]) == observed_ids]
    if len(matches) != 1:
        raise AssertionError(f"pointer expected one token match, got {matches} for {observed!r}")
    return matches[0]
