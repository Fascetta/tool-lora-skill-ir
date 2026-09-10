"""Deterministic learner-visible acquisition into the frozen P0 contract.

The learner sees only semi-natural request text and observable mock traces. It
recovers the two-way conditional relation and exact literal by consensus; it
does not receive evaluator objects or convention identifiers.
"""
from __future__ import annotations

import json
import re

from .contract import ArgumentBinding, GenericPolicyIR, PolicyBranch

_DEMO_RE = re.compile(
    r"Example \d+: (?P<request>.*?)\nObserved action trace: (?P<trace>\{.*?\})\n",
    re.S,
)
_ENTITY_RE = re.compile(r"\bentity-[a-z0-9-]+\b")
_CONDITION_RE = re.compile(r"\b(urgent|routine)\b", re.I)


def _observations(support: str) -> list[tuple[str, str, str, str]]:
    observations = []
    for match in _DEMO_RE.finditer(support):
        request, trace = match.group("request"), json.loads(match.group("trace"))
        condition = _CONDITION_RE.search(request)
        entity = _ENTITY_RE.search(request)
        if condition is None or entity is None:
            raise ValueError("demonstration lacks an observable condition or entity")
        action = str(trace["action"])
        arguments = {str(k): str(v) for k, v in trace["arguments"].items()}
        entity_matches = [key for key, value in arguments.items() if value == entity.group(0)]
        if len(entity_matches) != 1:
            raise ValueError("trace must contain exactly one entity argument")
        literal_items = [(key, value) for key, value in arguments.items() if key != entity_matches[0]]
        if len(literal_items) != 1:
            raise ValueError("trace must contain exactly one opaque literal argument")
        literal_key, literal = literal_items[0]
        observations.append((condition.group(1).lower(), action, literal_key, literal))
    if not observations:
        raise ValueError("support contains no parseable demonstrations")
    return observations


def acquire(support: str) -> GenericPolicyIR:
    """Acquire a frozen-contract IR from demonstrations only."""
    observations = _observations(support)
    values = sorted({item[0] for item in observations}, reverse=True)
    if len(values) != 2:
        raise ValueError("support must identify exactly two condition values")
    literals = {item[3] for item in observations}
    if len(literals) != 1:
        raise ValueError("opaque binding is inconsistent across demonstrations")
    literal_key = {item[2] for item in observations}
    if len(literal_key) != 1:
        raise ValueError("opaque argument key is inconsistent across demonstrations")
    branches = []
    for value in values:
        actions = {item[1] for item in observations if item[0] == value}
        if len(actions) != 1:
            raise ValueError("conditional action relation is ambiguous")
        action = next(iter(actions))
        branches.append(PolicyBranch(action, (
            ArgumentBinding("entity", "entity"),
            ArgumentBinding("scope", "literal", "scope"),
        )))
    return GenericPolicyIR("priority", values[0], tuple(branches), (("scope", next(iter(literals))),))


def parse_observations(support: str) -> list[tuple[str, str, str, str]]:
    """Expose fit-free observations for the evaluator-side identifiability audit."""
    return _observations(support)
