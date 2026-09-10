"""Minimal generic oracle contract for Stage 4-P0.

This module is deliberately independent of ``tool_lora.skill_ir``.  The IR
contains only a conditional relation, two generic alternatives, and exact
lexical bindings.  It has no evaluator identifiers or domain-specific factor
names.  Requests and the public schema are the only execution-time inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArgumentBinding:
    name: str
    source: str  # ``entity`` or ``literal``
    value: str | None = None


@dataclass(frozen=True)
class PolicyBranch:
    action: str
    arguments: tuple[ArgumentBinding, ...]


@dataclass(frozen=True)
class GenericPolicyIR:
    """Persistent semantic state, with generic relation and lexical sidecar."""

    condition_key: str
    condition_value: str
    alternatives: tuple[PolicyBranch, PolicyBranch]
    lexical_bindings: tuple[tuple[str, str], ...]
    format: str = "stage4_p0_generic_policy_ir_v1"

    def __post_init__(self) -> None:
        if len(self.alternatives) != 2:
            raise ValueError("exactly two policy alternatives are required")
        if not self.condition_key or not self.condition_value:
            raise ValueError("conditional relation must be non-empty")
        if not self.lexical_bindings:
            raise ValueError("at least one opaque lexical binding is required")
        for key, value in self.lexical_bindings:
            if not key or not value:
                raise ValueError("lexical bindings must preserve exact non-empty values")
        for branch in self.alternatives:
            for binding in branch.arguments:
                if binding.source not in {"entity", "literal"}:
                    raise ValueError(f"unsupported binding source: {binding.source}")

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "condition": {"key": self.condition_key, "value": self.condition_value},
            "alternatives": [
                {"action": b.action, "arguments": [
                    {"name": a.name, "source": a.source, **({"value": a.value} if a.value is not None else {})}
                    for a in b.arguments
                ]} for b in self.alternatives
            ],
            "lexical_bindings": [{"key": k, "value": v} for k, v in self.lexical_bindings],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n")

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "GenericPolicyIR":
        if payload.get("format") != "stage4_p0_generic_policy_ir_v1":
            raise ValueError("unexpected Stage-4-P0 IR format")
        condition = payload["condition"]
        alternatives = tuple(PolicyBranch(
            str(branch["action"]), tuple(ArgumentBinding(
                str(arg["name"]), str(arg["source"]), arg.get("value")
            ) for arg in branch["arguments"])
        ) for branch in payload["alternatives"])
        bindings = tuple((str(item["key"]), str(item["value"])) for item in payload["lexical_bindings"])
        return cls(str(condition["key"]), str(condition["value"]), alternatives, bindings)

    @classmethod
    def load(cls, path: str | Path) -> "GenericPolicyIR":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class MockRequest:
    entity: str
    attributes: tuple[tuple[str, str], ...]

    def value(self, key: str) -> str | None:
        return dict(self.attributes).get(key)


@dataclass(frozen=True)
class MockToolSchema:
    actions: tuple[tuple[str, tuple[str, ...]], ...]

    def arguments_for(self, action: str) -> tuple[str, ...]:
        for name, arguments in self.actions:
            if name == action:
                return arguments
        raise ValueError(f"action absent from public schema: {action}")


@dataclass(frozen=True)
class ToolDecision:
    action: str
    arguments: tuple[tuple[str, str], ...]


def interpret(ir: GenericPolicyIR, request: MockRequest, schema: MockToolSchema) -> ToolDecision:
    """Resolve generic persistent state into one abstract tool decision."""
    branch = ir.alternatives[0] if request.value(ir.condition_key) == ir.condition_value else ir.alternatives[1]
    allowed = schema.arguments_for(branch.action)
    literals = dict(ir.lexical_bindings)
    arguments: list[tuple[str, str]] = []
    for binding in branch.arguments:
        if binding.name not in allowed:
            raise ValueError(f"argument absent from public schema: {binding.name}")
        if binding.source == "entity":
            value = request.entity
        else:
            if binding.value is None or binding.value not in literals:
                raise ValueError(f"unknown lexical binding key: {binding.value}")
            value = literals[binding.value]
        arguments.append((binding.name, value))
    return ToolDecision(branch.action, tuple(arguments))
