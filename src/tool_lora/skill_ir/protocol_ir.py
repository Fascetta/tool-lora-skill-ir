"""The exact legacy ProtocolIR algebra and persistent explicit Skill IR.

The tensor shape ``[8, 256]`` is only a view of one global 2048-coordinate
additive vector.  The nine semantic contributions remain the public schema.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import torch



@dataclass(frozen=True)
class ProtocolIR:
    function_binding: tuple[str, ...]
    field_binding: tuple[str, ...]
    enum_binding: tuple[str, ...]
    transform_binding: tuple[str, str]

    @classmethod
    def from_protocol(cls, protocol: Any) -> "ProtocolIR":
        return cls(
            tuple(protocol.operation_names[key] for key in ("weather", "forecast")),
            tuple(protocol.field_names[key] for key in ("city", "date", "units")),
            tuple(protocol.enum_names[key] for key in ("celsius", "fahrenheit")),
            (protocol.date_order, protocol.date_separator),
        )

    def canonical(self) -> tuple[str, ...]:
        return self.function_binding + self.field_binding + self.enum_binding + self.transform_binding

    def vector(self, *, dimension: int = 2048) -> torch.Tensor:
        if dimension % 8:
            raise ValueError("legacy ProtocolIR dimension must be divisible by 8")
        values = torch.zeros(dimension)
        for index, token in enumerate(self.canonical()):
            digest = hashlib.sha256(f"{index}:{token}".encode()).digest()
            position = int.from_bytes(digest[:4], "little") % dimension
            values[position] += 1.0
        return values.view(8, dimension // 8)


NINE_PROTOCOL_IR_COMPONENTS = (
    "fn_1", "fn_2", "field_1", "field_2", "field_3",
    "enum_1", "enum_2", "date_order", "date_separator",
)


def old_ir_component(index: int, token: str, *, dimension: int = 2048) -> torch.Tensor:
    if not 0 <= index < 9:
        raise IndexError(index)
    values = torch.zeros(dimension)
    digest = hashlib.sha256(f"{index}:{token}".encode()).digest()
    values[int.from_bytes(digest[:4], "little") % dimension] += 1.0
    return values.view(8, dimension // 8)


@dataclass(frozen=True)
class ExplicitSkillIR:
    """Persistent Stage-1 artifact; acquisition state is deliberately absent."""

    protocol_values: tuple[str, ...]
    legacy_ir_tensor: torch.Tensor
    schema_version: str = "stage1_explicit_ir_v1"
    confidence: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if len(self.protocol_values) != 9:
            raise ValueError("Stage-1 Skill IR requires exactly nine semantic values")
        if tuple(self.legacy_ir_tensor.shape) != (8, 256):
            raise ValueError("legacy IR tensor must have shape [8, 256]")

    @classmethod
    def from_values(cls, values: tuple[str, ...], confidence: Mapping[str, Any] | None = None) -> "ExplicitSkillIR":
        tensor = sum((old_ir_component(i, value) for i, value in enumerate(values)), torch.zeros(8, 256))
        return cls(tuple(values), tensor, confidence=confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "components": dict(zip(NINE_PROTOCOL_IR_COMPONENTS, self.protocol_values)),
            "legacy_ir_tensor": self.legacy_ir_tensor.detach().cpu().tolist(),
            "confidence": dict(self.confidence or {}),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ExplicitSkillIR":
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != "stage1_explicit_ir_v1":
            raise ValueError("unsupported explicit Skill IR schema")
        components = payload.get("components", {})
        values = tuple(components[name] for name in NINE_PROTOCOL_IR_COMPONENTS)
        tensor = torch.tensor(payload["legacy_ir_tensor"], dtype=torch.float32)
        result = cls(values, tensor, payload["schema_version"], payload.get("confidence"))
        expected = cls.from_values(values).legacy_ir_tensor
        if not torch.equal(result.legacy_ir_tensor, expected):
            raise ValueError("serialized legacy IR does not match its semantic values")
        return result
