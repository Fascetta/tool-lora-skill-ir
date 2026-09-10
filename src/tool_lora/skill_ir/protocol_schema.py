"""Named nine-component Stage-1 protocol schema."""
from __future__ import annotations

from dataclasses import dataclass

from tool_lora.skill_ir.protocol_ir import NINE_PROTOCOL_IR_COMPONENTS, ExplicitSkillIR


@dataclass(frozen=True)
class ProtocolValues:
    fn_1: str
    fn_2: str
    field_1: str
    field_2: str
    field_3: str
    enum_1: str
    enum_2: str
    date_order: str
    date_separator: str

    def as_tuple(self) -> tuple[str, ...]:
        return tuple(getattr(self, name) for name in NINE_PROTOCOL_IR_COMPONENTS)

    def to_skill_ir(self, confidence=None) -> ExplicitSkillIR:
        return ExplicitSkillIR.from_values(self.as_tuple(), confidence=confidence)

    @classmethod
    def from_tuple(cls, values: tuple[str, ...]) -> "ProtocolValues":
        if len(values) != 9:
            raise ValueError("expected nine protocol values")
        return cls(*values)
