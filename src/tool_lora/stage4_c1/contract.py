from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tool_lora.stage4_p0.contract import GenericPolicyIR


@dataclass(frozen=True)
class EffectAnchoredIR:
    base: GenericPolicyIR
    role_effects: tuple[tuple[str, str], ...]
    format: str = "stage4_c1_effect_anchored_ir_v1"

    def __post_init__(self) -> None:
        effects = dict(self.role_effects)
        roles = {branch.action for branch in self.base.alternatives}
        if roles != set(effects):
            raise ValueError("every abstract dispatch role needs one effect anchor")
        if not set(effects.values()) <= {"state_preserving", "state_mutating"}:
            raise ValueError("role effects must use the generic binary effect vocabulary")

    def to_dict(self) -> dict[str, Any]:
        return {"format": self.format, "base": self.base.to_dict(),
                "role_effects": [{"role": role, "effect": effect} for role, effect in self.role_effects]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EffectAnchoredIR":
        return cls(GenericPolicyIR.from_dict(payload["base"]),
                   tuple((str(item["role"]), str(item["effect"])) for item in payload["role_effects"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "EffectAnchoredIR":
        return cls.from_dict(json.loads(Path(path).read_text()))
