from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tool_lora.functional_hypernet.functional_lora import LoRAFactors
from tool_lora.skill_ir.registry import ExecutionSpec


@dataclass
class LoRAFastWeights:
    spec: ExecutionSpec
    factors: dict[str, LoRAFactors]
    alpha: float

    @property
    def scaling(self) -> float:
        return self.alpha / self.spec.capacity

    def generated_parameter_count(self) -> int:
        return sum(pair.A.numel() + pair.B.numel() for pair in self.factors.values())


@dataclass
class IA3FastWeights:
    spec: ExecutionSpec
    # Residual scales: executor applies activation * (1 + scale).
    scales: dict[str, torch.Tensor]

    def generated_parameter_count(self) -> int:
        return sum(value.numel() for value in self.scales.values())


FastWeights = LoRAFastWeights | IA3FastWeights


@dataclass
class SkillMemoryItem:
    """Serializable persistent memory. Compiled fast weights are never stored."""

    z: torch.Tensor
    experience_id: str | None = None
    metadata: dict[str, Any] | None = None

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "format": "skill_ir_v1",
                "z": self.z.detach().cpu(),
                "experience_id": self.experience_id,
                "metadata": self.metadata or {},
            },
            Path(path),
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str = "cpu") -> "SkillMemoryItem":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if payload.get("format") != "skill_ir_v1":
            raise ValueError("Not a Skill IR memory item")
        return cls(payload["z"], payload.get("experience_id"), payload.get("metadata"))
