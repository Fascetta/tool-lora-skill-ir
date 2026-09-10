from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


Mechanism = Literal["lora", "ia3", "prefix", "adapter", "reft"]

LORA_TARGET_PROFILES: dict[str, tuple[str, ...]] = {
    "T1": ("q_proj", "v_proj"),
    "T2": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "T3": ("gate_proj", "up_proj", "down_proj"),
    "T4": ("q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"),
    "T5": (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ),
}
IA3_TARGET_PROFILES: dict[str, tuple[str, ...]] = {
    # Standard IA3: scale K/V activations and the intermediate FFN activation.
    "IA3_STANDARD": ("k_proj", "v_proj", "down_proj"),
}
LAYER_PROFILES = ("all", "lower_half", "upper_half")


@dataclass(frozen=True)
class ExecutionSpec:
    mechanism: Mechanism
    capacity: int
    target_profile: str
    layer_profile: str = "all"
    budget: float | None = None
    split: Literal["train", "validation", "test"] = "train"

    @property
    def identifier(self) -> str:
        return (
            f"{self.mechanism}_{self.target_profile.lower()}_c{self.capacity}_"
            f"{self.layer_profile}"
        )

    @property
    def target_modules(self) -> tuple[str, ...]:
        profiles = LORA_TARGET_PROFILES if self.mechanism == "lora" else IA3_TARGET_PROFILES
        if self.target_profile not in profiles:
            raise ValueError(
                f"Unknown {self.mechanism} target profile: {self.target_profile}"
            )
        return profiles[self.target_profile]

    def layer_indices(self, total_layers: int) -> tuple[int, ...]:
        if self.layer_profile == "all":
            return tuple(range(total_layers))
        midpoint = total_layers // 2
        if self.layer_profile == "lower_half":
            return tuple(range(midpoint))
        if self.layer_profile == "upper_half":
            return tuple(range(midpoint, total_layers))
        raise ValueError(f"Unknown layer profile: {self.layer_profile}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_execution_registry(
    *,
    lora_ranks: tuple[int, ...] = (2, 4, 8, 16, 32, 64),
    lora_profiles: tuple[str, ...] = tuple(LORA_TARGET_PROFILES),
    include_ia3: bool = True,
) -> tuple[ExecutionSpec, ...]:
    """Finite registry; callers may filter or add arbitrary-rank LoRA specs."""
    result = [
        ExecutionSpec("lora", rank, profile)
        for rank in lora_ranks
        for profile in lora_profiles
    ]
    if include_ia3:
        result.append(ExecutionSpec("ia3", 1, "IA3_STANDARD"))
    return tuple(result)


def training_execution_registry(
    *, holdout_identifier: str = "lora_t2_c16_all"
) -> tuple[ExecutionSpec, ...]:
    """Controlled v1 family: ranks 4/16/32 x T1/T2/T5 plus IA3.

    One rank/profile combination is held out while both its rank and profile
    remain observed in other combinations.
    """
    specs = canonical_execution_registry(
        lora_ranks=(4, 16, 32), lora_profiles=("T1", "T2", "T5"), include_ia3=True
    )
    return tuple(
        ExecutionSpec(
            spec.mechanism,
            spec.capacity,
            spec.target_profile,
            spec.layer_profile,
            spec.budget,
            "validation" if spec.identifier == holdout_identifier else "train",
        )
        for spec in specs
    )
