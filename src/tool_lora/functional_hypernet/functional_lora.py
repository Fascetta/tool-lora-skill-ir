from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TargetLinearSpec:
    module_path: str
    layer_index: int
    module_name: str
    in_features: int
    out_features: int


@dataclass
class LoRAFactors:
    """Direct LoRA factors with A=[rank,in] and B=[out,rank]."""

    A: torch.Tensor
    B: torch.Tensor

    def validate(self, spec: TargetLinearSpec, rank: int) -> None:
        if tuple(self.A.shape) != (rank, spec.in_features):
            raise ValueError(
                f"A shape mismatch for {spec.module_path}: {tuple(self.A.shape)} "
                f"!= {(rank, spec.in_features)}"
            )
        if tuple(self.B.shape) != (spec.out_features, rank):
            raise ValueError(
                f"B shape mismatch for {spec.module_path}: {tuple(self.B.shape)} "
                f"!= {(spec.out_features, rank)}"
            )


GeneratedLoRA = dict[str, LoRAFactors]


def _layer_index(module_path: str) -> int | None:
    parts = module_path.split(".")
    for marker in ("layers", "h", "blocks"):
        if marker in parts:
            index = parts.index(marker) + 1
            if index < len(parts) and parts[index].isdigit():
                return int(parts[index])
    return None


def discover_target_linears(
    model: nn.Module,
    target_modules: list[str] | tuple[str, ...],
    target_layers: list[int] | tuple[int, ...] | None = None,
) -> list[TargetLinearSpec]:
    """Discover shape-correct Qwen linear targets without PEFT mutation."""
    module_order = {name: index for index, name in enumerate(target_modules)}
    allowed_layers = set(target_layers) if target_layers is not None else None
    specs: list[TargetLinearSpec] = []
    for path, module in model.named_modules():
        short_name = path.rsplit(".", 1)[-1]
        if short_name not in module_order:
            continue
        layer = _layer_index(path)
        if layer is None or (allowed_layers is not None and layer not in allowed_layers):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Target {path} is {type(module).__name__}, expected nn.Linear")
        specs.append(
            TargetLinearSpec(
                module_path=path,
                layer_index=layer,
                module_name=short_name,
                in_features=module.in_features,
                out_features=module.out_features,
            )
        )
    specs.sort(key=lambda spec: (spec.layer_index, module_order[spec.module_name]))
    if not specs:
        raise ValueError(f"No target linears found for modules={target_modules}, layers={target_layers}")
    found_modules = {spec.module_name for spec in specs}
    missing = set(target_modules) - found_modules
    if missing:
        raise ValueError(f"Missing requested target modules: {sorted(missing)}")
    return specs


def module_feature_sizes(specs: list[TargetLinearSpec]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    for spec in specs:
        shape = (spec.in_features, spec.out_features)
        previous = sizes.setdefault(spec.module_name, shape)
        if previous != shape:
            raise ValueError(
                f"Module {spec.module_name} changes shape across layers: {previous} vs {shape}"
            )
    return sizes


def zero_lora(
    specs: list[TargetLinearSpec], rank: int, *, device: torch.device | str, dtype: torch.dtype
) -> GeneratedLoRA:
    return {
        spec.module_path: LoRAFactors(
            A=torch.zeros(rank, spec.in_features, device=device, dtype=dtype),
            B=torch.zeros(spec.out_features, rank, device=device, dtype=dtype),
        )
        for spec in specs
    }


@contextmanager
def apply_functional_lora(
    model: nn.Module,
    factors: Mapping[str, LoRAFactors],
    *,
    scaling: float,
    dropout: float = 0.0,
    strict: bool = True,
) -> Iterator[None]:
    """Temporarily apply generated factors while retaining their autograd graph."""
    modules = dict(model.named_modules())
    originals: list[tuple[nn.Linear, Any]] = []
    missing: list[str] = []
    try:
        for path, pair in factors.items():
            module = modules.get(path)
            if not isinstance(module, nn.Linear):
                missing.append(path)
                continue
            if getattr(module, "_functional_lora_active", False):
                raise RuntimeError(f"Nested functional LoRA application is not allowed: {path}")
            if pair.A.ndim != 2 or pair.B.ndim != 2 or pair.B.shape[1] != pair.A.shape[0]:
                raise ValueError(f"Invalid LoRA factor shapes for {path}: A={pair.A.shape}, B={pair.B.shape}")
            if pair.A.shape[1] != module.in_features or pair.B.shape[0] != module.out_features:
                raise ValueError(
                    f"LoRA factors do not match {path}: module=({module.in_features},{module.out_features}) "
                    f"A={tuple(pair.A.shape)} B={tuple(pair.B.shape)}"
                )
            original_forward = module.forward

            def forward(
                x: torch.Tensor,
                *,
                _original: Any = original_forward,
                _pair: LoRAFactors = pair,
            ) -> torch.Tensor:
                base = _original(x)
                a = _pair.A.to(device=x.device, dtype=x.dtype)
                b = _pair.B.to(device=x.device, dtype=x.dtype)
                delta = F.linear(F.linear(F.dropout(x, p=dropout, training=model.training), a), b)
                return base + delta.to(base.dtype) * scaling

            originals.append((module, original_forward))
            module._functional_lora_active = True  # type: ignore[attr-defined]
            module.forward = forward  # type: ignore[method-assign]
        if strict and missing:
            raise KeyError(f"Generated adapter targets not found in model: {missing[:20]}")
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original  # type: ignore[method-assign]
            module._functional_lora_active = False  # type: ignore[attr-defined]


def adapter_diagnostics(
    factors: Mapping[str, LoRAFactors],
    model: nn.Module | None = None,
    *,
    scaling: float = 1.0,
) -> dict[str, Any]:
    modules = dict(model.named_modules()) if model is not None else {}
    per_module: dict[str, dict[str, float]] = {}
    all_values = []
    for path, pair in factors.items():
        # Diagnostics are deliberately outside the optimization objective.
        # Detach before scalar conversion to avoid retaining the training graph.
        a = pair.A.detach().float()
        b = pair.B.detach().float()
        delta = (b @ a) * scaling
        module = modules.get(path)
        base_norm = float(module.weight.detach().float().norm()) if isinstance(module, nn.Linear) else 0.0
        delta_norm = float(delta.norm())
        per_module[path] = {
            "a_norm": float(a.norm()),
            "b_norm": float(b.norm()),
            "delta_norm": delta_norm,
            "max_abs": float(torch.maximum(a.abs().max(), b.abs().max())),
            "update_to_base_ratio": delta_norm / max(base_norm, 1e-12),
            "finite": float(torch.isfinite(a).all() and torch.isfinite(b).all()),
        }
        all_values.extend((a.flatten(), b.flatten()))
    flat = torch.cat(all_values) if all_values else torch.zeros(1)
    return {
        "factor_norm": float(flat.norm()),
        "factor_max_abs": float(flat.abs().max()),
        "finite": bool(torch.isfinite(flat).all()),
        "per_module": per_module,
    }


def delta_regularization(factors: Mapping[str, LoRAFactors], scaling: float = 1.0) -> torch.Tensor:
    penalties = [(pair.B.float() @ pair.A.float()).pow(2).mean() for pair in factors.values()]
    if not penalties:
        raise ValueError("Cannot regularize an empty generated adapter")
    return torch.stack(penalties).mean() * (scaling**2)


def load_peft_lora(adapter_path: str | Path, model: nn.Module) -> tuple[GeneratedLoRA, float, dict[str, Any]]:
    """Load a standard PEFT LoRA as functional tensors for teacher/equality tests."""
    from safetensors.torch import load_file

    adapter_path = Path(adapter_path)
    config = json.loads((adapter_path / "adapter_config.json").read_text())
    state = load_file(adapter_path / "adapter_model.safetensors")
    modules = dict(model.named_modules())
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if ".lora_A." in key:
            prefix = key.split(".lora_A.", 1)[0]
            factor = "A"
        elif ".lora_B." in key:
            prefix = key.split(".lora_B.", 1)[0]
            factor = "B"
        else:
            continue
        candidates = [prefix]
        for removable in ("base_model.model.", "base_model.model.model."):
            if prefix.startswith(removable):
                candidates.append(prefix[len(removable) :])
                if removable.endswith("model."):
                    candidates.append("model." + prefix[len(removable) :])
        path = next((candidate for candidate in candidates if candidate in modules), None)
        if path is None:
            raise KeyError(f"Cannot map PEFT key to base module: {key}")
        pairs.setdefault(path, {})[factor] = tensor
    result = {
        path: LoRAFactors(A=pair["A"], B=pair["B"])
        for path, pair in pairs.items()
        if "A" in pair and "B" in pair
    }
    if not result:
        raise ValueError(f"No LoRA factors found in {adapter_path}")
    scaling = float(config.get("lora_alpha", config["r"])) / float(config["r"])
    return result, scaling, config
