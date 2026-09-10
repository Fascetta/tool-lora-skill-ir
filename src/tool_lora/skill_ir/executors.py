from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn

from tool_lora.functional_hypernet.functional_lora import apply_functional_lora
from tool_lora.skill_ir.types import FastWeights, IA3FastWeights, LoRAFastWeights


@contextmanager
def apply_functional_ia3(
    model: nn.Module, fast_weights: IA3FastWeights, *, strict: bool = True
) -> Iterator[None]:
    """Apply standard IA3 activation scales without mutating persistent weights.

    K/V projection outputs are scaled along their output feature dimension.
    The FFN intermediate activation is scaled before ``down_proj``. Compiler
    outputs are residual logit-like values and execute as ``1 + delta``.
    """
    modules = dict(model.named_modules())
    originals: list[tuple[nn.Linear, Any]] = []
    missing: list[str] = []
    try:
        for path, residual_scale in fast_weights.scales.items():
            module = modules.get(path)
            if not isinstance(module, nn.Linear):
                missing.append(path)
                continue
            if getattr(module, "_functional_ia3_active", False):
                raise RuntimeError(f"Nested functional IA3 application is not allowed: {path}")
            original_forward = module.forward
            if path.endswith("down_proj"):
                if residual_scale.numel() != module.in_features:
                    raise ValueError(f"IA3 input scale shape mismatch for {path}")

                def forward(
                    x: torch.Tensor,
                    *,
                    _original: Any = original_forward,
                    _scale: torch.Tensor = residual_scale,
                ) -> torch.Tensor:
                    scale = (1.0 + _scale).to(device=x.device, dtype=x.dtype)
                    return _original(x * scale)
            else:
                if residual_scale.numel() != module.out_features:
                    raise ValueError(f"IA3 output scale shape mismatch for {path}")

                def forward(
                    x: torch.Tensor,
                    *,
                    _original: Any = original_forward,
                    _scale: torch.Tensor = residual_scale,
                ) -> torch.Tensor:
                    output = _original(x)
                    scale = (1.0 + _scale).to(device=output.device, dtype=output.dtype)
                    return output * scale
            originals.append((module, original_forward))
            module._functional_ia3_active = True  # type: ignore[attr-defined]
            module.forward = forward  # type: ignore[method-assign]
        if strict and missing:
            raise KeyError(f"Generated IA3 targets not found: {missing[:20]}")
        yield
    finally:
        for module, original in reversed(originals):
            module.forward = original  # type: ignore[method-assign]
            module._functional_ia3_active = False  # type: ignore[attr-defined]


@contextmanager
def apply_fast_weights(model: nn.Module, fast_weights: FastWeights) -> Iterator[None]:
    if isinstance(fast_weights, LoRAFastWeights):
        with apply_functional_lora(
            model, fast_weights.factors, scaling=fast_weights.scaling
        ):
            yield
    elif isinstance(fast_weights, IA3FastWeights):
        with apply_functional_ia3(model, fast_weights):
            yield
    else:
        raise TypeError(f"Unsupported fast-weight type: {type(fast_weights).__name__}")
