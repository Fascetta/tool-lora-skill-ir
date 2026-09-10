"""Frozen V6-G compiler contract: explicit legacy IR -> LoRA fast weights."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from tool_lora.functional_hypernet.functional_lora import LoRAFactors
from tool_lora.skill_ir.protocol_ir import ExplicitSkillIR
from tool_lora.skill_ir.registry import ExecutionSpec
from tool_lora.skill_ir.types import IA3FastWeights, LoRAFastWeights


class OracleCompiler(nn.Module):
    """Architecture declared by the self-describing ``v6g_compiler_v2`` artifact."""
    def __init__(self, paths, shapes, rank: int, alpha: float, dim: int = 2048):
        super().__init__()
        self.paths, self.shapes, self.rank, self.alpha = list(paths), dict(shapes), rank, alpha
        total = sum(rank * (o + i) for o, i in self.shapes.values())
        self.net = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, total))

    def fast(self, ir: ExplicitSkillIR | torch.Tensor, spec: ExecutionSpec) -> LoRAFastWeights:
        vector = ir.legacy_ir_tensor if isinstance(ir, ExplicitSkillIR) else ir
        out = self.net(vector.reshape(1, -1)).reshape(-1)
        factors, offset = {}, 0
        for path in self.paths:
            out_features, in_features = self.shapes[path]
            n_a = self.rank * in_features
            a = out[offset:offset+n_a].reshape(self.rank, in_features); offset += n_a
            n_b = self.rank * out_features
            b = out[offset:offset+n_b].reshape(out_features, self.rank); offset += n_b
            factors[path] = LoRAFactors(a, b)
        return LoRAFastWeights(spec, factors, self.alpha)


class OracleIA3Compiler(nn.Module):
    """Explicit-IR compiler for standard IA3 residual activation scales."""
    def __init__(self, paths, shapes, dim: int = 2048):
        super().__init__()
        self.paths, self.shapes = list(paths), dict(shapes)
        self.sizes = {path: (in_features if path.endswith("down_proj") else out_features) for path, (out_features, in_features) in self.shapes.items()}
        total = sum(self.sizes.values())
        self.net = nn.Sequential(nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, total))

    def fast(self, ir: ExplicitSkillIR | torch.Tensor, spec: ExecutionSpec) -> IA3FastWeights:
        vector = ir.legacy_ir_tensor if isinstance(ir, ExplicitSkillIR) else ir
        out = self.net(vector.reshape(1, -1)).reshape(-1)
        scales, offset = {}, 0
        for path in self.paths:
            size = self.sizes[path]
            scales[path] = out[offset:offset + size]
            offset += size
        return IA3FastWeights(spec, scales)


def load_compiler(path: str | Path, device: torch.device) -> tuple[OracleCompiler, dict, ExecutionSpec]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    metadata = payload.get("meta", {})
    if metadata.get("format") != "v6g_compiler_v2":
        raise ValueError(f"unexpected compiler format: {metadata.get('format')}")
    shapes = {key: tuple(value) for key, value in metadata["shapes"].items()}
    mechanism = metadata["executor"].get("mechanism", "lora")
    if mechanism == "ia3":
        compiler = OracleIA3Compiler(metadata["paths"], shapes).to(device)
    else:
        compiler = OracleCompiler(metadata["paths"], shapes, int(metadata["executor"]["rank"]), float(metadata["executor"]["alpha"])).to(device)
    compiler.load_state_dict(payload["state"]["compiler"])
    compiler.eval()
    capacity = int(metadata["executor"].get("rank", 1))
    spec = ExecutionSpec(mechanism, capacity, metadata["executor"]["target_profile"])
    for parameter in compiler.parameters(): parameter.requires_grad_(False)
    return compiler, metadata, spec
