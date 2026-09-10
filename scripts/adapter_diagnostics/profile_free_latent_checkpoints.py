#!/usr/bin/env python
"""Profile generated-update norms and spectra over free-latent checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from _bootstrap import add_src_to_path

add_src_to_path()

from tool_lora.functional_hypernet.functional_lora import TargetLinearSpec
from tool_lora.functional_hypernet.model import FreeLatentToolLoRAMetanetwork, FunctionalHyperNetConfig


def specs() -> list[TargetLinearSpec]:
    result = []
    for layer in range(28):
        for name, out in (("q_proj", 2048), ("v_proj", 1024)):
            result.append(TargetLinearSpec(f"model.layers.{layer}.self_attn.{name}", layer, name, 1024, out))
    return result


def compact_singular_values(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # For delta=B@A, compute the exact nonzero spectrum from the rank-sized
    # core after QR reduction of B and A.T.
    q_b, r_b = torch.linalg.qr(b, mode="reduced")
    q_a, r_a = torch.linalg.qr(a.transpose(0, 1), mode="reduced")
    del q_b, q_a
    return torch.linalg.svdvals(r_b @ r_a.transpose(0, 1))


def factor_profile(factors, scaling: float, device: torch.device | str = "cpu") -> dict[str, float | list[float]]:
    a, b = factors.A.float().to(device), factors.B.float().to(device)
    delta = b @ a * scaling
    singular = compact_singular_values(a, b)
    threshold = float(singular.max()) * 1e-4 if len(singular) else 0.0
    energy = singular.float().pow(2)
    raw_total_energy = energy.sum()
    if float(raw_total_energy) <= 1e-12:
        total_energy = energy.new_tensor(1.0)
        top_fraction = energy.new_tensor(0.0)
        entropy_rank = energy.new_tensor(0.0)
        participation_rank = energy.new_tensor(0.0)
    else:
        total_energy = raw_total_energy
        probabilities = energy / total_energy
        top_fraction = energy[0] / total_energy
        entropy_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
        participation_rank = total_energy.pow(2) / energy.pow(2).sum()
    return {
        "a_frobenius": float(a.norm()),
        "b_frobenius": float(b.norm()),
        "delta_frobenius": float(delta.norm()),
        "delta_spectral": float(singular.max()),
        "effective_rank_1e-4": int((singular > threshold).sum()),
        "top_energy_fraction": float(top_fraction),
        "entropy_effective_rank": float(entropy_rank),
        "participation_rank": float(participation_rank),
        "singular_values": [float(x) for x in singular],
    }


def direct_profiles(path: Path, device: torch.device | str) -> dict[str, dict[str, float | list[float]]]:
    state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
    pairs = {}
    for key, value in state.items():
        if ".lora_A.weight" in key:
            base = key.split(".lora_A.weight")[0]
            pairs.setdefault(base, {})["a"] = value
        elif ".lora_B.weight" in key:
            base = key.split(".lora_B.weight")[0]
            pairs.setdefault(base, {})["b"] = value
    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = float(cfg.get("lora_alpha", cfg["r"])) / float(cfg["r"])
    result = {}
    for module_path in [f"model.layers.{layer}.self_attn.{name}" for layer in range(28) for name in ("q_proj", "v_proj")]:
        candidates = [f"base_model.model.{module_path}", f"base_model.model.model.{module_path}", module_path]
        pair = next((pairs[c] for c in candidates if c in pairs), None)
        if pair is None:
            raise KeyError(f"No direct adapter tensor for {module_path} in {path}")
        class Pair: pass
        item = Pair(); item.A = pair["a"]; item.B = pair["b"]
        result[module_path] = factor_profile(item, scale, device)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-root", type=Path, required=True)
    p.add_argument("--oracle-root", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    target_specs = specs()
    checkpoints = sorted(args.checkpoint_root.glob("checkpoint-*.pt"), key=lambda x: int(x.stem.split("-")[-1]))
    if not checkpoints:
        raise SystemExit("No checkpoints found")
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    metadata = first["metadata"]
    config = FunctionalHyperNetConfig.from_dict(metadata["config"])
    tool_ids = list(metadata["tool_ids"])
    oracle_profiles = {tool: direct_profiles(args.oracle_root / tool, device) for tool in tool_ids}
    output = {"experiment": "stage4_p2_a_free_latent_geometry_profile", "tool_ids": tool_ids, "checkpoints": {}}
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        net = FreeLatentToolLoRAMetanetwork(config, target_specs, len(tool_ids), int(metadata["latent_size"])).to(device)
        net.load_state_dict(payload["model"], strict=True)
        net.eval()
        step = int(checkpoint.stem.split("-")[-1])
        step_result = {"tools": {}, "direct_oracle": {}}
        with torch.no_grad():
            for index, tool in enumerate(tool_ids):
                generated = net(torch.tensor([index], device=device))[0]
                module_profiles = {path: factor_profile(pair, net.scaling, device) for path, pair in generated.items()}
                step_result["tools"][tool] = {
                    "modules": module_profiles,
                    "aggregate": {
                        "mean_delta_frobenius": sum(x["delta_frobenius"] for x in module_profiles.values()) / len(module_profiles),
                        "mean_delta_spectral": sum(x["delta_spectral"] for x in module_profiles.values()) / len(module_profiles),
                        "mean_a_frobenius": sum(x["a_frobenius"] for x in module_profiles.values()) / len(module_profiles),
                        "mean_b_frobenius": sum(x["b_frobenius"] for x in module_profiles.values()) / len(module_profiles),
                        "mean_effective_rank": sum(x["effective_rank_1e-4"] for x in module_profiles.values()) / len(module_profiles),
                    },
                }
                oracle_path = args.oracle_root / tool
                step_result["direct_oracle"][tool] = {"modules": oracle_profiles[tool]}
        output["checkpoints"][str(step)] = step_result
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"checkpoints": len(checkpoints), "tools": tool_ids, "out": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
