#!/usr/bin/env python
"""Measure whether independently trained LoRA adapters are identifiable.

The audit compares raw A/B factors, effective updates B@A, deterministic
SVD-canonicalized factors, and (optionally) generated tool behavior across
multiple seeds for the same tool.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def adapter_paths(root: Path) -> list[tuple[int, str, Path]]:
    found = []
    for path in sorted(root.glob("seed_*/qwen3_0_6b/*/adapter_model.safetensors")):
        seed = int(path.parents[2].name.removeprefix("seed_"))
        tool = path.parents[0].name
        found.append((seed, tool, path))
    return found


def load_adapter(path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    tensors = load_file(str(path), device="cpu")
    a, b, delta, canonical = {}, {}, {}, {}
    for key, value in tensors.items():
        if ".lora_A.weight" in key:
            base = key.removesuffix(".lora_A.weight")
            a[base] = value.float()
        elif ".lora_B.weight" in key:
            base = key.removesuffix(".lora_B.weight")
            b[base] = value.float()
    config = read_json(path.with_name("adapter_config.json"))
    scale = float(config.get("lora_alpha", 1)) / float(config.get("r", 1))
    if set(a) != set(b):
        raise ValueError(f"A/B module mismatch in {path}")
    for base in sorted(a):
        d = b[base] @ a[base] * scale
        delta[base] = d
        # Exact compact SVD of B@A without decomposing the large matrix:
        # B=Qb Rb and A.T=Qa Ra, so B@A=Qb (Rb Ra.T) Qa.T.
        bd, ad = b[base].to(device), a[base].to(device)
        qb, rb = torch.linalg.qr(bd, mode="reduced")
        qa, ra = torch.linalg.qr(ad.transpose(0, 1), mode="reduced")
        uc, s, vhc = torch.linalg.svd(rb @ ra.transpose(0, 1), full_matrices=False)
        u, vh = qb @ uc, vhc @ qa.transpose(0, 1)
        root_s = s.clamp_min(0).sqrt()
        # Store SVD factors in the same shapes as B/A.  The sign convention
        # removes the usual per-singular-vector sign ambiguity.
        signs = torch.where(u[0] < 0, -torch.ones_like(root_s), torch.ones_like(root_s))
        u = u * signs
        vh = vh * signs[:, None]
        canonical[base + ".B"] = (u * root_s[None, :]).cpu()
        canonical[base + ".A"] = (root_s[:, None] * vh).cpu()
    return a, b, delta, {"canonical": canonical, "config": config}


def flatten(mapping: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([mapping[k].reshape(-1) for k in sorted(mapping)])


def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x))


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(x[None], y[None]).item())


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    raw_left = flatten({f"A/{k}": v for k, v in left["a"].items()} | {f"B/{k}": v for k, v in left["b"].items()})
    raw_right = flatten({f"A/{k}": v for k, v in right["a"].items()} | {f"B/{k}": v for k, v in right["b"].items()})
    delta_left = flatten(left["delta"])
    delta_right = flatten(right["delta"])
    can_left = flatten(left["canonical"])
    can_right = flatten(right["canonical"])
    delta_diff = delta_left - delta_right
    return {
        "raw_ab_l2": norm(raw_left - raw_right),
        "raw_ab_relative_l2": norm(raw_left - raw_right) / max(norm(raw_left), norm(raw_right), 1e-12),
        "raw_ab_cosine": cosine(raw_left, raw_right),
        "delta_l2": norm(delta_diff),
        "delta_relative_l2": norm(delta_diff) / max(norm(delta_left), norm(delta_right), 1e-12),
        "delta_cosine": cosine(delta_left, delta_right),
        "canonical_l2": norm(can_left - can_right),
        "canonical_relative_l2": norm(can_left - can_right) / max(norm(can_left), norm(can_right), 1e-12),
        "canonical_cosine": cosine(can_left, can_right),
        "delta_norm_left": norm(delta_left),
        "delta_norm_right": norm(delta_right),
    }


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows]
        out[key] = {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    paths = adapter_paths(args.root)
    if not paths:
        raise SystemExit(f"No adapters found below {args.root}")
    device = torch.device(args.device)
    records = []
    for seed, tool, path in paths:
        a, b, delta, extra = load_adapter(path, device)
        records.append({"seed": seed, "tool": tool, "path": str(path), "a": a, "b": b, "delta": delta, "canonical": extra["canonical"], "config": extra["config"]})
    pair_groups = {}
    for tool in sorted({r["tool"] for r in records}):
        group = [r for r in records if r["tool"] == tool]
        rows = []
        for left, right in combinations(sorted(group, key=lambda r: r["seed"]), 2):
            rows.append({"seed_left": left["seed"], "seed_right": right["seed"], **pair_metrics(left, right)})
        pair_groups[tool] = {
            "num_adapters": len(group),
            "seeds": [r["seed"] for r in sorted(group, key=lambda r: r["seed"])],
            "pairs": rows,
            "summary": summarize(rows, ["raw_ab_relative_l2", "raw_ab_cosine", "delta_relative_l2", "delta_cosine", "canonical_relative_l2", "canonical_cosine"]),
        }
    result = {
        "experiment": "multi_seed_lora_identifiability",
        "root": str(args.root),
        "num_adapters": len(records),
        "tools": sorted(pair_groups),
        "adapter_config": records[0]["config"],
        "groups": pair_groups,
        "interpretation_guidance": {
            "behaviorally_equivalent_but_parameter_different": "low delta_cosine or high delta_relative_l2 despite later behavioral agreement",
            "factorization_only_nonidentifiability": "raw_ab differs while delta and canonical metrics remain close",
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"num_adapters": len(records), "tools": sorted(pair_groups), "out": str(args.out_json)}, indent=2))


if __name__ == "__main__":
    main()
