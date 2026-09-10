"""Stage-2A compression of the legacy 2,048-dimensional Skill IR."""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import torch
from torch import nn

from tool_lora.skill_ir.protocol_ir import NINE_PROTOCOL_IR_COMPONENTS, ProtocolIR


IR_DIMENSION = 2048


def hash_coordinate(slot: int, token: str, dimension: int = IR_DIMENSION) -> int:
    digest = hashlib.sha256(f"{slot}:{token}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % dimension


def protocol_vector(values: Iterable[str]) -> torch.Tensor:
    values = tuple(values)
    if len(values) != len(NINE_PROTOCOL_IR_COMPONENTS):
        raise ValueError("Stage-2A examples require exactly nine protocol values")
    return ProtocolIR(values[:2], values[2:5], values[5:7], (values[7], values[8])).vector().reshape(-1)


def reachable_mask(pools: Mapping[str, Iterable[str]], dimension: int = IR_DIMENSION) -> torch.Tensor:
    """Compute R_train from complete identifier pools, never sampled examples."""
    families = ("functions", "arguments", "enums", "separators")
    slots = ((0, 1), (2, 3, 4), (5, 6), (8,))
    mask = torch.zeros(dimension, dtype=torch.bool)
    for family, family_slots in zip(families, slots):
        for slot in family_slots:
            for token in pools[family]:
                mask[hash_coordinate(slot, str(token), dimension)] = True
    # Date order is a fixed semantic family, not an opaque identifier pool.
    for token in ("DMY", "MDY", "YMD"):
        mask[hash_coordinate(7, token, dimension)] = True
    return mask


def occupancy(values: Iterable[str], dimension: int = IR_DIMENSION) -> dict:
    values = tuple(values)
    coordinates = [hash_coordinate(i, token, dimension) for i, token in enumerate(values)]
    counts = torch.bincount(torch.tensor(coordinates), minlength=dimension)
    collisions = [(i, int(count)) for i, count in enumerate(counts.tolist()) if count > 1]
    collision_pairs = []
    for coordinate in sorted({coordinate for coordinate, _ in collisions}):
        members = [NINE_PROTOCOL_IR_COMPONENTS[i] for i, value in enumerate(coordinates) if value == coordinate]
        collision_pairs.extend((members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members)))
    return {
        "component_coordinates": {
            name: coordinate for name, coordinate in zip(NINE_PROTOCOL_IR_COMPONENTS, coordinates)
        },
        "unique_active_coordinates": int((counts > 0).sum()),
        "collision_count": int(sum(count - 1 for count in counts.tolist() if count > 1)),
        "collision_component_pairs": int(sum(count * (count - 1) // 2 for count in counts.tolist() if count > 1)),
        "collision_component_pair_names": collision_pairs,
        "max_coordinate_count": int(counts.max()),
        "count_gt_one_coordinates": len(collisions),
    }


def dataset_occupancy(pools: Mapping[str, Iterable[str]], dimension: int = IR_DIMENSION) -> dict:
    rows = []
    all_coordinates: dict[int, list[str]] = {}
    families = ("functions", "arguments", "enums", "separators")
    slots = ((0, 1), (2, 3, 4), (5, 6), (8,))
    for family, family_slots in zip(families, slots):
        for slot in family_slots:
            coordinates = [hash_coordinate(slot, str(token), dimension) for token in pools[family]]
            for token, coordinate in zip(pools[family], coordinates):
                all_coordinates.setdefault(coordinate, []).append(f"{NINE_PROTOCOL_IR_COMPONENTS[slot]}:{token}")
            unique = len(set(coordinates))
            rows.append({"component_slot": slot, "family": family, "pool_size": len(coordinates),
                         "unique_coordinates": unique, "collision_count": len(coordinates) - unique})
    date_coordinates = [hash_coordinate(7, x, dimension) for x in ("DMY", "MDY", "YMD")]
    for token, coordinate in zip(("DMY", "MDY", "YMD"), date_coordinates):
        all_coordinates.setdefault(coordinate, []).append(f"date_order:{token}")
    rows.append({"component_slot": 7, "family": "date_order", "pool_size": 3,
                 "unique_coordinates": len(set(date_coordinates)),
                 "collision_count": 3 - len(set(date_coordinates))})
    collision_groups = [members for members in all_coordinates.values() if len(members) > 1]
    return {
        "dimension": dimension,
        "components": rows,
        "unique_active_coordinates": len(all_coordinates),
        "collision_count": sum(len(members) - 1 for members in collision_groups),
        "collision_component_pairs": [
            (members[i], members[j])
            for members in collision_groups
            for i in range(len(members)) for j in range(i + 1, len(members))
        ],
        "max_coordinate_count": max((len(members) for members in all_coordinates.values()), default=0),
        "count_gt_one_coordinates": len(collision_groups),
    }


class SkillIRAutoencoder(nn.Module):
    def __init__(self, code_size: int, hidden_size: int = 1024, input_size: int = IR_DIMENSION):
        super().__init__()
        self.input_size, self.code_size, self.hidden_size = input_size, code_size, hidden_size
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, code_size))
        self.decoder = nn.Sequential(nn.Linear(code_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, input_size))

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        return self.encoder(value)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encode(value)
        return code, self.decode(code)


def balanced_reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    active = target.gt(0)
    active_loss = ((predicted - target).square() * active).sum(-1) / active.sum(-1).clamp_min(1)
    inactive_loss = (predicted.square() * ~active).sum(-1) / (~active).sum(-1).clamp_min(1)
    return 0.5 * (active_loss + inactive_loss).mean()


@torch.no_grad()
def reconstruction_metrics(predicted: torch.Tensor, target: torch.Tensor, reachable: torch.Tensor | None = None) -> dict[str, float]:
    quantized = predicted.clamp_min(0).round()
    active_target, active_pred = target.gt(0), quantized.gt(0)
    exact = (quantized == target).all(-1).float()
    precision = (active_pred & active_target).sum(-1) / active_pred.sum(-1).clamp_min(1)
    recall = (active_pred & active_target).sum(-1) / active_target.sum(-1).clamp_min(1)
    result = {
        "balanced_loss": float(balanced_reconstruction_loss(predicted, target)),
        "mse": float((predicted - target).square().mean()),
        "mae": float((predicted - target).abs().mean()),
        "max_abs_error": float((predicted - target).abs().max()),
        "quantized_exact_hash_accuracy": float(exact.mean()),
        "active_precision": float(precision.mean()),
        "active_recall": float(recall.mean()),
        "total_mass_error": float((predicted.sum(-1) - target.sum(-1)).abs().mean()),
    }
    if reachable is not None:
        result["fraction_active_train_reachable"] = float((active_target & reachable).sum(-1).float().div(active_target.sum(-1).clamp_min(1)).mean())
    return result


def code_diagnostics(code: torch.Tensor) -> dict[str, float]:
    flat = code.detach().float()
    normalized = torch.nn.functional.normalize(flat, dim=-1)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~torch.eye(len(flat), dtype=torch.bool, device=flat.device)]
    return {
        "latent_mean_variance": float(flat.var(0, unbiased=False).mean()),
        "latent_mean_pairwise_cosine": float(off_diagonal.mean()) if off_diagonal.numel() else 1.0,
        "latent_output_variance": float(flat.var(unbiased=False)),
    }
