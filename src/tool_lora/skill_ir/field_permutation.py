"""Validated joint six-way field-role permutation scorer."""
from __future__ import annotations

import itertools
import torch
from torch import nn

PERMS = tuple(itertools.permutations(range(3)))
FIELD_ROLES = ("city", "date", "units")


class JointFieldPermutationScorer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.slot = nn.Parameter(torch.randn(3, width) * 0.02)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.score = nn.Sequential(nn.Linear(3 * width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, hidden: torch.Tensor, candidates: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        context = hidden.mean(0)
        q = self.q(self.slot + context)
        k = self.k(torch.stack(candidates))
        z = torch.cat((q[:, None, :].expand(3, 3, -1), k[None, :, :].expand(3, 3, -1), q[:, None, :] * k[None, :, :]), dim=-1)
        matrix = self.score(z).squeeze(-1)
        return torch.stack([matrix[range(3), perm].sum() for perm in PERMS]), matrix
