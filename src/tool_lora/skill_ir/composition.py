"""Controlled composition over the four structured semantic factors."""
from __future__ import annotations

import torch
from torch import nn


FACTOR_SIZES = (2, 6, 2, 3)


class FactorComposer(nn.Module):
    def __init__(self, codebook_size: int = 128, embedding_size: int = 16, hidden_size: int = 128):
        super().__init__()
        self.factor_embeddings = nn.ModuleList(nn.Embedding(size, embedding_size) for size in FACTOR_SIZES)
        self.network = nn.Sequential(
            nn.Linear(4 * embedding_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, codebook_size),
        )

    def forward(self, factors: torch.Tensor) -> torch.Tensor:
        embedded = torch.cat([embedding(factors[:, index]) for index, embedding in enumerate(self.factor_embeddings)], dim=-1)
        return self.network(embedded)


class SemanticFactorComposer(FactorComposer):
    """Same typed factor inputs, with structured heads instead of arbitrary VQ labels."""
    def __init__(self, embedding_size: int = 16, hidden_size: int = 128):
        super().__init__(codebook_size=1, embedding_size=embedding_size, hidden_size=hidden_size)
        self.heads = nn.ModuleList(nn.Linear(hidden_size, size) for size in FACTOR_SIZES)

    def forward(self, factors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        embedded = torch.cat([embedding(factors[:, index]) for index, embedding in enumerate(self.factor_embeddings)], dim=-1)
        hidden = self.network[0:3](embedded)
        return tuple(head(hidden) for head in self.heads)


class SeparableSemanticComposer(nn.Module):
    """Hard factor-separable composer: each output sees only its typed factor."""
    def __init__(self, embedding_size: int = 16, hidden_size: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_size) for size in FACTOR_SIZES)
        self.branches = nn.ModuleList(nn.Sequential(nn.Linear(embedding_size, hidden_size), nn.GELU()) for _ in FACTOR_SIZES)
        self.heads = nn.ModuleList(nn.Linear(hidden_size, size) for size in FACTOR_SIZES)

    def forward(self, factors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(head(branch(embedding(factors[:, index]))) for index, (embedding, branch, head) in enumerate(zip(self.embeddings, self.branches, self.heads)))


class AdditiveSemanticComposer(nn.Module):
    """Shared additive semantic composer with factor-typed inputs."""
    def __init__(self, embedding_size: int = 16, hidden_size: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, embedding_size) for size in FACTOR_SIZES)
        self.projection = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.GELU(),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_size, size) for size in FACTOR_SIZES)

    def forward(self, factors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        joint = sum((self.projection(embedding(factors[:, index])) for index, embedding in enumerate(self.embeddings)))
        return tuple(head(joint) for head in self.heads)
