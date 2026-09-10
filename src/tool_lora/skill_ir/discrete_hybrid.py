"""Vector-quantized structured Skill IR."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tool_lora.skill_ir.structured_hybrid import (
    DATE_ORDERS, ENUM_PERMS, FIELD_PERMS, FUNCTION_PERMS,
)


class VQStructuredSkill(nn.Module):
    def __init__(self, codebook_size: int = 128, code_dim: int = 16, hidden_size: int = 64):
        super().__init__()
        self.codebook_size, self.code_dim, self.hidden_size = codebook_size, code_dim, hidden_size
        self.encoder = nn.Sequential(nn.Linear(20, hidden_size), nn.GELU(), nn.Linear(hidden_size, code_dim))
        self.codebook = nn.Embedding(codebook_size, code_dim)
        nn.init.normal_(self.codebook.weight, std=0.5)
        self.decoder = nn.Sequential(nn.Linear(code_dim, hidden_size), nn.GELU())
        self.function_head = nn.Linear(hidden_size, len(FUNCTION_PERMS))
        self.field_head = nn.Linear(hidden_size, len(FIELD_PERMS))
        self.enum_head = nn.Linear(hidden_size, len(ENUM_PERMS))
        self.order_head = nn.Linear(hidden_size, len(DATE_ORDERS))

    def decode_code(self, code: torch.Tensor):
        hidden = self.decoder(code)
        return (self.function_head(hidden), self.field_head(hidden), self.enum_head(hidden), self.order_head(hidden))

    def forward(self, relation: torch.Tensor):
        z = self.encoder(relation)
        distances = (z.square().sum(-1, keepdim=True) - 2 * z @ self.codebook.weight.T + self.codebook.weight.square().sum(-1))
        indices = distances.argmin(-1)
        quantized = self.codebook(indices)
        straight_through = z + (quantized - z).detach()
        logits = self.decode_code(straight_through)
        commitment = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        return indices, straight_through, logits, commitment, codebook_loss, distances


@torch.no_grad()
def discrete_metrics(indices: torch.Tensor, labels: torch.Tensor, codebook_size: int) -> dict:
    used, counts = torch.unique(indices, return_counts=True)
    probabilities = counts.float() / len(indices)
    perplexity = float(torch.exp(-(probabilities * probabilities.log()).sum()))
    state_ids = labels[:, 0] * 36 + labels[:, 1] * 6 + labels[:, 2] * 3 + labels[:, 3]
    state_to_codes = {}
    code_to_states = {}
    for state, code in zip(state_ids.tolist(), indices.tolist()):
        state_to_codes.setdefault(str(state), set()).add(int(code))
        code_to_states.setdefault(str(code), set()).add(int(state))
    return {
        "codebook_entries_used": int(len(used)),
        "codebook_usage_fraction": float(len(used) / codebook_size),
        "codebook_perplexity": perplexity,
        "relation_states_observed": int(len(state_to_codes)),
        "states_with_multiple_codes": int(sum(len(values) > 1 for values in state_to_codes.values())),
        "codes_with_multiple_states": int(sum(len(values) > 1 for values in code_to_states.values())),
        "state_to_codes": {key: sorted(value) for key, value in state_to_codes.items()},
        "code_to_states": {key: sorted(value) for key, value in code_to_states.items()},
    }
