"""Hybrid structured Skill IR: lexical sidecar plus relational code."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

FUNCTIONS = ("weather", "forecast")
FIELDS = ("city", "date", "units")
ENUMS = ("celsius", "fahrenheit")
DATE_ORDERS = ("DMY", "MDY", "YMD")
FUNCTION_PERMS = tuple(itertools.permutations(range(2)))
FIELD_PERMS = tuple(itertools.permutations(range(3)))
ENUM_PERMS = tuple(itertools.permutations(range(2)))


@dataclass(frozen=True)
class LexicalSidecar:
    functions: tuple[str, str]
    fields: tuple[str, str, str]
    enums: tuple[str, str]
    separator: str

    @classmethod
    def from_values(cls, values: Iterable[str]) -> "LexicalSidecar":
        values = tuple(values)
        return cls(tuple(sorted(values[:2])), tuple(sorted(values[2:5])),
                   tuple(sorted(values[5:7])), values[8])

    def to_dict(self) -> dict[str, object]:
        return {"functions": list(self.functions), "fields": list(self.fields),
                "enums": list(self.enums), "separator": self.separator}

    def byte_size(self) -> int:
        return sum(len(value.encode("utf-8")) for value in (*self.functions, *self.fields, *self.enums, self.separator))


def relation_labels(values: Iterable[str]) -> tuple[int, int, int, int]:
    values = tuple(values)
    sidecar = LexicalSidecar.from_values(values)
    return (
        FUNCTION_PERMS.index(tuple(sidecar.functions.index(values[index]) for index in range(2))),
        FIELD_PERMS.index(tuple(sidecar.fields.index(values[index]) for index in range(2, 5))),
        ENUM_PERMS.index(tuple(sidecar.enums.index(values[index]) for index in range(5, 7))),
        DATE_ORDERS.index(values[7]),
    )


def relation_input(values: Iterable[str]) -> torch.Tensor:
    labels = relation_labels(values)
    result = torch.zeros(20, dtype=torch.float32)
    function_perm = FUNCTION_PERMS[labels[0]]
    field_perm = FIELD_PERMS[labels[1]]
    enum_perm = ENUM_PERMS[labels[2]]
    result[0:4] = F.one_hot(torch.tensor(function_perm), 2).reshape(-1).float()
    result[4:13] = F.one_hot(torch.tensor(field_perm), 3).reshape(-1).float()
    result[13:17] = F.one_hot(torch.tensor(enum_perm), 2).reshape(-1).float()
    result[17 + labels[3]] = 1.0
    return result


def reconstruct_values(sidecar: LexicalSidecar, labels: tuple[int, int, int, int]) -> tuple[str, ...]:
    function_perm, field_perm, enum_perm, order = (FUNCTION_PERMS[labels[0]], FIELD_PERMS[labels[1]], ENUM_PERMS[labels[2]], labels[3])
    functions = tuple(sidecar.functions[index] for index in function_perm)
    fields = tuple(sidecar.fields[index] for index in field_perm)
    enums = tuple(sidecar.enums[index] for index in enum_perm)
    return (*functions, *fields, *enums, DATE_ORDERS[order], sidecar.separator)


class StructuredHybridAutoencoder(nn.Module):
    def __init__(self, code_size: int = 16, hidden_size: int = 64):
        super().__init__()
        self.code_size, self.hidden_size = code_size, hidden_size
        self.encoder = nn.Sequential(nn.Linear(20, hidden_size), nn.GELU(), nn.Linear(hidden_size, code_size))
        self.decoder = nn.Sequential(nn.Linear(code_size, hidden_size), nn.GELU())
        self.function_head = nn.Linear(hidden_size, len(FUNCTION_PERMS))
        self.field_head = nn.Linear(hidden_size, len(FIELD_PERMS))
        self.enum_head = nn.Linear(hidden_size, len(ENUM_PERMS))
        self.order_head = nn.Linear(hidden_size, len(DATE_ORDERS))

    def encode(self, relation: torch.Tensor) -> torch.Tensor:
        return self.encoder(relation)

    def forward(self, relation: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        code = self.encode(relation)
        hidden = self.decoder(code)
        return code, (self.function_head(hidden), self.field_head(hidden), self.enum_head(hidden), self.order_head(hidden))


def structured_loss(logits: tuple[torch.Tensor, ...], labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([F.cross_entropy(logit, labels[:, index]) for index, logit in enumerate(logits)]).mean()


@torch.no_grad()
def structured_metrics(logits: tuple[torch.Tensor, ...], labels: torch.Tensor) -> dict[str, float]:
    predictions = torch.stack([logit.argmax(-1) for logit in logits], dim=1)
    component = predictions.eq(labels)
    return {
        "function_accuracy": float(component[:, 0].float().mean()),
        "field_accuracy": float(component[:, 1].float().mean()),
        "enum_accuracy": float(component[:, 2].float().mean()),
        "date_order_accuracy": float(component[:, 3].float().mean()),
        "relation_exact_accuracy": float(component.all(-1).float().mean()),
    }


def artifact_bytes(sidecar: LexicalSidecar, code: torch.Tensor, *, dtype_bytes: int = 4, schema_overhead: int = 16) -> dict[str, int]:
    lexical = sidecar.byte_size()
    semantic = int(code.numel()) * dtype_bytes
    return {"lexical_bytes": lexical, "semantic_code_bytes": semantic,
            "schema_overhead_bytes": schema_overhead,
            "total_bytes": lexical + semantic + schema_overhead}
