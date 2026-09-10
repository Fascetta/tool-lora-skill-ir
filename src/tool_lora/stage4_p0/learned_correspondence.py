"""Tiny learned correspondence mechanism for Stage 4-A2.

Training uses only synthetic, anonymized two-condition/two-action structural
examples. Test acquisition receives parsed observable demonstrations and uses
the frozen P0 contract plus exact lexical copying.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from .acquisition import parse_observations
from .contract import ArgumentBinding, GenericPolicyIR, PolicyBranch


class CorrespondenceNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _features(observations: list[tuple[str, str, str, str]]) -> torch.Tensor:
    conditions = sorted({item[0] for item in observations}, reverse=True)
    actions = sorted({item[1] for item in observations})
    if len(conditions) != 2 or len(actions) != 2:
        raise ValueError("correspondence requires two observable conditions and actions")
    values = [0.0] * 4
    for condition, action, _key, _literal in observations:
        values[conditions.index(condition) * 2 + actions.index(action)] += 1.0
    scale = max(values)
    return torch.tensor([value / scale for value in values], dtype=torch.float32)


def _synthetic_examples(seed: int, count: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    features, labels = [], []
    for _ in range(count):
        conditions = [f"condition-{rng.randrange(10**9)}-{j}" for j in range(2)]
        actions = [f"action-{rng.randrange(10**9)}-{j}" for j in range(2)]
        first_action = rng.randrange(2)
        # Tokens are anonymized; only the generic correspondence is retained.
        rows = [(conditions[0], actions[first_action]), (conditions[0], actions[first_action]),
                (conditions[1], actions[1 - first_action]), (conditions[1], actions[1 - first_action])]
        observations = [(condition, action, "arg", "opaque") for condition, action in rows]
        features.append(_features(observations))
        # Targets use the same canonical condition coordinate system as
        # ``_features``: the first row is the reverse-sorted condition.
        canonical_first_condition = sorted(conditions, reverse=True)[0]
        canonical_action = next(action for condition, action in rows if condition == canonical_first_condition)
        labels.append(sorted(actions).index(canonical_action))
    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def supervision_consistency_audit(seed: int, count: int = 128) -> float:
    """Evaluator-only check that generated targets match canonical features."""
    features, labels = _synthetic_examples(seed, count)
    correct = 0
    for feature, label in zip(features, labels):
        # A canonical 2x2 relation has exactly one active action per row;
        # the target is the action column active in the first canonical row.
        expected = int(feature[:2].argmax().item())
        correct += int(expected == int(label))
    accuracy = correct / count
    if accuracy != 1.0:
        raise RuntimeError(f"training supervision consistency audit failed: {accuracy:.6f}")
    return accuracy


@dataclass(frozen=True)
class TrainedCorrespondence:
    model: CorrespondenceNet
    seed: int
    training_examples: int


def train(seed: int, steps: int = 500) -> TrainedCorrespondence:
    torch.manual_seed(seed)
    supervision_consistency_audit(seed + 1000)
    model = CorrespondenceNet()
    features, labels = _synthetic_examples(seed + 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(features), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return TrainedCorrespondence(model, seed, len(labels))


def acquire_with_model(trained: TrainedCorrespondence, support: str) -> GenericPolicyIR:
    observations = parse_observations(support)
    conditions = sorted({item[0] for item in observations}, reverse=True)
    actions = sorted({item[1] for item in observations})
    if len(conditions) != 2 or len(actions) != 2:
        raise ValueError("support must contain exactly two conditions and actions")
    prediction = int(trained.model(_features(observations).unsqueeze(0)).argmax(-1).item())
    first_action = actions[prediction]
    by_condition = {condition: {item[1] for item in observations if item[0] == condition} for condition in conditions}
    if any(len(values) != 1 for values in by_condition.values()):
        raise ValueError("support relation is not deterministic")
    if first_action not in by_condition[conditions[0]]:
        raise ValueError("learned correspondence disagrees with observable relation")
    second_action = next(iter(by_condition[conditions[1]]))
    literals = {item[3] for item in observations}
    if len(literals) != 1:
        raise ValueError("opaque lexical binding is inconsistent")
    return GenericPolicyIR(
        "priority", conditions[0],
        tuple(PolicyBranch(action, (ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope")))
              for action in (first_action, second_action)),
        (("scope", next(iter(literals))),),
    )
