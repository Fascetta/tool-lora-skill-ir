"""Frozen P6-R1 generic learned correspondence matcher."""
from __future__ import annotations

import itertools
import re
from pathlib import Path
import torch
from torch import nn
from .acquisition import PersistentSkillIR, RelationalRecord, acquire, observable_transformations

PERMUTATIONS = tuple(itertools.permutations(range(3)))


class ExactPermutationMatcher:
    """Constraint-exact correspondence for observable structured values.

    This is intentionally separate from the frozen P6-R1 model.  It is a
    candidate replacement for domains where the source components and target
    occurrences are directly observable; it does not claim to solve opaque
    correspondence in general.
    """

    parameter_count = 0

    def predict(self, source: str, target: str) -> tuple[tuple[int, ...], str, str]:
        candidates = observable_transformations(source, target)
        if len(candidates) != 1:
            raise ValueError("observable correspondence is ambiguous")
        return candidates[0]

def _encode(values, embedding, encoder):
    device = next(encoder.parameters()).device
    lengths = torch.tensor([len(v.encode()) for v in values], device=device)
    width = int(lengths.max().item())
    tokens = torch.zeros(len(values), width, dtype=torch.long, device=device)
    for row, value in enumerate(values):
        raw = list(value.encode()); tokens[row, :len(raw)] = torch.tensor(raw, device=device)
    hidden, _ = encoder(embedding(tokens))
    mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    return (hidden * mask[..., None]).sum(1) / lengths[:, None].clamp_min(1)

class CorrespondenceMatcher(nn.Module):
    def __init__(self, hidden: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(256, 16)
        self.encoder = nn.GRU(16, hidden, batch_first=True, bidirectional=True)
        self.compatibility = nn.Sequential(nn.Linear(hidden * 4, 32), nn.GELU(), nn.Linear(32, 1))

    def matrix(self, source: str, target: str) -> torch.Tensor:
        source_parts = tuple(re.findall(r"\d+", source))
        candidates = observable_transformations(source, target)
        if len(source_parts) != 3 or not candidates:
            raise ValueError("P6 matcher expects three observable components")
        target_parts = tuple(source_parts[i] for i in sorted(range(3), key=lambda i: target.find(source_parts[i])))
        left = _encode(list(source_parts), self.embedding, self.encoder)[:, None, :].expand(3, 3, -1)
        right = _encode(list(target_parts), self.embedding, self.encoder)[None, :, :].expand(3, 3, -1)
        return self.compatibility(torch.cat((left, right), -1)).squeeze(-1)

    def assignment(self, matrix: torch.Tensor) -> tuple[int, ...]:
        return max(((sum(float(matrix.detach()[perm[j], j]) for j in range(3)), perm) for perm in PERMUTATIONS), key=lambda x: (x[0], x[1]))[1]

def load_p6(path: str | Path, device: torch.device | str = "cpu") -> CorrespondenceMatcher:
    model = CorrespondenceMatcher().to(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(payload["model"]); model.eval()
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    return model

def acquire_with_matcher(model: CorrespondenceMatcher, support_text: str) -> PersistentSkillIR:
    inferred = []
    from .acquisition import _records
    for _predicate, request, response in _records(support_text):
        source = request.get("structured")
        if not source: continue
        for arguments in response.values():
            for value in arguments.values():
                candidates = observable_transformations(source, str(value))
                if len(candidates) == 1:
                    matrix = model.matrix(source, str(value))
                    permutation = model.assignment(matrix)
                    inferred.append((permutation, candidates[0][1], candidates[0][2]))
    base = acquire(support_text, tuple(sorted(set(inferred))))
    return base


def acquire_with_exact_matcher(matcher: ExactPermutationMatcher, support_text: str) -> PersistentSkillIR:
    """Acquire with exact observable constraints instead of learned scores."""
    inferred = []
    from .acquisition import _records
    for _predicate, request, response in _records(support_text):
        source = request.get("structured")
        if not source:
            continue
        for arguments in response.values():
            for value in arguments.values():
                candidates = observable_transformations(source, str(value))
                if len(candidates) == 1:
                    inferred.append(matcher.predict(source, str(value)))
    return acquire(support_text, tuple(sorted(set(inferred))))
