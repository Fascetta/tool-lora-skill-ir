"""Deterministic, ontology-free canonicalization of correspondence hypotheses."""
from __future__ import annotations
from collections import defaultdict
from .acquisition import PersistentSkillIR, RelationalRecord

def choose_hypothesis(evidence):
    groups = defaultdict(lambda: {"count": 0, "score": 0.0, "margin": 0.0})
    for item in evidence:
        group = groups[item["hypothesis"]]; group["count"] += 1; group["score"] += item.get("assignment_score", 0.0); group["margin"] += item.get("confidence_margin", 0.0)
    ranked = sorted(groups.items(), key=lambda x: (x[1]["count"], x[1]["score"], x[1]["margin"], x[0]), reverse=True)
    if not ranked: return None
    if len(ranked) > 1 and ranked[0][1]["count"] == ranked[1][1]["count"] and ranked[0][1]["score"] == ranked[1][1]["score"]:
        return None
    return ranked[0][0]

def canonicalize(ir: PersistentSkillIR, evidence) -> PersistentSkillIR:
    winner = choose_hypothesis(evidence)
    return PersistentSkillIR(tuple(RelationalRecord(r.predicate, r.observations, r.output_key, (winner,) if winner is not None and r.transformations else (), r.literal_bindings) for r in ir.records))
