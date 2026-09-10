"""Ontology-free structural resolver for the persistent hybrid Skill IR."""
from __future__ import annotations
from .protocol_ir import ExplicitSkillIR

def apply_permutation(source, permutation):
    """Persisted convention: ``target[j] = source[permutation[j]]``."""
    if len(source) != len(permutation) or sorted(permutation) != list(range(len(source))):
        raise ValueError("permutation must be a bijection over source positions")
    return tuple(source[index] for index in permutation)

def resolve_values(ir) -> tuple[str, ...]:
    if not ir.records: raise ValueError("empty persistent artifact")
    base = sorted((r for r in ir.records if not r.transformations), key=lambda r: (r.predicate, r.output_key))
    transformed = sorted((r for r in ir.records if r.transformations), key=lambda r: (r.predicate, r.output_key))
    if not base or not transformed: raise ValueError("artifact lacks distinct record families")
    observations = {}
    for record in (base[0], transformed[0]):
        for relation, key in record.observations: observations.setdefault(relation, set()).add(key)
    def unique(relation):
        values = sorted(observations.get(relation, ()))
        if len(values) != 1: raise ValueError(f"ambiguous or missing relation {relation}: {values}")
        return values[0]
    exact, enum, transformed_key = unique("exact"), unique("enum"), unique("transformed")
    literals = {context: value for context, key, value in (x for r in ir.records for x in r.literal_bindings) if key == enum and context}
    if len(literals) != 2: raise ValueError(f"expected two literal bindings, got {literals}")
    hypotheses = sorted({t for r in transformed for t in r.transformations})
    if len(hypotheses) != 1: raise ValueError(f"ambiguous transformation records: {hypotheses}")
    permutation, _source_delimiter, target_delimiter = hypotheses[0]
    names = {(0, 1, 2): "YMD", (1, 2, 0): "MDY", (2, 1, 0): "DMY"}
    if tuple(permutation) not in names: raise ValueError(f"unknown component order: {permutation}")
    return (base[0].output_key, transformed[0].output_key, exact, transformed_key, enum, *[v for _k, v in sorted(literals.items())], names[tuple(permutation)], target_delimiter)

def resolve_explicit(ir, device=None):
    result = ExplicitSkillIR.from_values(resolve_values(ir))
    return result if device is None else ExplicitSkillIR(result.protocol_values, result.legacy_ir_tensor.to(device), result.schema_version, result.confidence)
