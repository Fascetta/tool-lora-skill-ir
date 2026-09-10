from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import permutations
from typing import Any

from .contract import EffectAnchoredIR

_PRESERVING = {"show", "get", "list", "search", "find", "check", "read", "view"}
_MUTATING = {"add", "append", "approve", "create", "delete", "mark", "move", "remove", "send", "set", "update", "updated", "write"}


def _text(schema: dict[str, Any]) -> str:
    f = schema["function"]
    # Operation semantics come from the public operation description/name;
    # parameter prose can mention unrelated words (for example, a list of
    # valid values) and must not override the operation-level evidence.
    return " ".join([str(f.get("description", "")), str(f.get("name", "")).replace("_", " ")]).lower()


def infer_effect(schema: dict[str, Any]) -> str | None:
    tokens = set(re.findall(r"[a-z]+", _text(schema)))
    preserving = bool(tokens & _PRESERVING)
    mutating = bool(tokens & _MUTATING)
    if preserving == mutating:
        return None
    return "state_preserving" if preserving else "state_mutating"


def _identifier_property(schema: dict[str, Any]) -> str | None:
    props = schema["function"].get("parameters", {}).get("properties", {})
    candidates = []
    for name, spec in props.items():
        desc = str(spec.get("description", "")).lower()
        if spec.get("type") == "integer" and (name.endswith("_id") or "id of" in desc):
            candidates.append(name)
    return sorted(candidates)[0] if len(candidates) == 1 else None


def compatible(schema: dict[str, Any]) -> bool:
    props = schema["function"].get("parameters", {}).get("properties", {})
    return infer_effect(schema) is not None and _identifier_property(schema) is not None and any(
        spec.get("type") == "string" and "access token" in str(spec.get("description", "")).lower()
        for spec in props.values()
    )


def candidate_effect_mapping(ir: EffectAnchoredIR, schemas: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    result = {effect: [] for _, effect in ir.role_effects}
    for name, schema in schemas.items():
        effect = infer_effect(schema)
        if compatible(schema) and effect in result:
            result[effect].append(name)
    return {key: sorted(value) for key, value in result.items()}


def audit_identifiability(irs: list[EffectAnchoredIR], schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for index, ir in enumerate(irs):
        roles = sorted(dict(ir.role_effects))
        compatible_tools = sorted(name for name, schema in schemas.items() if compatible(schema))
        assignments = []
        for assignment in permutations(compatible_tools, len(roles)):
            mapping = dict(zip(roles, assignment))
            if all(infer_effect(schemas[mapping[role]]) == dict(ir.role_effects)[role] for role in roles):
                assignments.append(mapping)
        rows.append({"index": index, "candidate_mapping_count": len(assignments), "mappings": assignments,
                     "compatible_tools": compatible_tools})
    return {"accuracy": sum(row["candidate_mapping_count"] == 1 for row in rows) / len(rows),
            "unique_mapping": all(row["candidate_mapping_count"] == 1 for row in rows), "rows": rows,
            "evidence": ["generic role effect property", "public names/descriptions/argument schemas"],
            "excluded": ["application names", "evaluator IDs", "gold mappings", "API lookup tables"]}


@dataclass(frozen=True)
class ResolvedCall:
    api: str
    arguments: tuple[tuple[str, str], ...]


def resolve(ir: EffectAnchoredIR, condition: str, entity: str, schemas: dict[str, dict[str, Any]]) -> ResolvedCall:
    mapping = candidate_effect_mapping(ir, schemas)
    if any(len(values) != 1 for values in mapping.values()):
        raise ValueError("effect-anchored schema mapping is not unique")
    action = ir.base.alternatives[0].action if condition == ir.base.condition_value else ir.base.alternatives[1].action
    effect = dict(ir.role_effects)[action]
    api_name = mapping[effect][0]
    schema = schemas[api_name]
    identifier = _identifier_property(schema)
    literal = dict(ir.base.lexical_bindings)["scope"]
    # The public schema supplies the concrete parameter name; auth is injected
    # by the AppWorld worker, not represented in the persistent IR.
    return ResolvedCall(api_name.replace("__", "."), ((identifier, literal),))
