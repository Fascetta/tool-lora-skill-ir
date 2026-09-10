from __future__ import annotations

import json
import re
from typing import Any

from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import infer_effect, compatible, _identifier_property
from tool_lora.stage4_p0.contract import ArgumentBinding, GenericPolicyIR, PolicyBranch

CONDITION_RE = re.compile(r"\b(urgent|routine)\b", re.I)


def _schema_map(demos: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for demo in demos:
        schemas.update(demo["public_schemas"])
    return schemas


def acquire(support_text: str) -> EffectAnchoredIR:
    demos = json.loads(support_text)
    if not demos:
        raise ValueError("empty support")
    schemas = _schema_map(demos)
    candidates = sorted(name for name, schema in schemas.items() if compatible(schema))
    if len(candidates) != 2:
        raise ValueError("support must expose exactly two compatible public actions")
    role_by_api = {name: f"dispatch_{'alpha' if index == 0 else 'beta'}" for index, name in enumerate(candidates)}
    observations = []
    for demo in demos:
        match = CONDITION_RE.search(str(demo["request"]["text"]))
        if match is None:
            raise ValueError("support lacks an observable request condition")
        api = str(demo["tool_call"]["api"])
        schema_api = api.replace(".", "__")
        if schema_api not in role_by_api:
            raise ValueError("tool call is absent from public candidate schemas")
        identifier = _identifier_property(schemas[schema_api])
        if identifier is None or identifier not in demo["tool_call"]["arguments"]:
            raise ValueError("support lacks an exact generic object binding")
        observations.append((match.group(1).lower(), role_by_api[schema_api], str(demo["tool_call"]["arguments"][identifier])))
    conditions = sorted({x[0] for x in observations}, reverse=True)
    if len(conditions) != 2:
        raise ValueError("support must contain two conditions")
    literals = {x[2] for x in observations}
    if len(literals) != 1:
        raise ValueError("opaque binding is inconsistent")
    branches = []
    for condition in conditions:
        actions = {x[1] for x in observations if x[0] == condition}
        if len(actions) != 1:
            raise ValueError("relation is ambiguous")
        branches.append(PolicyBranch(next(iter(actions)), (ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope"))))
    base = GenericPolicyIR("priority", conditions[0], tuple(branches), (("scope", next(iter(literals))),))
    effects = tuple((role_by_api[api], infer_effect(schemas[api])) for api in candidates)
    if any(effect is None for _, effect in effects):
        raise ValueError("public schema effect is not generically classifiable")
    if len({effect for _, effect in effects}) != 2:
        raise ValueError("effect anchor is not identifiable")
    return EffectAnchoredIR(base, effects)


def audit_support(support_text: str) -> dict[str, Any]:
    demos = json.loads(support_text)
    schemas = _schema_map(demos)
    compatible_candidates = sorted(name for name, schema in schemas.items() if compatible(schema))
    effects = {name: infer_effect(schemas[name]) for name in compatible_candidates}
    return {"candidate_count": len(compatible_candidates), "compatible_tools": compatible_candidates,
            "effect_classes": effects, "unique": len(compatible_candidates) == 2 and len(set(effects.values())) == 2}
