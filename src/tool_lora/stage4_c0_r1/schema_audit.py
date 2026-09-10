"""Fit-free identifiability audit for generic abstract-action realization."""
from __future__ import annotations

from itertools import permutations
from typing import Any


def _schema_properties(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return schema["function"]["parameters"].get("properties", {})


def _compatible_with_generic_dispatch(schema: dict[str, Any]) -> bool:
    """Check only generic shape evidence, never an API identity or label."""
    properties = _schema_properties(schema)
    has_integer_identifier = any(value.get("type") == "integer" for value in properties.values())
    has_authentication = any(
        value.get("type") == "string"
        and "access token" in str(value.get("description", "")).lower()
        for value in properties.values()
    )
    return has_integer_identifier and has_authentication


def audit_role_identifiability(ir_payloads: list[dict[str, Any]], candidate_schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    roles = sorted({branch["action"] for payload in ir_payloads for branch in payload["alternatives"]})
    compatible_tools = sorted(name for name, schema in candidate_schemas.items() if _compatible_with_generic_dispatch(schema))
    mappings = [dict(zip(roles, assignment)) for assignment in permutations(compatible_tools, len(roles))]
    rows = [{"roles": roles, "compatible_candidate_tools": compatible_tools,
             "candidate_mapping_count": len(mappings), "mappings": mappings}]
    return {
        "accuracy": 1.0 if len(mappings) == 1 else 0.0,
        "unique_mapping": len(mappings) == 1,
        "roles": roles,
        "compatible_candidate_tools": compatible_tools,
        "candidate_mapping_count": len(mappings),
        "rows": rows,
        "evidence_used": ["abstract dispatch-role structure", "public API names", "public descriptions", "public argument schemas"],
        "evidence_excluded": ["evaluator policy IDs", "gold concrete mappings", "prior C0 expected calls", "application-specific lookup tables"],
    }
