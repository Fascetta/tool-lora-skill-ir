"""Deterministic mapping from frozen P0 semantics to AppWorld calls."""
from __future__ import annotations

from dataclasses import dataclass

from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest, interpret


@dataclass(frozen=True)
class AppWorldCall:
    api: str
    arguments: tuple[tuple[str, int], ...]


API_BY_ABSTRACT_ACTION = {
    "dispatch_alpha": "todoist.show_task",
    "dispatch_beta": "todoist.update_task",
}


def resolve(ir: GenericPolicyIR, request: MockRequest) -> AppWorldCall:
    schema = type("Schema", (), {"arguments_for": lambda _self, _action: ("entity", "scope")})()
    decision = interpret(ir, request, schema)
    if decision.action not in API_BY_ABSTRACT_ACTION:
        raise ValueError(f"unsupported abstract action: {decision.action}")
    try:
        task_id = int(dict(decision.arguments)["scope"])
    except (KeyError, ValueError) as exc:
        raise ValueError("opaque binding is not a concrete AppWorld task identifier") from exc
    return AppWorldCall(API_BY_ABSTRACT_ACTION[decision.action], (("task_id", task_id),))
