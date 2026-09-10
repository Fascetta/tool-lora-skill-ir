"""Deterministic acquisition from learner-visible AppWorld traces."""
from __future__ import annotations

import json
import re

from tool_lora.stage4_p0.contract import ArgumentBinding, GenericPolicyIR, PolicyBranch

API_TO_ACTION = {"todoist.show_task": "dispatch_alpha", "todoist.update_task": "dispatch_beta"}


def acquire(support_text: str) -> GenericPolicyIR:
    demos = json.loads(support_text)
    if not demos:
        raise ValueError("empty AppWorld support")
    observations = []
    for demo in demos:
        match = re.search(r"\b(urgent|routine)\b", str(demo["request"]["text"]), re.I)
        if match is None:
            raise ValueError("request text lacks an observable condition")
        condition = match.group(1).lower()
        api = str(demo["tool_call"]["api"])
        action = API_TO_ACTION.get(api)
        if action is None:
            raise ValueError(f"unsupported public API: {api}")
        task_id = str(demo["tool_call"]["arguments"]["task_id"])
        if demo["observable_result"].get("task_id") not in (None, int(task_id)) and demo["observable_state"].get("task_id") != int(task_id):
            raise ValueError("observable result/state does not match opaque task binding")
        observations.append((condition, action, task_id))
    conditions = sorted({x[0] for x in observations}, reverse=True)
    if len(conditions) != 2:
        raise ValueError("support must contain exactly two observable conditions")
    literals = {x[2] for x in observations}
    if len(literals) != 1:
        raise ValueError("support must contain exactly one exact opaque binding")
    branches = []
    for condition in conditions:
        actions = {x[1] for x in observations if x[0] == condition}
        if len(actions) != 1:
            raise ValueError("observable route is not deterministic")
        branches.append(PolicyBranch(next(iter(actions)), (
            ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope")
        )))
    return GenericPolicyIR("priority", conditions[0], tuple(branches), (("scope", next(iter(literals))),))
