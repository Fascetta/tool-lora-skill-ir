"""B2 adapter from learner-visible AppWorld traces to frozen A2-R1."""
from __future__ import annotations

import json
import re

import torch

from tool_lora.stage4_p0.contract import ArgumentBinding, GenericPolicyIR, PolicyBranch
from tool_lora.stage4_p0.learned_correspondence import (
    TrainedCorrespondence,
    _features,
)

# This is public-schema normalization only: it names the two observable
# action alternatives without deciding which condition selects which one.
PUBLIC_API_TO_GENERIC_ACTION = {
    "todoist.show_task": "dispatch_alpha",
    "todoist.update_task": "dispatch_beta",
}
CONDITION_RE = re.compile(r"\b(urgent|routine)\b", re.I)


def _observations(support_text: str) -> list[tuple[str, str, str, str]]:
    demos = json.loads(support_text)
    observations = []
    for demo in demos:
        request = demo["request"]["text"]
        match = CONDITION_RE.search(request)
        if match is None:
            raise ValueError("request text lacks an observable condition")
        api = str(demo["tool_call"]["api"])
        action = PUBLIC_API_TO_GENERIC_ACTION.get(api)
        if action is None:
            raise ValueError(f"unsupported public API: {api}")
        arguments = demo["tool_call"]["arguments"]
        if "task_id" not in arguments:
            raise ValueError("observable tool call lacks its exact binding")
        literal = str(arguments["task_id"])
        observations.append((match.group(1).lower(), action, "scope", literal))
    if not observations:
        raise ValueError("empty AppWorld support")
    return observations


def acquire_with_frozen_model(trained: TrainedCorrespondence, support_text: str) -> GenericPolicyIR:
    """Use frozen A2-R1 for relation choice and generic copy for the literal."""
    observations = _observations(support_text)
    conditions = sorted({item[0] for item in observations}, reverse=True)
    actions = sorted({item[1] for item in observations})
    if len(conditions) != 2 or len(actions) != 2:
        raise ValueError("support must contain two observable conditions and actions")

    prediction = int(trained.model(_features(observations).unsqueeze(0)).argmax(-1).item())
    first_action = actions[prediction]
    by_condition = {
        condition: {item[1] for item in observations if item[0] == condition}
        for condition in conditions
    }
    if any(len(values) != 1 for values in by_condition.values()):
        raise ValueError("observable route is not deterministic")
    if first_action not in by_condition[conditions[0]]:
        raise ValueError("frozen correspondence model disagrees with observable relation")
    second_action = next(iter(by_condition[conditions[1]]))

    literals = {item[3] for item in observations}
    if len(literals) != 1:
        raise ValueError("opaque binding is inconsistent")
    return GenericPolicyIR(
        "priority",
        conditions[0],
        tuple(
            PolicyBranch(
                action,
                (
                    ArgumentBinding("entity", "entity"),
                    ArgumentBinding("scope", "literal", "scope"),
                ),
            )
            for action in (first_action, second_action)
        ),
        (("scope", next(iter(literals))),),
    )
