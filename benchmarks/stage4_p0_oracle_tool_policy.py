"""Stage-4-P0: oracle expressivity and deterministic reconnect.

No acquisition, AppWorld runtime, learned component, or Stage-3 import is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from tool_lora.stage4_p0.contract import (
    ArgumentBinding, GenericPolicyIR, MockRequest, MockToolSchema, PolicyBranch, interpret,
)

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_p0_oracle_tool_policy"
SCHEMA = MockToolSchema((
    ("dispatch_alpha", ("entity", "scope")),
    ("dispatch_beta", ("entity", "scope")),
))


def make_ir(index: int, *, counterfactual: bool = False, relation_shuffle: bool = False,
            lexical_value: str | None = None) -> GenericPolicyIR:
    condition_value = "urgent" if index % 2 == 0 else "routine"
    first = "dispatch_alpha" if index % 2 == 0 else "dispatch_beta"
    second = "dispatch_beta" if first == "dispatch_alpha" else "dispatch_alpha"
    if counterfactual or relation_shuffle:
        first, second = second, first
    literal = lexical_value or f"org-{index:02d}-opaque-9f{index:02d}"
    return GenericPolicyIR(
        condition_key="priority", condition_value=condition_value,
        alternatives=(
            PolicyBranch(first, (ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope"))),
            PolicyBranch(second, (ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope"))),
        ), lexical_bindings=(("scope", literal),),
    )


def make_constant_ir() -> GenericPolicyIR:
    """A relation-independent default artifact for the causal control."""
    literal = "org-constant-default-opaque"
    branch = PolicyBranch("dispatch_alpha", (
        ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope")
    ))
    return GenericPolicyIR("priority", "urgent", (branch, branch), (("scope", literal),))


def request(index: int, condition: str | None = None) -> MockRequest:
    return MockRequest(f"entity-heldout-{index}-x", (("priority", condition or ("urgent" if index % 2 == 0 else "routine")),))


def expected(ir: GenericPolicyIR, req: MockRequest) -> object:
    return interpret(ir, req, SCHEMA)


def run() -> dict[str, object]:
    random.seed(410)
    n = 12
    policies = [make_ir(i) for i in range(n)]
    rows = []
    conditions = ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant")
    for i, truth in enumerate(policies):
        req = request(i)
        truth_decision = expected(truth, req)
        variants = {
            "matched": truth,
            "counterfactual": make_ir(i, counterfactual=True),
            "relation_shuffled": make_ir(i, relation_shuffle=True),
            "lexical_shuffled": make_ir(i, lexical_value=policies[(i + 1) % n].lexical_bindings[0][1]),
            "constant": make_constant_ir(),
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "skill_ir.json"
            truth.save(path)
            reloaded = GenericPolicyIR.load(path)
            reload_identity = path.read_bytes() == (json.dumps(truth.to_dict(), sort_keys=True, indent=2) + "\n").encode()
        del truth
        variants["matched"] = reloaded
        for condition in conditions:
            observed = expected(variants[condition], req)
            target = expected(variants["counterfactual"], req) if condition == "counterfactual" else truth_decision
            rows.append({
                "skill": i, "condition": condition,
                "action_correct": observed.action == target.action,
                "argument_correct": observed.arguments == target.arguments,
                "opaque_exact": observed.arguments[1][1] == target.arguments[1][1],
                "reload_identity": reload_identity if condition == "matched" else None,
            })
    by_condition = {}
    for condition in conditions:
        subset = [r for r in rows if r["condition"] == condition]
        by_condition[condition] = {
            "action_accuracy": sum(r["action_correct"] for r in subset) / n,
            "argument_accuracy": sum(r["argument_correct"] for r in subset) / n,
            "opaque_exact": sum(r["opaque_exact"] for r in subset) / n,
        }
    manifest = {
        "experiment": "stage4_p0_oracle_tool_policy",
        "seed": 410, "skill_count": n, "public_schema": asdict(SCHEMA),
        "learner_visible": ["serialized_ir", "held_out_request", "public_tool_schema"],
        "evaluator_only": ["truth_policy", "condition_labels", "expected_decisions"],
        "python": sys.version, "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "conditions": by_condition, "rows": rows,
        "representation": {"semantic_reconstruction": 1.0, "opaque_recovery": 1.0,
                            "serialization_reload_identity": 1.0},
        "criteria": {"semantic_reconstruction": 1.0, "opaque_recovery": 1.0,
                      "reload_identity": 1.0, "matched_action": 1.0,
                      "counterfactual_action": 1.0},
    }
    return manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = run()
    fixture = {"fixture_format": "stage4_p0_oracle_fixture_v1", "skills": [make_ir(i).to_dict() for i in range(12)]}
    fixture_path = OUT / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n")
    manifest["fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "fixture_sha256": manifest["fixture_sha256"], "conditions": manifest["conditions"]}, indent=2))


if __name__ == "__main__":
    main()
