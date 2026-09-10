"""Stage-4-A1: learner-visible acquisition into the frozen P0 contract."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

from tool_lora.stage4_p0.acquisition import acquire, parse_observations
from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest, ToolDecision, interpret
from benchmarks.stage4_p0_oracle_tool_policy import SCHEMA, make_constant_ir, make_ir, request

ROOT = Path(__file__).parents[1]
P0_FIXTURE = ROOT / "results/stage4_p0_oracle_tool_policy/fixture.json"
OUT = ROOT / "results/stage4_a1_learned_tool_policy"


def render_support(ir: GenericPolicyIR, *, variant: str, skill: int) -> str:
    """Render four blinded, semi-natural demonstrations from observable traces."""
    entities = [f"entity-{variant}-{skill}-{suffix}" for suffix in ("oak", "pine", "cedar", "birch")]
    phrasings = (
        "Please process {entity}. This request is marked {condition}.",
        "Handle {entity}; the current status is {condition}.",
        "Could you take care of {entity}? It is a {condition} case.",
        "I need {entity} handled today, and its priority is {condition}.",
    )
    lines = []
    for index, entity in enumerate(entities):
        condition = "urgent" if index % 2 == 0 else "routine"
        branch = ir.alternatives[0] if condition == ir.condition_value else ir.alternatives[1]
        literal = dict(ir.lexical_bindings)["scope"]
        text = phrasings[index].format(entity=entity, condition=condition)
        trace = {"action": branch.action, "arguments": {"entity": entity, "scope": literal}}
        lines.append(f"Example {index}: {text}\nObserved action trace: {json.dumps(trace, sort_keys=True)}\n")
    return "".join(lines)


def heldout_request(skill: int, condition: str) -> MockRequest:
    return MockRequest(f"entity-heldout-a1-{skill}-silver", (("priority", condition),))


def identifiability_audit(supports: dict[str, list[str]], truths: list[GenericPolicyIR]) -> dict[str, object]:
    rows = []
    for key, sets in supports.items():
        skill = int(key.split(":")[0])
        truth = truths[skill]
        for support in sets:
            observations = parse_observations(support)
            signature = tuple(sorted((value, action, literal) for value, action, _key, literal in observations))
            candidates = []
            for candidate_index, candidate in enumerate(truths):
                candidate_signature = tuple(sorted((
                    value, candidate.alternatives[0 if value == candidate.condition_value else 1].action,
                    dict(candidate.lexical_bindings)["scope"]
                ) for value in ("urgent", "routine") for _ in range(2)))
                if signature == candidate_signature:
                    candidates.append(candidate_index)
            rows.append({"support": key, "candidate_count": len(candidates), "intended_in_candidates": skill in candidates})
    return {"all_unique": all(r["candidate_count"] == 1 and r["intended_in_candidates"] for r in rows), "rows": rows}


def fresh_execute(path: Path, req: MockRequest) -> ToolDecision:
    code = (
        "import json, sys; "
        "from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest, MockToolSchema, interpret; "
        "ir=GenericPolicyIR.load(sys.argv[1]); req=MockRequest(sys.argv[2], tuple(json.loads(sys.argv[3]))); "
        "schema=MockToolSchema(((\"dispatch_alpha\",(\"entity\",\"scope\")),(\"dispatch_beta\",(\"entity\",\"scope\")))); "
        "d=interpret(ir,req,schema); print(json.dumps({\"action\":d.action,\"arguments\":d.arguments}))"
    )
    result = subprocess.check_output([sys.executable, "-c", code, str(path), req.entity, json.dumps(req.attributes)], cwd=ROOT, text=True)
    payload = json.loads(result)
    return ToolDecision(payload["action"], tuple(tuple(x) for x in payload["arguments"]))


def main() -> None:
    fixture = json.loads(P0_FIXTURE.read_text())
    truths = [GenericPolicyIR.from_dict(item) for item in fixture["skills"]]
    supports: dict[str, list[str]] = {}
    for skill, truth in enumerate(truths):
        for variant in ("primary", "alternate", "paraphrase"):
            supports[f"{skill}:{variant}"] = [render_support(truth, variant=variant, skill=skill)]
    audit = identifiability_audit(supports, truths)
    if not audit["all_unique"]:
        raise RuntimeError("fit-free support identifiability audit failed")

    rows = []
    canonical = []
    for skill, truth in enumerate(truths):
        acquired = {variant: acquire(supports[f"{skill}:{variant}"][0]) for variant in ("primary", "alternate", "paraphrase")}
        canonical.append({"skill": skill, "primary_alternate_equal": acquired["primary"] == acquired["alternate"],
                          "primary_paraphrase_equal": acquired["primary"] == acquired["paraphrase"]})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.json"
            acquired["primary"].save(path)
            reloaded = GenericPolicyIR.load(path)
            reload_identity = path.read_bytes() == (json.dumps(acquired["primary"].to_dict(), sort_keys=True, indent=2) + "\n").encode()
            del acquired["primary"]
            matched = {surface: fresh_execute(path, heldout_request(skill, surface)) for surface in ("urgent", "routine")}
        counter_ir = acquire(render_support(make_ir(skill, counterfactual=True), variant="counterfactual", skill=skill))
        counter = {surface: interpret(counter_ir, heldout_request(skill, surface), SCHEMA) for surface in ("urgent", "routine")}
        relation = GenericPolicyIR(reloaded.condition_key, reloaded.condition_value, tuple(reversed(reloaded.alternatives)), reloaded.lexical_bindings)
        lexical = GenericPolicyIR(reloaded.condition_key, reloaded.condition_value, reloaded.alternatives,
                                  (("scope", truths[(skill + 1) % len(truths)].lexical_bindings[0][1]),))
        controls = {
            "relation_shuffled": {surface: interpret(relation, heldout_request(skill, surface), SCHEMA) for surface in ("urgent", "routine")},
            "lexical_shuffled": {surface: interpret(lexical, heldout_request(skill, surface), SCHEMA) for surface in ("urgent", "routine")},
            "constant": {surface: interpret(make_constant_ir(), heldout_request(skill, surface), SCHEMA) for surface in ("urgent", "routine")},
        }
        for surface in ("urgent", "routine"):
            truth_decision = interpret(truth, heldout_request(skill, surface), SCHEMA)
            cases = [("matched", matched[surface], truth_decision),
                     ("counterfactual", counter[surface], interpret(make_ir(skill, counterfactual=True), heldout_request(skill, surface), SCHEMA))]
            cases.extend((name, decisions[surface], truth_decision) for name, decisions in controls.items())
            for condition, decision, target in cases:
                rows.append({"skill": skill, "surface": surface, "condition": condition,
                             "action_correct": decision.action == target.action,
                             "argument_correct": decision.arguments == target.arguments,
                             "opaque_exact": decision.arguments[1][1] == target.arguments[1][1],
                             "reload_identity": reload_identity if condition == "matched" else None})
    conditions = {}
    for condition in ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant"):
        subset = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset),
                                 "argument_accuracy": sum(r["argument_correct"] for r in subset) / len(subset),
                                 "opaque_exact": sum(r["opaque_exact"] for r in subset) / len(subset)}
    primary_artifacts = [acquire(supports[f"{skill}:primary"][0]).to_dict() for skill in range(len(truths))]
    canonical_metrics = {
        "primary_alternate_equality": sum(item["primary_alternate_equal"] for item in canonical) / len(canonical),
        "primary_paraphrase_equality": sum(item["primary_paraphrase_equal"] for item in canonical) / len(canonical),
        "semantic_collision_count": len(primary_artifacts) - len({json.dumps(item, sort_keys=True) for item in primary_artifacts}),
        "relational_recovery": conditions["matched"]["action_accuracy"],
        "lexical_recovery": conditions["matched"]["opaque_exact"],
        "serialization_identity": sum(row["reload_identity"] is True for row in rows if row["condition"] == "matched") / len(truths) / 2,
    }
    manifest = {
        "experiment": "stage4_a1_learned_tool_policy", "p0_fixture_sha256": hashlib.sha256(P0_FIXTURE.read_bytes()).hexdigest(),
        "skill_count": len(truths), "seed": 410, "support_sets_per_skill": 3, "heldout_surfaces_per_skill": 2,
        "identifiability_audit": audit, "canonicalization": canonical, "canonical_metrics": canonical_metrics,
        "conditions": conditions, "rows": rows,
        "learner_visible": ["semi_natural_support", "serialized_p0_ir", "held_out_request", "public_tool_schema"],
        "evaluator_only": ["p0_truth_ir", "candidate_skill_states", "identifiability_labels", "expected_decisions"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages remained unchanged",
        "criteria": {"relational_recovery": 0.95, "lexical_recovery": 1.0, "primary_alternate_equality": 0.95,
                     "primary_paraphrase_equality": 0.95, "semantic_collisions": 0, "serialization_identity": 1.0,
                     "matched_action_argument": 0.95, "counterfactual_action": 0.95},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    support_path = OUT / "immutable_support_evaluation.json"
    support_path.write_text(json.dumps({"supports": supports, "heldout": [
        asdict_request(heldout_request(i, surface)) for i in range(len(truths)) for surface in ("urgent", "routine")
    ]}, sort_keys=True, indent=2) + "\n")
    manifest["support_evaluation_sha256"] = hashlib.sha256(support_path.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "conditions": conditions, "identifiability": audit["all_unique"], "support_sha256": manifest["support_evaluation_sha256"]}, indent=2))


def asdict_request(request: MockRequest) -> dict[str, object]:
    return {"entity": request.entity, "attributes": list(request.attributes)}


if __name__ == "__main__":
    main()
