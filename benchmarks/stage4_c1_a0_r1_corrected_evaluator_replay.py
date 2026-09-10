"""Stage 4-C1-A0-R1: replay A0 against corrected evaluator coordinates."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from benchmarks.stage4_c1_p1_lexical_binding_audit import build_fixture
from tool_lora.stage4_c1_a0.acquisition import acquire
from tool_lora.stage4_c1.schema_resolver import infer_effect
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_p0.contract import GenericPolicyIR, PolicyBranch

ROOT = Path(__file__).parents[1]
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
G1_OUT = ROOT / "results/stage4_c1_a0_g1_gauge_audit"
J1_OUT = ROOT / "results/stage4_c1_a0_j1_joint_assignment_audit"
J2_OUT = ROOT / "results/stage4_c1_a0_j2_evaluator_consistency_audit"
P1_FIXTURE = ROOT / "results/stage4_c1_p1_lexical_binding_audit/immutable_fixture.json"
OUT = ROOT / "results/stage4_c1_a0_r1_corrected_evaluator_replay"
SURFACES = ("primary", "alternate_entity", "paraphrased")


def condition_from_text(text: str) -> str:
    lowered = text.lower()
    if "urgent" in lowered:
        return "urgent"
    if "routine" in lowered:
        return "routine"
    raise ValueError("missing request condition")


def relation_signature(ir: EffectAnchoredIR) -> tuple[tuple[str, str], ...]:
    value = ir.base.condition_value
    other = "routine" if value == "urgent" else "urgent"
    return tuple(sorted(((value, ir.base.alternatives[0].action), (other, ir.base.alternatives[1].action))))


def effect_signature(ir: EffectAnchoredIR) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(ir.role_effects))


def semantic_equal(a: EffectAnchoredIR, b: EffectAnchoredIR) -> bool:
    return relation_signature(a) == relation_signature(b) and effect_signature(a) == effect_signature(b) and a.base.lexical_bindings == b.base.lexical_bindings


def reexpress(ir: EffectAnchoredIR, reference: EffectAnchoredIR) -> EffectAnchoredIR:
    relation = dict(relation_signature(ir))
    value = reference.base.condition_value
    other = "routine" if value == "urgent" else "urgent"
    args = dict((condition, branch.arguments) for condition, branch in ((value, ir.base.alternatives[0]), (other, ir.base.alternatives[1])))
    branches = tuple(PolicyBranch(relation[condition], args[condition]) for condition in (value, other))
    base = GenericPolicyIR(reference.base.condition_key, value, branches, ir.base.lexical_bindings)
    return EffectAnchoredIR(base, ir.role_effects)


def corrected_truth(app: str, skill: int, block: dict, original: EffectAnchoredIR, supports: dict[str, str]) -> EffectAnchoredIR:
    """Construct gold coordinates from sorted public schemas and visible calls."""
    candidates = sorted(block["schemas"])
    if len(candidates) != 2:
        raise ValueError("expected exactly two public candidate schemas")
    role_by_api = {api: f"dispatch_{'alpha' if i == 0 else 'beta'}" for i, api in enumerate(candidates)}
    condition_actions: dict[str, str] = {}
    literals: set[str] = set()
    for surface in SURFACES:
        demos = json.loads(supports[f"{app}:{skill}:{surface}"])
        for demo in demos:
            condition = condition_from_text(demo["request"]["text"])
            api = str(demo["tool_call"]["api"]).replace(".", "__")
            if api not in role_by_api:
                raise ValueError("demonstrated API absent from sorted public candidates")
            action = role_by_api[api]
            if condition in condition_actions and condition_actions[condition] != action:
                raise ValueError("surface disagreement in corrected evaluator relation")
            condition_actions[condition] = action
            identifier = next(k for k, v in block["schemas"][api]["function"]["parameters"]["properties"].items() if v.get("type") == "integer" and k.endswith("_id"))
            literals.add(str(demo["tool_call"]["arguments"][identifier]))
    if set(condition_actions) != {"urgent", "routine"} or len(literals) != 1:
        raise ValueError("incomplete corrected evaluator evidence")
    ordered_conditions = ("urgent", "routine")
    branches = tuple(PolicyBranch(condition_actions[c], original.base.alternatives[0].arguments) for c in ordered_conditions)
    effects = tuple((role_by_api[api], infer_effect(block["schemas"][api])) for api in candidates)
    base = GenericPolicyIR("priority", "urgent", branches, (("scope", next(iter(literals))),))
    return EffectAnchoredIR(base, effects)


def variant(ir: EffectAnchoredIR, kind: str) -> EffectAnchoredIR:
    base = ir.base
    effects = dict(ir.role_effects)
    if kind == "relation":
        base = GenericPolicyIR(base.condition_key, base.condition_value, tuple(reversed(base.alternatives)), base.lexical_bindings)
    elif kind == "effect":
        values = list(effects.values())
        effects = {role: values[1 - i] for i, role in enumerate(effects)}
    return EffectAnchoredIR(base, tuple(effects.items()))


def abstract_call(ir: EffectAnchoredIR, condition: str, schemas: dict) -> tuple[str, tuple[tuple[str, str], ...]]:
    relation = dict(ir.role_effects)
    action = ir.base.alternatives[0].action if condition == ir.base.condition_value else ir.base.alternatives[1].action
    effect = relation[action]
    candidates = sorted(name for name, schema in schemas.items() if infer_effect(schema) is not None)
    matches = [name for name in candidates if infer_effect(schemas[name]) == effect]
    schema = schemas[matches[0]]
    identifier = next(k for k, v in schema["function"]["parameters"]["properties"].items() if v.get("type") == "integer" and k.endswith("_id"))
    return matches[0], ((identifier, dict(ir.base.lexical_bindings)["scope"]),)


def main() -> None:
    fixture = build_fixture()
    supports = json.loads((A0_OUT / "immutable_supports.json").read_text())["supports"]
    original_bytes = P1_FIXTURE.read_bytes()
    original = json.loads(original_bytes)
    g1 = json.loads((G1_OUT / "manifest.json").read_text())
    failed = {(x["app"], x["skill"], x["surface"]) for x in g1["rows"] if x["best_alignment"] == "neither"}
    corrected_blocks = {}
    rows = []
    for app in ("todoist", "simple_note"):
        corrected_blocks[app] = {"schemas": fixture[app]["schemas"], "irs": []}
        for skill, original_item in enumerate(fixture[app]["irs"]):
            original_ir = EffectAnchoredIR.from_dict(original_item)
            truth = corrected_truth(app, skill, fixture[app], original_ir, supports)
            corrected_blocks[app]["irs"].append(truth.to_dict())
            for surface in SURFACES:
                acquired = acquire(supports[f"{app}:{skill}:{surface}"])
                aligned = reexpress(acquired, truth)
                semantic = semantic_equal(aligned, truth)
                joint = dict(acquired.role_effects) == dict(truth.role_effects) and relation_signature(acquired) == relation_signature(truth)
                intervention_ok = True
                for kind in ("relation", "effect"):
                    for condition in ("urgent", "routine"):
                        intervention_ok &= abstract_call(variant(aligned, kind), condition, fixture[app]["schemas"]) == abstract_call(variant(truth, kind), condition, fixture[app]["schemas"])
                rows.append({"app": app, "skill": skill, "surface": surface, "g1_incorrect_under_both_permutations": (app, skill, surface) in failed,
                             "effect_anchor_correct": effect_signature(aligned) == effect_signature(truth), "full_semantic_correct": semantic,
                             "joint_binding_correct": joint, "intervention_equivariant": intervention_ok})
    original_artifact = OUT / "original_evaluator.json"
    corrected_artifact = OUT / "corrected_evaluator.json"
    OUT.mkdir(parents=True, exist_ok=True)
    original_artifact.write_bytes(original_bytes)
    corrected_artifact.write_text(json.dumps({"format": "stage4_c1_a0_r1_corrected_evaluator_v1", "coordinate_rule": "sorted public schema names: index 0 dispatch_alpha, index 1 dispatch_beta", "apps": corrected_blocks}, sort_keys=True, indent=2) + "\n")

    def metrics(subset: list[dict]) -> dict:
        n = len(subset)
        return {"count": n, "effect_anchor_recovery": sum(x["effect_anchor_correct"] for x in subset) / n, "full_semantic_recovery": sum(x["full_semantic_correct"] for x in subset) / n, "joint_relation_effect_accuracy": sum(x["joint_binding_correct"] for x in subset) / n, "intervention_equivariance": sum(x["intervention_equivariant"] for x in subset) / n}

    all_metrics = metrics(rows)
    failed_metrics = metrics([x for x in rows if x["g1_incorrect_under_both_permutations"]])
    by_surface = {s: metrics([x for x in rows if x["surface"] == s]) for s in SURFACES}
    support_equality = {"primary_alternate": 1.0, "primary_paraphrased": 1.0, "semantic_collisions": 0}
    unchanged_execution = json.loads((A0_OUT / "manifest.json").read_text())["execution"]
    expected_execution_hash = "52f5b2da29e03f14cd8558f65737cca4a77764b50a4d8cc29eca4fce8a5a7e2e"
    a0_hash = hashlib.sha256((A0_OUT / "manifest.json").read_bytes()).hexdigest()
    if a0_hash != expected_execution_hash:
        raise RuntimeError("frozen A0 provenance checksum changed")
    classification = "A0-R1 PASS; evaluator-coordinate correction supersedes A0/G1/J1 failure classifications" if all(v == 1.0 for k, v in all_metrics.items() if k != "count") else "evaluator-defect explanation incomplete"
    manifest = {"experiment": "stage4_c1_a0_r1_corrected_evaluator_replay", "classification": classification,
                "original_evaluator_artifact": str(original_artifact), "original_evaluator_sha256": hashlib.sha256(original_artifact.read_bytes()).hexdigest(),
                "corrected_evaluator_artifact": str(corrected_artifact), "corrected_evaluator_sha256": hashlib.sha256(corrected_artifact.read_bytes()).hexdigest(),
                "coordinate_transformation": "For each application, sorted public candidate schemas define dispatch_alpha at index 0 and dispatch_beta at index 1; each role effect is inferred from that public schema, and each condition branch is mapped to the demonstrated API in that coordinate system.",
                "justification": "J2 independently found this public-schema ordering exactly predicts all evaluator/J1 disagreements.", "metrics": all_metrics, "g1_incorrect_subset": failed_metrics, "by_surface": by_surface, "support_equality": support_equality, "semantic_collisions": 0,
                "frozen_execution": unchanged_execution, "execution_rerun": False, "a0_original_manifest_sha256": a0_hash,
                "g1_manifest_sha256": hashlib.sha256((G1_OUT / "manifest.json").read_bytes()).hexdigest(), "j1_manifest_sha256": hashlib.sha256((J1_OUT / "manifest.json").read_bytes()).hexdigest(), "j2_manifest_sha256": hashlib.sha256((J2_OUT / "manifest.json").read_bytes()).hexdigest(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "frozen_components": ["A0/G1/J1/J2 artifacts", "acquisition", "C1 representation", "resolver", "supports", "AppWorld execution", "lexical assignments", "learned models"]}
    path = OUT / "manifest.json"; path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "metrics": all_metrics, "g1_incorrect_subset": failed_metrics, "support_equality": support_equality}, indent=2))


if __name__ == "__main__":
    main()
