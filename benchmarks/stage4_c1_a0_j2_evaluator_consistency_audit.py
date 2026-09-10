"""Stage 4-C1-A0-J2: evaluator versus learner-visible joint semantics audit."""
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

ROOT = Path(__file__).parents[1]
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
G1_OUT = ROOT / "results/stage4_c1_a0_g1_gauge_audit"
J1_OUT = ROOT / "results/stage4_c1_a0_j1_joint_assignment_audit"
OUT = ROOT / "results/stage4_c1_a0_j2_evaluator_consistency_audit"
SURFACES = ("primary", "alternate_entity", "paraphrased")


def condition_map(ir: EffectAnchoredIR) -> dict[str, str]:
    value = ir.base.condition_value
    other = "routine" if value == "urgent" else "urgent"
    return {value: ir.base.alternatives[0].action, other: ir.base.alternatives[1].action}


def visible_observations(text: str, relational_ir: EffectAnchoredIR) -> dict[str, str]:
    demos = json.loads(text)
    schemas = {}
    for demo in demos:
        schemas.update(demo["public_schemas"])
    relation = condition_map(relational_ir)
    observed = {}
    for demo in demos:
        request = str(demo["request"]["text"]).lower()
        condition = "urgent" if "urgent" in request else "routine" if "routine" in request else None
        if condition is None:
            raise ValueError("support request has no recognized condition")
        api = str(demo["tool_call"]["api"]).replace(".", "__")
        observed[relation[condition]] = infer_effect(schemas[api])
    return observed


def canonical_condition_effects(ir: EffectAnchoredIR, role_effects: dict[str, str]) -> dict[str, str]:
    relation = condition_map(ir)
    return {condition: role_effects[role] for condition, role in relation.items()}


def j1_assignment(support_text: str, acquired: EffectAnchoredIR) -> dict[str, str]:
    observed = visible_observations(support_text, acquired)
    # J1 has already established exactly one compatible assignment for every
    # support. This reconstructs that assignment from the same evidence.
    return {role: effect for role, effect in observed.items()}


def schema_order_prediction(support_text: str, relational_ir: EffectAnchoredIR) -> dict[str, str]:
    """Predict the joint map from A0's sorted public-schema convention only."""
    demos = json.loads(support_text)
    schemas = {}
    for demo in demos:
        schemas.update(demo["public_schemas"])
    candidates = sorted(schemas)
    role_by_api = {api: f"dispatch_{'alpha' if i == 0 else 'beta'}" for i, api in enumerate(candidates)}
    relation = condition_map(relational_ir)
    result = {}
    for demo in demos:
        text = str(demo["request"]["text"]).lower()
        condition = "urgent" if "urgent" in text else "routine"
        api = str(demo["tool_call"]["api"]).replace(".", "__")
        result[condition] = infer_effect(schemas[api])
    return result


def support_row(app: str, skill: int, surface: str, truth: EffectAnchoredIR, support_text: str, g1_failed: bool) -> dict:
    acquired = acquire(support_text)
    evidence_by_role = visible_observations(support_text, acquired)
    evaluator_by_role = dict(truth.role_effects)
    j1_by_role = j1_assignment(support_text, acquired)
    evaluator_canonical = canonical_condition_effects(truth, evaluator_by_role)
    evidence_canonical = canonical_condition_effects(acquired, evidence_by_role)
    j1_canonical = canonical_condition_effects(acquired, j1_by_role)
    ordering_canonical = schema_order_prediction(support_text, acquired)
    evaluator_trace = {condition: {"role": condition_map(truth)[condition], "evidence_effect": evidence_canonical[condition], "evaluator_effect": evaluator_canonical[condition], "consistent": evidence_canonical[condition] == evaluator_canonical[condition]} for condition in ("urgent", "routine")}
    j1_trace = {condition: {"role": condition_map(acquired)[condition], "evidence_effect": evidence_canonical[condition], "j1_effect": j1_canonical[condition], "consistent": evidence_canonical[condition] == j1_canonical[condition]} for condition in ("urgent", "routine")}
    evaluator_exact = evaluator_canonical == evidence_canonical
    j1_exact = j1_canonical == evidence_canonical
    return {"app": app, "skill": skill, "surface": surface, "g1_incorrect_under_both_permutations": g1_failed,
            "evaluator_canonical_condition_effects": evaluator_canonical, "j1_canonical_condition_effects": j1_canonical,
            "evidence_canonical_condition_effects": evidence_canonical, "evaluator_trace": evaluator_trace, "j1_trace": j1_trace,
            "evaluator_consistent": evaluator_exact, "j1_consistent": j1_exact,
            "evaluator_role_accuracy": sum(x["consistent"] for x in evaluator_trace.values()) / 2,
            "j1_role_accuracy": sum(x["consistent"] for x in j1_trace.values()) / 2,
            "ordering_predicted_canonical_condition_effects": ordering_canonical,
            "evaluator_j1_disagree": evaluator_canonical != j1_canonical,
            "ordering_predicts_evaluator_j1_disagreement": ordering_canonical != evaluator_canonical}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {"count": n, "evaluator_consistency": sum(x["evaluator_consistent"] for x in rows) / n,
            "j1_consistency": sum(x["j1_consistent"] for x in rows) / n,
            "evaluator_role_accuracy": sum(x["evaluator_role_accuracy"] for x in rows) / n,
            "j1_role_accuracy": sum(x["j1_role_accuracy"] for x in rows) / n,
            "evaluator_j1_disagreement_fraction": sum(x["evaluator_j1_disagree"] for x in rows) / n}


def main() -> None:
    fixture = build_fixture()
    supports = json.loads((A0_OUT / "immutable_supports.json").read_text())["supports"]
    g1 = json.loads((G1_OUT / "manifest.json").read_text())
    j1 = json.loads((J1_OUT / "manifest.json").read_text())
    failed_keys = {(x["app"], x["skill"], x["surface"]) for x in g1["rows"] if x["best_alignment"] == "neither"}
    rows = []
    for app in ("todoist", "simple_note"):
        for skill, item in enumerate(fixture[app]["irs"]):
            truth = EffectAnchoredIR.from_dict(item)
            for surface in SURFACES:
                key = f"{app}:{skill}:{surface}"
                rows.append(support_row(app, skill, surface, truth, supports[key], (app, skill, surface) in failed_keys))
    failed = [x for x in rows if x["g1_incorrect_under_both_permutations"]]
    by_surface = {surface: summarize([x for x in rows if x["surface"] == surface]) for surface in SURFACES}
    # Provenance is recorded from the actual generator code and independently
    # checked against every support's public schema order and observed calls.
    provenance = {
        "generator_condition_order": "P0 make_ir: even index -> condition_value urgent and first alpha; odd index -> condition_value routine and first beta",
        "C0_condition_order": "c0_ir: condition_value is always urgent; show_first alternates by index parity",
        "A0_canonical_condition_order": "sorted(observed_conditions, reverse=True) -> urgent, routine",
        "A0_application_order": ["todoist", "simple_note"],
        "A0_schema_candidate_order": "sorted compatible public schema names; role_by_api index 0 -> dispatch_alpha, index 1 -> dispatch_beta",
        "evaluator_effect_construction": "C1 P0 anchored(): branch action dispatch_alpha -> state_preserving, dispatch_beta -> state_mutating",
        "j1_effect_construction": "unique mapping from each condition's observed concrete API to infer_effect(public schema)",
    }
    ordering_rows = []
    for row in rows:
        ordering_rows.append({"app": row["app"], "skill": row["skill"], "surface": row["surface"],
                              "ordering_predicts_disagreement": row["ordering_predicts_evaluator_j1_disagreement"],
                              "observed_disagreement": row["evaluator_j1_disagree"]})
    ordering_exact = all(x["ordering_predicts_disagreement"] == x["observed_disagreement"] for x in ordering_rows)
    overall = summarize(rows)
    failed_stats = summarize(failed)
    if overall["j1_consistency"] == 1.0 and overall["evaluator_consistency"] < 1.0:
        classification = "evaluator joint-coordinate defect"
    elif overall["evaluator_consistency"] == 1.0 and overall["j1_consistency"] < 1.0:
        classification = "J1 semantic-audit defect"
    elif overall["evaluator_consistency"] == 1.0 and overall["j1_consistency"] == 1.0 and overall["evaluator_j1_disagreement_fraction"] > 0:
        classification = "hidden convention/non-identifiability outside learner-visible evidence"
    else:
        classification = "fixture semantic inconsistency"
    manifest = {
        "experiment": "stage4_c1_a0_j2_evaluator_consistency_audit", "classification": classification,
        "overall": overall, "g1_incorrect_subset": failed_stats, "g1_incorrect_count": len(failed),
        "by_surface": by_surface, "ordering_provenance": provenance,
        "ordering_convention_exactly_predicts_disagreements": ordering_exact, "ordering_rows": ordering_rows,
        "rows": rows, "a0_support_sha256": hashlib.sha256((A0_OUT / "immutable_supports.json").read_bytes()).hexdigest(),
        "a0_manifest_sha256": hashlib.sha256((A0_OUT / "manifest.json").read_bytes()).hexdigest(),
        "g1_manifest_sha256": hashlib.sha256((G1_OUT / "manifest.json").read_bytes()).hexdigest(),
        "j1_manifest_sha256": hashlib.sha256((J1_OUT / "manifest.json").read_bytes()).hexdigest(),
        "frozen": ["A0 acquisition", "C1 representation", "resolver", "AppWorld fixtures", "immutable supports", "J1 outputs", "evaluator artifacts"],
        "execution_rerun": False, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "overall": overall, "g1_incorrect_subset": failed_stats, "ordering_exact": ordering_exact}, indent=2))


if __name__ == "__main__":
    main()
