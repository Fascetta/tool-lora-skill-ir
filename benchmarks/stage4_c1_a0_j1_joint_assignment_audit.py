"""Stage 4-C1-A0-J1: learner-visible joint relation/effect assignment audit."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from itertools import permutations
from pathlib import Path

from benchmarks.stage4_c1_p1_lexical_binding_audit import build_fixture
from tool_lora.stage4_c1_a0.acquisition import acquire
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import infer_effect

ROOT = Path(__file__).parents[1]
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
G1_OUT = ROOT / "results/stage4_c1_a0_g1_gauge_audit"
OUT = ROOT / "results/stage4_c1_a0_j1_joint_assignment_audit"
SURFACES = ("primary", "alternate_entity", "paraphrased")
EFFECTS = ("state_preserving", "state_mutating")


def relation_map(ir: EffectAnchoredIR) -> dict[str, str]:
    value = ir.base.condition_value
    other = "routine" if value == "urgent" else "urgent"
    return {value: ir.base.alternatives[0].action, other: ir.base.alternatives[1].action}


def compatible_assignments(support_text: str, relational_ir: EffectAnchoredIR) -> list[dict[str, str]]:
    """Use only support-visible relation, schemas, calls, and effects."""
    demos = json.loads(support_text)
    role_for_condition = relation_map(relational_ir)
    schemas = {}
    for demo in demos:
        schemas.update(demo["public_schemas"])
    observations = []
    for demo in demos:
        text = str(demo["request"]["text"]).lower()
        condition = "urgent" if "urgent" in text else "routine" if "routine" in text else None
        if condition is None or condition not in role_for_condition:
            return []
        api = str(demo["tool_call"]["api"]).replace(".", "__")
        effect = infer_effect(schemas[api])
        if effect not in EFFECTS:
            return []
        observations.append((role_for_condition[condition], effect))
    roles = tuple(sorted(role_for_condition.values()))
    assignments = []
    for values in permutations(EFFECTS):
        mapping = dict(zip(roles, values))
        if all(mapping[role] == effect for role, effect in observations):
            assignments.append(mapping)
    return assignments


def same_mapping(a: dict[str, str], b: dict[str, str]) -> bool:
    return a == b


def main() -> None:
    fixture = build_fixture()
    supports = json.loads((A0_OUT / "immutable_supports.json").read_text())["supports"]
    g1 = json.loads((G1_OUT / "manifest.json").read_text())
    failed_keys = {(row["app"], row["skill"], row["surface"])
                   for row in g1["rows"] if row["best_alignment"] == "neither"}
    rows = []
    for app in ("todoist", "simple_note"):
        truths = [EffectAnchoredIR.from_dict(x) for x in fixture[app]["irs"]]
        for skill, truth in enumerate(truths):
            for surface in SURFACES:
                key = f"{app}:{skill}:{surface}"
                acquired = acquire(supports[key])
                assignments = compatible_assignments(supports[key], acquired)
                unique = assignments[0] if len(assignments) == 1 else None
                truth_mapping = dict(truth.role_effects)
                rows.append({
                    "app": app, "skill": skill, "surface": surface,
                    "g1_incorrect_under_both_permutations": (app, skill, surface) in failed_keys,
                    "compatible_assignment_count": len(assignments),
                    "compatible_assignments": assignments,
                    "assignment_class": "zero" if not assignments else "one" if len(assignments) == 1 else "two",
                    "unique_mapping_matches_evaluator_truth": unique is not None and same_mapping(unique, truth_mapping),
                    "a0_joint_binding_matches_evaluator_truth": dict(acquired.role_effects) == truth_mapping,
                })

    def stats(subset: list[dict]) -> dict:
        n = len(subset)
        counts = {name: sum(row["assignment_class"] == name for row in subset) / n for name in ("zero", "one", "two")}
        return {
            "count": n,
            "overall_identifiability": sum(row["compatible_assignment_count"] == 1 for row in subset) / n,
            "fraction_zero": counts["zero"], "fraction_one": counts["one"], "fraction_two": counts["two"],
            "unique_mapping_evaluator_accuracy": (sum(row["unique_mapping_matches_evaluator_truth"] for row in subset if row["assignment_class"] == "one") /
                                                    max(1, sum(row["assignment_class"] == "one" for row in subset))),
            "a0_joint_binding_accuracy_restricted_to_unique": (sum(row["a0_joint_binding_matches_evaluator_truth"] for row in subset if row["assignment_class"] == "one") /
                                                               max(1, sum(row["assignment_class"] == "one" for row in subset))),
        }

    by_surface = {surface: stats([r for r in rows if r["surface"] == surface]) for surface in SURFACES}
    failed = [r for r in rows if r["g1_incorrect_under_both_permutations"]]
    overall = stats(rows)
    failed_stats = stats(failed)
    equivalence_rows = []
    for app in ("todoist", "simple_note"):
        for skill in range(len(fixture[app]["irs"])):
            group = [r for r in rows if r["app"] == app and r["skill"] == skill]
            classes = [r["assignment_class"] for r in group]
            mappings = [json.dumps(r["compatible_assignments"][0], sort_keys=True) if r["assignment_class"] == "one" else json.dumps(r["compatible_assignments"], sort_keys=True) for r in group]
            equivalence_rows.append({"app": app, "skill": skill, "same_ambiguity_class": len(set(classes)) == 1, "same_joint_assignment_or_ambiguity": len(set(mappings)) == 1, "classes": classes, "mappings": mappings})

    zero = sum(r["assignment_class"] == "zero" for r in rows)
    two = sum(r["assignment_class"] == "two" for r in rows)
    if zero:
        classification = "support/effect-model inconsistency"
    elif two:
        classification = "joint semantic non-identifiability"
    elif overall["a0_joint_binding_accuracy_restricted_to_unique"] < 1.0:
        classification = "deterministic relation-effect binding acquisition failure"
    else:
        classification = "PASS"
    support_path = A0_OUT / "immutable_supports.json"
    manifest = {
        "experiment": "stage4_c1_a0_j1_joint_assignment_audit",
        "classification": classification,
        "a0_manifest_sha256": hashlib.sha256((A0_OUT / "manifest.json").read_bytes()).hexdigest(),
        "g1_manifest_sha256": hashlib.sha256((G1_OUT / "manifest.json").read_bytes()).hexdigest(),
        "a0_support_sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(),
        "overall": overall, "g1_incorrect_subset": failed_stats, "g1_incorrect_count": len(failed),
        "by_surface": by_surface, "support_equivalence": equivalence_rows, "rows": rows,
        "compatibility_evidence": ["request condition text", "public schemas", "observed concrete calls and arguments", "inferred public schema effects", "observable results/state not used as hidden labels"],
        "excluded_from_compatibility": ["evaluator role labels", "gold role-effect mappings", "gold C1 artifacts", "AppWorld execution rerun"],
        "frozen": ["A0 supports", "A0 acquisition", "C1 representation", "resolver", "evaluator", "AppWorld fixtures", "G1 output"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "execution_rerun": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "overall": overall, "g1_incorrect_subset": failed_stats, "by_surface": by_surface}, indent=2))


if __name__ == "__main__":
    main()
