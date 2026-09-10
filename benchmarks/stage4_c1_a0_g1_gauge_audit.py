"""Stage 4-C1-A0-G1: anonymous-role permutation/gauge audit."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from benchmarks import stage4_c1_p0_effect_anchor as c1
from tool_lora.stage4_c1_a0.acquisition import acquire
from benchmarks.stage4_c1_p1_lexical_binding_audit import build_fixture
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import resolve
from tool_lora.stage4_p0.contract import GenericPolicyIR, PolicyBranch

ROOT = Path(__file__).parents[1]
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
OUT = ROOT / "results/stage4_c1_a0_g1_gauge_audit"
SWAP = {"dispatch_alpha": "dispatch_beta", "dispatch_beta": "dispatch_alpha"}


def permute(ir: EffectAnchoredIR, swapped: bool) -> EffectAnchoredIR:
    if not swapped:
        return ir
    base = ir.base
    branches = tuple(PolicyBranch(SWAP[b.action], b.arguments) for b in base.alternatives)
    effects = tuple((SWAP[role], effect) for role, effect in ir.role_effects)
    return EffectAnchoredIR(GenericPolicyIR(base.condition_key, base.condition_value, branches, base.lexical_bindings), effects)


def effect_signature(ir: EffectAnchoredIR) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(ir.role_effects))


def relation_signature(ir: EffectAnchoredIR) -> tuple[tuple[str, str], ...]:
    value = ir.base.condition_value
    other = "routine" if value == "urgent" else "urgent"
    return tuple(sorted(((value, ir.base.alternatives[0].action),
                         (other, ir.base.alternatives[1].action))))


def relation_effect_equal(a: EffectAnchoredIR, b: EffectAnchoredIR) -> bool:
    return (relation_signature(a), effect_signature(a), a.base.lexical_bindings) == (relation_signature(b), effect_signature(b), b.base.lexical_bindings)


def reexpress_relation(ir: EffectAnchoredIR, reference: EffectAnchoredIR) -> EffectAnchoredIR:
    mapping = dict(relation_signature(ir))
    value = reference.base.condition_value
    other = "routine" if value == "urgent" else "urgent"
    branches = tuple(PolicyBranch(mapping[condition], args) for condition, args in (
        (value, ir.base.alternatives[0].arguments),
        (other, ir.base.alternatives[1].arguments),
    ))
    base = GenericPolicyIR(reference.base.condition_key, value, branches, ir.base.lexical_bindings)
    return EffectAnchoredIR(base, ir.role_effects)


def best_alignment(identity: bool, swapped: bool) -> str:
    if identity and swapped:
        return "tied"
    if identity:
        return "identity"
    if swapped:
        return "swap"
    return "neither"


def main() -> None:
    fixture = build_fixture()
    support_data = json.loads((A0_OUT / "immutable_supports.json").read_text())
    a0_manifest = json.loads((A0_OUT / "manifest.json").read_text())
    supports = support_data["supports"]
    rows = []
    canonical = []
    equivariance_rows = []
    for app in ("todoist", "simple_note"):
        block = fixture[app]
        truths = [EffectAnchoredIR.from_dict(x) for x in block["irs"]]
        for skill, truth in enumerate(truths):
            aligned = {}
            for surface in ("primary", "alternate_entity", "paraphrased"):
                acquired = acquire(supports[f"{app}:{skill}:{surface}"])
                identity = relation_effect_equal(acquired, truth)
                swapped = relation_effect_equal(permute(acquired, True), truth)
                identity_effect = effect_signature(acquired) == effect_signature(truth)
                swapped_effect = effect_signature(permute(acquired, True)) == effect_signature(truth)
                chosen = acquired if identity else permute(acquired, True) if swapped else acquired
                aligned[surface] = (reexpress_relation(chosen, truth), identity, swapped)
                rows.append({"app": app, "skill": skill, "surface": surface,
                             "raw_effect_anchor_recovery": identity_effect,
                             "identity_effect_recovery": identity_effect,
                             "swapped_effect_recovery": swapped_effect,
                             "identity_full_recovery": identity,
                             "swapped_full_recovery": swapped,
                             "best_alignment": best_alignment(identity, swapped)})
            canonical.append({"app": app, "skill": skill,
                              "primary_alternate_equivalent": aligned["primary"][0] == aligned["alternate_entity"][0],
                              "primary_paraphrased_equivalent": aligned["primary"][0] == aligned["paraphrased"][0],
                              "alignments": {key: best_alignment(value[1], value[2]) for key, value in aligned.items()}})
            for surface, (artifact, _, _) in aligned.items():
                schemas = block["schemas"]
                for intervention in ("relation", "effect"):
                    truth_variant = c1._variant(truth, "relation_shuffled" if intervention == "relation" else "effect_anchor_shuffled")
                    artifact_variant = c1._variant(artifact, "relation_shuffled" if intervention == "relation" else "effect_anchor_shuffled")
                    for condition in ("urgent", "routine"):
                        truth_call = resolve(truth_variant, condition, "heldout", schemas)
                        artifact_call = resolve(artifact_variant, condition, "heldout", schemas)
                        equivariance_rows.append({"app": app, "skill": skill, "surface": surface, "intervention": intervention, "condition": condition, "same_concrete_api": truth_call.api == artifact_call.api, "same_binding": dict(truth_call.arguments) == dict(artifact_call.arguments)})
    raw_effect = sum(row["raw_effect_anchor_recovery"] for row in rows) / len(rows)
    invariant_effect = sum(row["identity_effect_recovery"] or row["swapped_effect_recovery"] for row in rows) / len(rows)
    invariant_full = sum(row["identity_full_recovery"] or row["swapped_full_recovery"] for row in rows) / len(rows)
    swap_fraction = sum(row["best_alignment"] == "swap" for row in rows) / len(rows)
    ties = sum(row["best_alignment"] == "tied" for row in rows) / len(rows)
    manifest = {"experiment": "stage4_c1_a0_g1_gauge_audit", "classification": "semantic effect acquisition PASS with anonymous-coordinate gauge ambiguity" if invariant_effect == invariant_full == 1.0 and all(x["primary_alternate_equivalent"] and x["primary_paraphrased_equivalent"] for x in canonical) and all(x["same_concrete_api"] and x["same_binding"] for x in equivariance_rows) else "true effect-semantic acquisition failure", "a0_manifest": str(A0_OUT / "manifest.json"), "a0_support_sha256": a0_manifest["support_sha256"], "raw_effect_anchor_recovery": raw_effect, "permutation_invariant_effect_anchor_recovery": invariant_effect, "permutation_invariant_full_relational_effect_recovery": invariant_full, "fraction_requiring_swap": swap_fraction, "fraction_tied_best_alignment": ties, "canonical_equivalence": {"primary_alternate": sum(x["primary_alternate_equivalent"] for x in canonical) / len(canonical), "primary_paraphrased": sum(x["primary_paraphrased_equivalent"] for x in canonical) / len(canonical), "incompatible_support_alignments": sum(len(set(x["alignments"].values())) > 1 for x in canonical)}, "intervention_equivariance": sum(x["same_concrete_api"] and x["same_binding"] for x in equivariance_rows) / len(equivariance_rows), "rows": rows, "canonical_rows": canonical, "equivariance_rows": equivariance_rows, "predeclared_pass": {"permutation_invariant_effect": 1.0, "permutation_invariant_full_semantics": 1.0, "support_equivalence": 1.0, "intervention_equivariance": 1.0}, "frozen": ["C1-A0 acquisition output", "supports", "C1 contract", "resolver", "evaluator", "AppWorld fixtures", "balanced lexical identities"], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "learned_acquisition_used": False, "execution_rerun": False}
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / "manifest.json"; path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": manifest["classification"], "raw_effect": raw_effect, "invariant_effect": invariant_effect, "invariant_full": invariant_full, "swap_fraction": swap_fraction, "equivariance": manifest["intervention_equivariance"]}, indent=2))


if __name__ == "__main__":
    main()
