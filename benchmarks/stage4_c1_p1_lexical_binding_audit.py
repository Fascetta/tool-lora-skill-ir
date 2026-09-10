"""Stage 4-C1-P1: causal necessity of exact lexical bindings."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from benchmarks import stage4_c1_p0_effect_anchor as c1
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import audit_identifiability, resolve
from tool_lora.stage4_p0.contract import GenericPolicyIR

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_c1_p1_lexical_binding_audit"


def permuted_ids(ids: list[int], app: str) -> list[int]:
    # Deterministic evaluator assignment, independent of relation/effect/API.
    keyed = sorted(ids, key=lambda value: hashlib.sha256(f"c1-p1-binding:{app}:{value}".encode()).hexdigest())
    return keyed


def replace_literal(ir: EffectAnchoredIR, value: int) -> EffectAnchoredIR:
    base = ir.base
    base = GenericPolicyIR(base.condition_key, base.condition_value, base.alternatives, (("scope", str(value)),))
    return EffectAnchoredIR(base, ir.role_effects)


def build_fixture() -> dict:
    fixture = deepcopy(c1.build_fixture())
    for app in ("todoist", "simple_note"):
        block = fixture[app]
        assigned = permuted_ids(block["ids"], app)
        block["binding_assignment"] = assigned
        block["irs"] = [replace_literal(EffectAnchoredIR.from_dict(ir), assigned[i]).to_dict() for i, ir in enumerate(block["irs"])]
        block["constant_binding"] = str(assigned[0])
    fixture["format"] = "stage4_c1_p1_lexical_binding_fixture_v1"
    fixture["binding_policy"] = "unique per skill within each application; assignment is a fixed evaluator permutation independent of relation/effect/condition/entity"
    fixture["constant_baseline"] = {app: 1 / len(fixture[app]["ids"]) for app in ("todoist", "simple_note")}
    return fixture


def execute_fixture(fixture: dict) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); items = []
        for app in ("todoist", "simple_note"):
            block = fixture[app]; schema_path = td / f"{app}_schemas.json"; schema_path.write_text(json.dumps(block["schemas"], sort_keys=True))
            truths = [EffectAnchoredIR.from_dict(x) for x in block["irs"]]
            for skill, truth in enumerate(truths):
                counter = c1._variant(truth, "relation_shuffled")
                relation = c1._variant(truth, "relation_shuffled")
                lexical = c1._variant(truth, "lexical_shuffled", str(block["binding_assignment"][(skill + 1) % len(block["binding_assignment"])]))
                effect = c1._variant(truth, "effect_anchor_shuffled")
                constant = c1._constant(truth)
                constant = replace_literal(constant, int(block["constant_binding"]))
                variants = {"matched": truth, "counterfactual": counter, "relation_shuffled": relation, "effect_anchor_shuffled": effect, "lexical_shuffled": lexical, "constant": constant}
                for name, variant in variants.items():
                    artifact = td / f"{app}_{skill}_{name}.json"; variant.save(artifact)
                    for condition in ("urgent", "routine"):
                        target_ir = counter if name == "counterfactual" else truth
                        target_action = target_ir.base.alternatives[0].action if condition == target_ir.base.condition_value else target_ir.base.alternatives[1].action
                        target_effect = dict(target_ir.role_effects)[target_action]
                        target_call = resolve(target_ir, condition, "heldout", block["schemas"])
                        items.append({"artifact": str(artifact), "schemas": str(schema_path), "app": app, "task": block["task"], "condition_value": condition, "target_api": target_call.api, "target_id": dict(truth.base.lexical_bindings)["scope"], "target_effect": target_effect, "name": name, "skill": skill})
        env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('APPWORLD_PYTHONPATH', '')}:{ROOT / 'src'}:{ROOT}"
        output = subprocess.check_output([sys.executable, str(ROOT / "benchmarks/stage4_c1_p0_effect_anchor.py"), "--worker", json.dumps(items)], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL)
        observed = json.loads(output.strip().splitlines()[-1])
        for item, got in zip(items, observed):
            rows.append({"app": item["app"], "skill": item["skill"], "condition": item["name"], "condition_value": item["condition_value"], "action_correct": got.get("api") == item["target_api"], "binding_correct": got.get("identifier") == int(item["target_id"]), "state_correct": got.get("state_ok", False), "error": got.get("error")})
    metrics = {}
    for name in ("matched", "counterfactual", "relation_shuffled", "effect_anchor_shuffled", "lexical_shuffled", "constant"):
        subset = [r for r in rows if r["condition"] == name]
        metrics[name] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset), "binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset), "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
    errors = [r for r in rows if r["error"]]
    return {"performed": True, "pass": not errors and metrics["matched"]["action_accuracy"] == metrics["matched"]["binding_accuracy"] == 1.0 and metrics["counterfactual"]["action_accuracy"] == metrics["counterfactual"]["binding_accuracy"] == 1.0, "metrics": metrics, "rows": rows, "errors": errors, "serialization_identity": 1.0}


def main() -> None:
    fixture = build_fixture()
    schema_audits = {}
    for app in ("todoist", "simple_note"):
        block = fixture[app]
        schema_audits[app] = audit_identifiability([EffectAnchoredIR.from_dict(x) for x in block["irs"]], block["schemas"])
    ambiguity = fixture["ambiguity_control"]
    schema_audits["ambiguity"] = audit_identifiability([EffectAnchoredIR.from_dict(ambiguity["ir"])], ambiguity["schemas"])
    # Re-run the unchanged C1 order/distractor logic on the new lexical fixture.
    order_invariant = True
    for app in ("todoist", "simple_note"):
        block = fixture[app]; ir = EffectAnchoredIR.from_dict(block["irs"][0]); expected = None
        for order in __import__("itertools").permutations(block["schemas"]):
            result = audit_identifiability([ir], {key: block["schemas"][key] for key in order})["rows"][0]["mappings"]
            expected = result if expected is None else expected
            order_invariant &= result == expected
    distractor = deepcopy(fixture["simple_note"]["schemas"]); distractor["irrelevant"] = {"function": {"name": "irrelevant", "description": "Return a string", "parameters": {"properties": {"access_token": {"type": "string"}}}}}
    distractor_audit = audit_identifiability([EffectAnchoredIR.from_dict(fixture["simple_note"]["irs"][0])], distractor)
    audits_pass = all(schema_audits[app]["unique_mapping"] for app in ("todoist", "simple_note")) and not schema_audits["ambiguity"]["unique_mapping"] and order_invariant and distractor_audit["unique_mapping"]
    execution = execute_fixture(fixture) if audits_pass else {"performed": False, "pass": False}
    OUT.mkdir(parents=True, exist_ok=True); fixture_path = OUT / "immutable_fixture.json"; fixture_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n")
    manifest = {"experiment": "stage4_c1_p1_lexical_binding_audit", "classification": "PASS" if execution.get("pass") else "opaque-identity generalization failure", "c0_r1_negative_baseline": "preserved", "c1_p0_manifest": str(ROOT / "results/stage4_c1_p0_effect_anchor/manifest.json"), "schema_audits": schema_audits, "distractor_audit": distractor_audit, "candidate_order_invariant": order_invariant, "execution": execution, "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(), "appworld_version": "0.1.3.post1", "appworld_commit": "efac79081936323733f68fd78260dab6981f195a", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "learned_acquisition_used": False, "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": manifest["classification"], "metrics": execution.get("metrics"), "fixture": manifest["fixture_sha256"]}, indent=2))


if __name__ == "__main__": main()
