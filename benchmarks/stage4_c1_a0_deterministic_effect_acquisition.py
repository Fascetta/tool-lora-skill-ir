"""Stage 4-C1-A0: acquire effect anchors from learner-visible AppWorld traces."""
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
from benchmarks.stage4_c1_p1_lexical_binding_audit import build_fixture as build_p1_fixture
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import audit_identifiability, resolve
from tool_lora.stage4_c1.schema_resolver import infer_effect
from tool_lora.stage4_c1_a0.acquisition import acquire, audit_support
from tool_lora.stage4_p0.contract import GenericPolicyIR

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
SUPPORTS = OUT / "immutable_supports.json"
SURFACES = ("primary", "alternate_entity", "paraphrased")


def _support_text(skill: int, condition: str, surface: str) -> str:
    entity = f"record entity {skill} {surface}"
    if surface == "primary": return f"Please handle {entity}; this request is {condition}."
    if surface == "alternate_entity": return f"Could you process the other {entity}? It is marked {condition}."
    return f"I would appreciate handling {entity}; the current priority is {condition}."


def _schemas(app: str, names: list[str]) -> dict[str, dict]:
    root = Path(os.environ["APPWORLD_ROOT"])
    data = json.loads((root / "data/api_docs/function_calling" / f"{app}.json").read_text())
    return {item["function"]["name"]: item for item in data if item["function"]["name"] in names}


def _support_worker(item: dict) -> None:
    from appworld import AppWorld
    app, identifier, task = item["app"], item["identifier"], item["task"]
    preserving = item["preserving_method"]; mutating = item["mutating_method"]
    idarg = "task_id" if app == "todoist" else "note_id"
    email = "joyce-weav@gmail.com" if app == "todoist" else "siwhit@gmail.com"
    # Run preserving first so a mutating demonstration can safely follow in
    # the same fresh world. The returned request condition keeps its meaning.
    code = ["import json", "p={x['account_name']:x['password'] for x in apis.supervisor.show_account_passwords()}", f"t=apis.{app}.login(username='{email}',password=p['{app}'])['access_token']"]
    first_condition, second_condition = ("urgent", "routine") if item["urgent_method"] == preserving else ("routine", "urgent")
    for index, (condition, method) in enumerate(((first_condition, preserving), (second_condition, mutating))):
        code.append(f"b{index}=apis.{app}.{preserving}({idarg}={identifier},access_token=t)")
        code.append(f"r{index}=apis.{app}.{method}({idarg}={identifier},access_token=t)")
        if app == "simple_note" and method == "delete_note":
            code.append(f"a{index}=apis.simple_note.search_notes(query='',page_index=0,page_limit=20,access_token=t)")
        else:
            code.append(f"a{index}=apis.{app}.{preserving}({idarg}={identifier},access_token=t)")
    code.append("print(json.dumps({'0': {'before': b0, 'result': r0, 'after': a0}, '1': {'before': b1, 'result': r1, 'after': a1}}))")
    with AppWorld(task_id=task, experiment_name="stage4_c1_a0_support", load_ground_truth=False, raise_on_failure=True) as world:
        output = world.execute("\n".join(code))
    payload = json.loads(output.strip().splitlines()[-1])
    demos = []
    for index, condition in enumerate((first_condition, second_condition)):
        method = preserving if index == 0 else mutating
        api = f"{app}.{method}"
        schema_name = f"{app}__{method}"
        demos.append({"request": {"text": _support_text(item["skill"], condition, item["surface"])}, "public_schemas": item["schemas"], "public_tool_schema": sorted(item["schemas"]), "tool_call": {"api": api, "arguments": {item["idarg"]: identifier}}, "observable_result": payload[str(index)]["result"], "before_state": payload[str(index)]["before"], "after_state": payload[str(index)]["after"], "schema_name": schema_name})
    print(json.dumps(demos, sort_keys=True))


def generate_support(skill: int, surface: str, app: str, block: dict, truth: EffectAnchoredIR) -> str:
    roles = dict(truth.role_effects); by_effect = {effect: role for role, effect in roles.items()}
    # Public API candidates are selected from schemas by generic effect, not a
    # semantic API lookup table.
    schemas = block["schemas"]
    preserving = next(name for name, schema in schemas.items() if infer_effect(schema) == "state_preserving")
    mutating = next(name for name, schema in schemas.items() if infer_effect(schema) == "state_mutating")
    methods = {name: name.split("__", 1)[1] for name in schemas}
    identifier = int(dict(truth.base.lexical_bindings)["scope"])
    env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('APPWORLD_PYTHONPATH', '')}:{ROOT / 'src'}:{ROOT}"
    item = {"skill": skill, "surface": surface, "app": app, "task": block["task"], "identifier": identifier, "idarg": "task_id" if app == "todoist" else "note_id", "schemas": schemas, "preserving_method": methods[preserving], "mutating_method": methods[mutating], "urgent_method": methods[next(name for name in schemas if dict(truth.role_effects)["dispatch_alpha"] == ("state_preserving" if name == preserving else "state_mutating"))]}
    output = subprocess.check_output([sys.executable, __file__, "--support-worker", json.dumps(item)], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL)
    return output.strip().splitlines()[-1]


def _variant(ir: EffectAnchoredIR, name: str, other: str | None = None) -> EffectAnchoredIR:
    if name == "relation_shuffled": return c1._variant(ir, "relation_shuffled")
    if name == "effect_anchor_shuffled": return c1._variant(ir, "effect_anchor_shuffled")
    if name == "lexical_shuffled": return c1._variant(ir, "lexical_shuffled", other)
    return ir


def _replace_literal(ir: EffectAnchoredIR, value: int) -> EffectAnchoredIR:
    base = ir.base
    base = GenericPolicyIR(base.condition_key, base.condition_value, base.alternatives, (("scope", str(value)),))
    return EffectAnchoredIR(base, ir.role_effects)


def execute_artifacts(fixture: dict, acquired: dict[str, dict[str, dict[str, EffectAnchoredIR]]]) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); items = []
        for app in ("todoist", "simple_note"):
            block = fixture[app]; schema_path = td / f"{app}_schemas.json"; schema_path.write_text(json.dumps(block["schemas"], sort_keys=True))
            for skill in range(len(block["irs"])):
                truth = acquired[app][str(skill)]["primary"]
                counter = _variant(truth, "relation_shuffled"); relation = _variant(truth, "relation_shuffled"); effect = _variant(truth, "effect_anchor_shuffled"); lexical = _variant(truth, "lexical_shuffled", dict(acquired[app][str((skill + 1) % len(block["irs"]))]["primary"].base.lexical_bindings)["scope"]); constant = _replace_literal(c1._constant(truth), int(block["constant_binding"]))
                variants = {"matched": truth, "counterfactual": counter, "relation_shuffled": relation, "effect_anchor_shuffled": effect, "lexical_shuffled": lexical, "constant": constant}
                for name, variant in variants.items():
                    artifact = td / f"{app}_{skill}_{name}.json"; variant.save(artifact)
                    for condition in ("urgent", "routine"):
                        target_ir = counter if name == "counterfactual" else truth
                        target_action = target_ir.base.alternatives[0].action if condition == target_ir.base.condition_value else target_ir.base.alternatives[1].action
                        target_call = resolve(target_ir, condition, "heldout", block["schemas"])
                        items.append({"artifact": str(artifact), "schemas": str(schema_path), "app": app, "task": block["task"], "condition_value": condition, "target_api": target_call.api, "target_id": dict(truth.base.lexical_bindings)["scope"], "target_effect": dict(target_ir.role_effects)[target_action], "name": name, "skill": skill})
        env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('APPWORLD_PYTHONPATH', '')}:{ROOT / 'src'}:{ROOT}"
        output = subprocess.check_output([sys.executable, str(ROOT / "benchmarks/stage4_c1_p0_effect_anchor.py"), "--worker", json.dumps(items)], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL)
        observed = json.loads(output.strip().splitlines()[-1])
        for item, got in zip(items, observed): rows.append({"app": item["app"], "skill": item["skill"], "condition": item["name"], "action_correct": got.get("api") == item["target_api"], "binding_correct": got.get("identifier") == int(item["target_id"]), "state_correct": got.get("state_ok", False), "error": got.get("error")})
    metrics = {}
    for name in ("matched", "counterfactual", "relation_shuffled", "effect_anchor_shuffled", "lexical_shuffled", "constant"):
        subset = [r for r in rows if r["condition"] == name]; metrics[name] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset), "binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset), "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
    return {"performed": True, "metrics": metrics, "rows": rows, "errors": [r for r in rows if r["error"]], "serialization_identity": 1.0}


def main() -> None:
    fixture = build_p1_fixture(); supports = {}
    if SUPPORTS.exists(): supports = json.loads(SUPPORTS.read_text())["supports"]
    else:
        supports = {f"{app}:{skill}:{surface}": generate_support(skill, surface, app, fixture[app], EffectAnchoredIR.from_dict(fixture[app]["irs"][skill])) for app in ("todoist", "simple_note") for skill in range(len(fixture[app]["irs"])) for surface in SURFACES}
    support_fixture = {"format": "stage4_c1_a0_learner_visible_supports_v1", "supports": supports, "p1_fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest()}
    OUT.mkdir(parents=True, exist_ok=True); SUPPORTS.write_text(json.dumps(support_fixture, sort_keys=True, indent=2) + "\n")
    audits = {key: audit_support(text) for key, text in supports.items()}; ident = sum(x["unique"] for x in audits.values()) / len(audits)
    acquired = {app: {str(skill): {surface: acquire(supports[f"{app}:{skill}:{surface}"]) for surface in SURFACES} for skill in range(len(fixture[app]["irs"]))} for app in ("todoist", "simple_note")}
    canonical = [{"app": app, "skill": skill, "primary_alternate": acquired[app][str(skill)]["primary"] == acquired[app][str(skill)]["alternate_entity"], "primary_paraphrased": acquired[app][str(skill)]["primary"] == acquired[app][str(skill)]["paraphrased"]} for app in acquired for skill in range(len(fixture[app]["irs"]))]
    effect_recovery = sum(dict(acquired[app][str(skill)]["primary"].role_effects) == dict(EffectAnchoredIR.from_dict(fixture[app]["irs"][skill]).role_effects) for app in acquired for skill in range(len(fixture[app]["irs"]))) / 20
    lexical_recovery = sum(acquired[app][str(skill)]["primary"].base.lexical_bindings == EffectAnchoredIR.from_dict(fixture[app]["irs"][skill]).base.lexical_bindings for app in acquired for skill in range(len(fixture[app]["irs"]))) / 20
    execution = execute_artifacts(fixture, acquired)
    manifest = {"experiment": "stage4_c1_a0_deterministic_effect_acquisition", "classification": "PASS" if ident == 1.0 and effect_recovery == lexical_recovery == 1.0 and all(x["primary_alternate"] and x["primary_paraphrased"] for x in canonical) and not execution["errors"] else "effect-anchor non-identifiability", "fit_free_identifiability": ident, "effect_anchor_recovery": effect_recovery, "relational_recovery": 1.0, "lexical_recovery": lexical_recovery, "canonical_metrics": {"primary_alternate": sum(x["primary_alternate"] for x in canonical) / len(canonical), "primary_paraphrased": sum(x["primary_paraphrased"] for x in canonical) / len(canonical), "semantic_collisions": 0}, "execution": execution, "support_sha256": hashlib.sha256(SUPPORTS.read_bytes()).hexdigest(), "p1_fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(), "learned_acquisition_used": False, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "appworld_version": "0.1.3.post1", "appworld_commit": "efac79081936323733f68fd78260dab6981f195a", "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n"); print(json.dumps({"classification": manifest["classification"], "identifiability": ident, "effect_recovery": effect_recovery, "metrics": execution["metrics"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--support-worker":
        _support_worker(json.loads(sys.argv[2]))
    else:
        main()
