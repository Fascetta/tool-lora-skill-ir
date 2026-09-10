"""Stage 4-C1-P0: evaluator-truth effect anchoring and schema resolution."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from copy import deepcopy
from itertools import permutations
from pathlib import Path

from benchmarks.stage4_b0_appworld_oracle_reconnect import b0_ir
from benchmarks.stage4_c0_unseen_tool_oracle import c0_ir
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import audit_identifiability, resolve
from tool_lora.stage4_p0.contract import GenericPolicyIR, PolicyBranch, ArgumentBinding

ROOT = Path(__file__).parents[1]
APP_ROOT = Path(os.environ.get("APPWORLD_ROOT", "/tmp/appworld-root-b0-ohMzCT"))
OUT = ROOT / "results/stage4_c1_p0_effect_anchor"


def schemas(app: str, names: tuple[str, ...]) -> dict[str, dict]:
    path = APP_ROOT / "data/api_docs/function_calling" / f"{app}.json"
    if path.exists():
        data = json.loads(path.read_text())
        return {item["function"]["name"]: item for item in data if item["function"]["name"] in names}
    fixture_path = ROOT / "results/stage4_c1_p0_effect_anchor/immutable_fixture.json"
    fixture = json.loads(fixture_path.read_text())
    stored = fixture[app]["schemas"]
    return {name: stored[name] for name in names}


def anchored(base):
    return EffectAnchoredIR(base, tuple((branch.action, "state_preserving" if branch.action == "dispatch_alpha" else "state_mutating") for branch in base.alternatives))


def build_fixture():
    todo = schemas("todoist", ("todoist__show_task", "todoist__update_task"))
    note = schemas("simple_note", ("simple_note__show_note", "simple_note__delete_note"))
    todo_irs = [anchored(b0_ir(i)) for i in range(12)]
    note_irs = [anchored(c0_ir(i)) for i in range(8)]
    ambiguity = deepcopy(note["simple_note__show_note"]); ambiguity["function"]["name"] = "unrelated_read_candidate"
    return {"format": "stage4_c1_p0_effect_anchor_fixture_v1", "todoist": {"task": "82e2fac_1", "ids": list(range(4766, 4778)), "schemas": todo, "irs": [x.to_dict() for x in todo_irs]},
            "simple_note": {"task": "cf6abd2_2", "ids": list(range(3042, 3050)), "schemas": note, "irs": [x.to_dict() for x in note_irs]},
            "ambiguity_control": {"schemas": {**note, "unrelated_read_candidate": ambiguity}, "ir": note_irs[0].to_dict()},
            "frozen": ["P0 contract", "C0 hidden-policy relations/bindings", "Todoist/Simple Note environments", "lexical semantics", "causal controls"],
            "provenance": {"appworld_version": "0.1.3.post1", "appworld_commit": "efac79081936323733f68fd78260dab6981f195a"}}


def _constant(ir: EffectAnchoredIR) -> EffectAnchoredIR:
    branch = ir.base.alternatives[0]
    base = GenericPolicyIR(ir.base.condition_key, ir.base.condition_value, (branch, branch), ir.base.lexical_bindings)
    return EffectAnchoredIR(base, ((branch.action, dict(ir.role_effects)[branch.action]),))


def _variant(ir: EffectAnchoredIR, name: str, other_literal: str | None = None) -> EffectAnchoredIR:
    base = ir.base
    effects = dict(ir.role_effects)
    if name == "relation_shuffled":
        base = GenericPolicyIR(base.condition_key, base.condition_value, tuple(reversed(base.alternatives)), base.lexical_bindings)
    elif name == "lexical_shuffled":
        base = GenericPolicyIR(base.condition_key, base.condition_value, base.alternatives, (("scope", other_literal),))
    elif name == "effect_anchor_shuffled":
        values = list(effects.values())
        effects = {role: values[1 - i] for i, role in enumerate(effects)}
    return EffectAnchoredIR(base, tuple(effects.items()))


def _worker(items: list[dict[str, str]]) -> None:
    from appworld import AppWorld
    output = []
    for item in items:
        try:
            ir = EffectAnchoredIR.load(item["artifact"])
            schemas = json.loads(Path(item["schemas"]).read_text())
            call = resolve(ir, item["condition_value"], "heldout", schemas)
            app = item["app"]
            method = call.api.split(".", 1)[1]
            identifier = int(dict(call.arguments).values().__iter__().__next__())
            account_email = "joyce-weav@gmail.com" if app == "todoist" else "siwhit@gmail.com"
            account_arg = "task_id" if app == "todoist" else "note_id"
            code = ["import json", "p={x['account_name']:x['password'] for x in apis.supervisor.show_account_passwords()}", f"t=apis.{app}.login(username='{account_email}',password=p['{app}'])['access_token']", f"r=apis.{app}.{method}({account_arg}={identifier},access_token=t)"]
            if app == "todoist":
                code.append(f"s=apis.todoist.show_task(task_id={identifier},access_token=t)")
                code.append("ok=bool(s.get('task_id') == %d)" % identifier)
            elif item["target_effect"] == "state_preserving":
                code.append(f"s=apis.simple_note.show_note(note_id={identifier},access_token=t)")
                code.append("ok=bool(s.get('note_id') == %d)" % identifier)
            else:
                code.append("s=apis.simple_note.search_notes(query='',page_index=0,page_limit=20,access_token=t)")
                code.append("ok=not any(x.get('note_id') == %d for x in s)" % identifier)
            code.append("print(json.dumps({'api': '%s', 'identifier': %d, 'state_ok': ok}))" % (call.api, identifier))
            with AppWorld(task_id=item["task"], experiment_name="stage4_c1_eval", load_ground_truth=False, raise_on_failure=True) as world:
                result = world.execute("\n".join(code))
            output.append(json.loads(result.strip().splitlines()[-1]))
        except Exception as exc:
            output.append({"error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps(output))


def execute_fixture(fixture: dict) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); items = []
        for app in ("todoist", "simple_note"):
            block = fixture[app]; schemas_path = td / f"{app}_schemas.json"; schemas_path.write_text(json.dumps(block["schemas"], sort_keys=True))
            truths = [EffectAnchoredIR.from_dict(x) for x in block["irs"]]
            for skill, truth in enumerate(truths):
                variants = {"matched": truth, "counterfactual": _variant(truth, "relation_shuffled"), "relation_shuffled": _variant(truth, "relation_shuffled"), "lexical_shuffled": _variant(truth, "lexical_shuffled", str(block["ids"][(skill + 1) % len(block["ids"])])), "effect_anchor_shuffled": _variant(truth, "effect_anchor_shuffled"), "constant": _constant(truth)}
                for name, variant in variants.items():
                    artifact = td / f"{app}_{skill}_{name}.json"; variant.save(artifact)
                    for condition in ("urgent", "routine"):
                        target_ir = variants["counterfactual"] if name == "counterfactual" else truth
                        target_action = target_ir.base.alternatives[0].action if condition == target_ir.base.condition_value else target_ir.base.alternatives[1].action
                        target_effect = dict(target_ir.role_effects)[target_action]
                        target_call = resolve(target_ir, condition, "heldout", block["schemas"])
                        items.append({"artifact": str(artifact), "schemas": str(schemas_path), "app": app, "task": block["task"], "condition_value": condition, "target_api": target_call.api, "target_id": str(block["ids"][skill]), "target_effect": target_effect, "name": name, "skill": skill})
        env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('APPWORLD_PYTHONPATH', '')}:{ROOT / 'src'}:{ROOT}"
        result = subprocess.check_output([sys.executable, __file__, "--worker", json.dumps(items)], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL)
        observed = json.loads(result.strip().splitlines()[-1])
        for item, got in zip(items, observed):
            rows.append({"app": item["app"], "skill": item["skill"], "condition": item["name"], "condition_value": item["condition_value"], "action_correct": got.get("api") == item["target_api"], "binding_correct": got.get("identifier") == int(item["target_id"]), "state_correct": got.get("state_ok", False), "error": got.get("error")})
    metrics = {}
    for name in ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "effect_anchor_shuffled", "constant"):
        subset = [x for x in rows if x["condition"] == name]
        metrics[name] = {"action_accuracy": sum(x["action_correct"] for x in subset) / len(subset), "binding_accuracy": sum(x["binding_correct"] for x in subset) / len(subset), "state_accuracy": sum(x["state_correct"] for x in subset) / len(subset)}
    errors = [row for row in rows if row["error"]]
    reload_total = 0
    reload_ok = 0
    for item in items:
        if item["name"] == "matched":
            reload_total += 1
            reloaded = EffectAnchoredIR.load(item["artifact"])
            reload_ok += int(reloaded.to_dict() == json.loads(Path(item["artifact"]).read_text()))
    serialization_identity = reload_ok / reload_total
    return {"performed": True, "pass": not errors and serialization_identity == 1.0 and metrics["matched"]["action_accuracy"] == metrics["matched"]["binding_accuracy"] == metrics["matched"]["state_accuracy"] == 1.0 and metrics["counterfactual"]["state_accuracy"] == 1.0, "classification": "substrate/executor failure" if errors else None, "errors": errors, "metrics": metrics, "rows": rows, "serialization_identity": serialization_identity}


def main():
    fixture = build_fixture()
    audit_t = audit_identifiability([EffectAnchoredIR.from_dict(x) for x in fixture["todoist"]["irs"]], fixture["todoist"]["schemas"])
    audit_n = audit_identifiability([EffectAnchoredIR.from_dict(x) for x in fixture["simple_note"]["irs"]], fixture["simple_note"]["schemas"])
    # Candidate order and irrelevant public tools must not affect a unique map.
    order_rows = []
    for name, block in (("todoist", fixture["todoist"]), ("simple_note", fixture["simple_note"])):
        ir = EffectAnchoredIR.from_dict(block["irs"][0]); base = block["schemas"]
        expected = None
        for order in permutations(base):
            ordered = {key: base[key] for key in order}
            result = audit_identifiability([ir], ordered)
            mapping = result["rows"][0]["mappings"]
            if expected is None: expected = mapping
            order_rows.append({"family": name, "order": list(order), "invariant": mapping == expected})
    distractor = deepcopy(fixture["simple_note"]["schemas"])
    distractor["irrelevant"] = {"function": {"name": "irrelevant", "description": "Return a string", "parameters": {"properties": {"access_token": {"type": "string"}}}}}
    distractor_audit = audit_identifiability([EffectAnchoredIR.from_dict(fixture["simple_note"]["irs"][0])], distractor)
    ambiguity_audit = audit_identifiability([EffectAnchoredIR.from_dict(fixture["ambiguity_control"]["ir"])], fixture["ambiguity_control"]["schemas"])
    all_identifiable = audit_t["unique_mapping"] and audit_n["unique_mapping"] and all(x["invariant"] for x in order_rows) and distractor_audit["unique_mapping"] and not ambiguity_audit["unique_mapping"]
    execution = execute_fixture(fixture) if all_identifiable else {"performed": False, "reason": "action-effect semantics insufficient"}
    manifest = {"experiment": "stage4_c1_p0_effect_anchor", "classification": "PASS" if execution.get("pass") else ("action-effect semantics insufficient" if not all_identifiable else execution.get("classification", "generic resolver implementation failure")), "schema_audits": {"todoist": audit_t, "simple_note": audit_n, "distractor": distractor_audit, "ambiguity": ambiguity_audit}, "order_controls": order_rows, "fixture": fixture, "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True, indent=2).encode()).hexdigest(), "execution": execution, "execution_performed": execution.get("performed", False), "learned_acquisition_used": False, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged"}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "immutable_fixture.json").write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n"); manifest["fixture_sha256"] = hashlib.sha256((OUT / "immutable_fixture.json").read_bytes()).hexdigest(); (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": manifest["classification"], "todoist": audit_t["accuracy"], "simple_note": audit_n["accuracy"], "ambiguity_unique": ambiguity_audit["unique_mapping"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(json.loads(sys.argv[2]))
    else:
        main()
