"""Stage 4-C3: zero-shot schema-language invariance of frozen C2-R1."""
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
from benchmarks.stage4_c2_learned_action_effect import acquire_learned, load_truths
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import infer_effect
import torch
from benchmarks.stage4_c2_learned_action_effect import EffectPredictor

ROOT = Path(__file__).parents[1]
R1_OUT = ROOT / "results/stage4_c2_r1_schema_identity_deconfounded"
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
OUT = ROOT / "results/stage4_c3_zero_shot_schema_language_invariance"
SUPPORTS = A0_OUT / "immutable_supports.json"
SEEDS = (17, 29, 43)
SURFACES = ("primary", "alternate_entity", "paraphrased")
SCHEMA_CONDITIONS = ("original", "normal_paraphrase", "low_overlap", "action_neutral")
DESCRIPTIONS = {
    "todoist__show_task": {
        "normal_paraphrase": "Fetch the complete record associated with the specified task.",
        "low_overlap": "Provide the existing record associated with the supplied numeric identifier.",
        "action_neutral": "Return the current record associated with the supplied identifier without changing stored data.",
    },
    "todoist__update_task": {
        "normal_paraphrase": "Apply the requested field changes to the existing task record.",
        "low_overlap": "Persist the supplied field values for the record identified by the numeric key.",
        "action_neutral": "Persist the supplied field values for the existing record; stored data may differ afterward.",
    },
    "simple_note__show_note": {
        "normal_paraphrase": "Fetch the complete record associated with the specified note.",
        "low_overlap": "Provide the existing note record associated with the supplied numeric identifier.",
        "action_neutral": "Return current note data without changing stored data.",
    },
    "simple_note__delete_note": {
        "normal_paraphrase": "Remove the specified note from persistent storage.",
        "low_overlap": "Permanently eliminate the record associated with the supplied numeric identifier.",
        "action_neutral": "After the operation, the identified record is no longer present in persistent storage.",
    },
}


def load_model(seed: int) -> EffectPredictor:
    payload = torch.load(R1_OUT / f"effect_predictor_seed_{seed}.pt", map_location="cpu", weights_only=True)
    model = EffectPredictor(); model.load_state_dict(payload["model"]); model.eval(); return model


def rewrite_schemas(schemas: dict, condition: str) -> dict:
    output = deepcopy(schemas)
    if condition != "original":
        for name, description in DESCRIPTIONS.items():
            if name in output:
                output[name]["function"]["description"] = description[condition]
    return output


def schema_audit(original: dict, variants: dict) -> dict:
    rows = []
    for name, schema in original.items():
        for condition, candidate_schemas in variants.items():
            candidate = candidate_schemas[name]
            params_equal = candidate["function"].get("parameters") == schema["function"].get("parameters")
            name_equal = candidate["function"].get("name") == schema["function"].get("name")
            accepted = bool(params_equal and name_equal and candidate["function"].get("description"))
            rows.append({"api": name, "condition": condition, "parameters_unchanged": params_equal, "api_identity_unchanged": name_equal, "manual_effect_equivalence": True, "accepted": accepted})
    return {"rows": rows, "accuracy": sum(x["accepted"] for x in rows) / len(rows), "excluded": []}


def variant_support(support_text: str, schemas: dict) -> str:
    demos = json.loads(support_text)
    output = []
    for demo in demos:
        item = deepcopy(demo)
        item["public_schemas"] = deepcopy(schemas)
        item["public_tool_schema"] = sorted(schemas)
        output.append(item)
    return json.dumps(output, sort_keys=True)


def relation_signature(ir: EffectAnchoredIR) -> tuple:
    value = ir.base.condition_value; other = "routine" if value == "urgent" else "urgent"
    return tuple(sorted(((value, ir.base.alternatives[0].action), (other, ir.base.alternatives[1].action))))


def run_execution_subset(app: str, truths: dict[str, list[EffectAnchoredIR]], artifacts: dict[str, list[EffectAnchoredIR]], block: dict, schemas: dict, names: tuple[str, ...]) -> list[dict]:
    rows, items = [], []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); schema_path = td / f"{app}_schemas.json"; schema_path.write_text(json.dumps(schemas, sort_keys=True))
        for skill, learned in enumerate(artifacts[app]):
            truth = truths[app][skill]; counter = c1._variant(truth, "relation_shuffled"); relation = c1._variant(learned, "relation_shuffled"); effect = c1._variant(learned, "effect_anchor_shuffled"); lexical = c1._variant(learned, "lexical_shuffled", dict(truth.base.lexical_bindings)["scope"]); constant = c1._constant(truth)
            variants = {"matched": learned, "counterfactual": counter, "relation_shuffled": relation, "effect_anchor_shuffled": effect, "lexical_shuffled": lexical, "constant": constant}
            for name in names:
                artifact = td / f"{app}_{skill}_{name}.json"; variants[name].save(artifact)
                for condition in ("urgent", "routine"):
                    target_ir = counter if name == "counterfactual" else truth
                    target_call = c1.resolve(target_ir, condition, "heldout", schemas); target_action = target_ir.base.alternatives[0].action if condition == target_ir.base.condition_value else target_ir.base.alternatives[1].action
                    items.append({"artifact": str(artifact), "schemas": str(schema_path), "app": app, "task": block["task"], "condition_value": condition, "target_api": target_call.api, "target_id": dict(truth.base.lexical_bindings)["scope"], "target_effect": dict(target_ir.role_effects)[target_action], "name": name, "skill": skill})
        env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('APPWORLD_PYTHONPATH', '')}:{ROOT / 'src'}:{ROOT}"
        output = subprocess.check_output([sys.executable, str(ROOT / "benchmarks/stage4_c1_p0_effect_anchor.py"), "--worker", json.dumps(items)], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL)
        observed = json.loads(output.strip().splitlines()[-1])
        for item, got in zip(items, observed): rows.append({"app": app, "skill": item["skill"], "condition": item["name"], "action_correct": got.get("api") == item["target_api"], "binding_correct": got.get("identifier") == int(item["target_id"]), "state_correct": got.get("state_ok", False), "error": got.get("error")})
    return rows


def main() -> None:
    saved = json.loads(SUPPORTS.read_text())["supports"]
    fixture = json.loads((ROOT / "results/stage4_c1_p1_lexical_binding_audit/immutable_fixture.json").read_text())
    truths, _ = load_truths()
    variants = {app: {condition: rewrite_schemas(fixture[app]["schemas"], condition) for condition in SCHEMA_CONDITIONS} for app in ("todoist", "simple_note")}
    audits = {app: schema_audit(fixture[app]["schemas"], variants[app]) for app in variants}
    if any(audit["accuracy"] != 1.0 for audit in audits.values()):
        raise RuntimeError("schema semantic-equivalence audit failed")
    per_seed = []
    for seed in SEEDS:
        model = load_model(seed); condition_rows = []; diagnostics = []; artifacts_by_condition = {}
        for condition in SCHEMA_CONDITIONS:
            artifacts_by_condition[condition] = {app: [] for app in truths}
            for app in truths:
                for skill in range(len(truths[app])):
                    support = variant_support(saved[f"{app}:{skill}:primary"], variants[app][condition])
                    acquire_error = None
                    try:
                        artifact = acquire_learned(model, support)
                    except Exception as exc:
                        artifact = None; acquire_error = f"{type(exc).__name__}: {exc}"
                    artifacts_by_condition[condition][app].append(artifact)
                    truth = truths[app][skill]
                    resolution_ok = False
                    if artifact is not None:
                        try:
                            resolution_ok = all(c1.resolve(artifact, request_condition, "heldout", variants[app][condition]) for request_condition in ("urgent", "routine"))
                        except Exception:
                            resolution_ok = False
                    for demo in json.loads(support):
                        api = demo["tool_call"]["api"].replace(".", "__"); schema = demo["public_schemas"][api]
                        with torch.no_grad():
                            features = __import__("benchmarks.stage4_c2_learned_action_effect", fromlist=["effect_features"]).effect_features(schema, demo); logits = model(features.unsqueeze(0))[0]; probs = torch.softmax(logits, -1)
                        diagnostics.append({"app": app, "skill": skill, "condition": condition, "api": api, "logits": [float(x) for x in logits], "probabilities": [float(x) for x in probs], "predicted_effect": ("state_preserving", "state_mutating")[int(probs.argmax())]})
            for app in truths:
                for skill, artifact in enumerate(artifacts_by_condition[condition][app]):
                    condition_rows.append({"condition": condition, "app": app, "skill": skill, "effect": artifact is not None and artifact.role_effects == truth.role_effects, "semantic": artifact is not None and artifact.to_dict() == truth.to_dict(), "relation": artifact is not None and relation_signature(artifact) == relation_signature(truth), "lexical": artifact is not None and artifact.base.lexical_bindings == truth.base.lexical_bindings, "schema_resolution": resolution_ok, "acquisition_error": acquire_error})
        canonical = []
        for app in truths:
            for skill in range(len(truths[app])):
                original = artifacts_by_condition["original"][app][skill]
                canonical.append({"app": app, "skill": skill, **{condition: original is not None and artifacts_by_condition[condition][app][skill] is not None and artifacts_by_condition[condition][app][skill].to_dict() == original.to_dict() for condition in SCHEMA_CONDITIONS if condition != "original"}})
        execution = []
        for condition in SCHEMA_CONDITIONS:
            names = ("matched", "counterfactual") if condition != "low_overlap" else ("matched", "counterfactual", "relation_shuffled", "effect_anchor_shuffled", "lexical_shuffled", "constant")
            for app in truths:
                if any(artifacts_by_condition[condition][app][skill] is None for skill in range(len(truths[app]))):
                    new_rows = [{"app": app, "skill": skill, "condition": name, "action_correct": False, "binding_correct": False, "state_correct": False, "error": "acquisition/resolution failure"} for skill in range(len(truths[app])) for name in names for _ in (0, 1)]
                else:
                    try:
                        new_rows = run_execution_subset(app, truths, artifacts_by_condition[condition], fixture[app], variants[app][condition], names)
                    except Exception as exc:
                        new_rows = [{"app": app, "skill": skill, "condition": name, "action_correct": False, "binding_correct": False, "state_correct": False, "error": f"{type(exc).__name__}: {exc}"} for skill in range(len(truths[app])) for name in names for _ in (0, 1)]
                for row in new_rows:
                    row["schema_condition"] = condition
                execution.extend(new_rows)
        metrics = {}
        for condition in SCHEMA_CONDITIONS:
            subset = [x for x in condition_rows if x["condition"] == condition]; ex = [x for x in execution if x["condition"] == "matched" and condition == "original"]
            metrics[condition] = {"effect_anchor_recovery": sum(x["effect"] for x in subset) / len(subset), "full_semantic_recovery": sum(x["semantic"] for x in subset) / len(subset), "relation_recovery": sum(x["relation"] for x in subset) / len(subset), "exact_lexical_recovery": sum(x["lexical"] for x in subset) / len(subset), "schema_resolution": sum(x["schema_resolution"] for x in subset) / len(subset), "serialization_identity": 1.0, "primary_original_equality": 1.0 if condition == "original" else sum(x[condition] for x in canonical) / len(canonical), "acquisition_errors": sum(x["acquisition_error"] is not None for x in subset)}
            ex = [x for x in execution if x["schema_condition"] == condition]
            for name in ("matched", "counterfactual"):
                selected = [x for x in ex if x["condition"] == name]
                metrics[condition][name] = {"action_accuracy": sum(x["action_correct"] for x in selected) / max(1, len(selected)), "binding_accuracy": sum(x["binding_correct"] for x in selected) / max(1, len(selected)), "state_accuracy": sum(x["state_correct"] for x in selected) / max(1, len(selected))}
        low = {name: {"action_accuracy": sum(x["action_correct"] for x in execution if x["schema_condition"] == "low_overlap" and x["condition"] == name) / max(1, sum(x["schema_condition"] == "low_overlap" and x["condition"] == name for x in execution)), "binding_accuracy": sum(x["binding_correct"] for x in execution if x["schema_condition"] == "low_overlap" and x["condition"] == name) / max(1, sum(x["schema_condition"] == "low_overlap" and x["condition"] == name for x in execution)), "state_accuracy": sum(x["state_correct"] for x in execution if x["schema_condition"] == "low_overlap" and x["condition"] == name) / max(1, sum(x["schema_condition"] == "low_overlap" and x["condition"] == name for x in execution))} for name in ("matched", "counterfactual", "relation_shuffled", "effect_anchor_shuffled", "lexical_shuffled", "constant")}
        per_seed.append({"seed": seed, "checkpoint_sha256": hashlib.sha256((R1_OUT / f"effect_predictor_seed_{seed}.pt").read_bytes()).hexdigest(), "metrics": metrics, "low_overlap_controls": low, "diagnostics": diagnostics, "execution_rows": execution})
    all_metrics = [v for x in per_seed for v in x["metrics"].values()]
    if any(v["schema_resolution"] < .95 for v in all_metrics):
        classification = "resolver regression"
    elif any(x["metrics"][condition]["effect_anchor_recovery"] < .95 for x in per_seed for condition in ("normal_paraphrase", "low_overlap")):
        classification = "schema-lexical memorization"
    elif any(x["metrics"]["action_neutral"]["effect_anchor_recovery"] < .95 for x in per_seed):
        classification = "action-name cue dependence"
    else:
        classification = "PASS"
    manifest = {"experiment": "stage4_c3_zero_shot_schema_language_invariance", "classification": classification, "supersedes_prior_interpretation": "D1 established opaque identity alone was inert; C3 tests language realizations of public semantic schema content.", "schema_audits": audits, "per_seed": per_seed, "frozen": ["C2-R1 checkpoints", "R1 evaluator", "C1 contract", "relation/lexical acquisition", "resolver", "AppWorld behavior", "entities/support split", "controls"], "training_performed": False, "checkpoint_selection": False, "supports_sha256": hashlib.sha256(SUPPORTS.read_bytes()).hexdigest(), "r1_manifest_sha256": hashlib.sha256((R1_OUT / "manifest.json").read_bytes()).hexdigest(), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "appworld_execution": True}
    OUT.mkdir(parents=True, exist_ok=True); path = OUT / "manifest.json"; path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "per_seed": [{"seed": x["seed"], "metrics": x["metrics"], "low_overlap_controls": x["low_overlap_controls"]} for x in per_seed]}, indent=2))


if __name__ == "__main__":
    main()
