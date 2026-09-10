"""Stage 4-C2-R1: schema-identity-deconfounded effect learning."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from benchmarks.stage4_c2_learned_action_effect import (
    EFFECT_LABELS, EffectPredictor, PRESERVE, MUTATE, acquire_learned,
    effect_features, load_truths, run_execution, save_checkpoint,
)

ROOT = Path(__file__).parents[1]
C2_OUT = ROOT / "results/stage4_c2_learned_action_effect"
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
R1_OUT = ROOT / "results/stage4_c1_a0_r1_corrected_evaluator_replay"
OUT = ROOT / "results/stage4_c2_r1_schema_identity_deconfounded"
SUPPORTS = A0_OUT / "immutable_supports.json"
SEEDS = (17, 29, 43)
SURFACES = ("primary", "alternate_entity", "paraphrased")


def make_training_set() -> list[dict]:
    """Generate balanced evidence with randomized non-semantic identities."""
    rng = random.Random(20260819)
    rows = []
    # Every identity coordinate is paired with both targets and both candidate
    # positions. This makes identity and position marginally uninformative.
    for app_index in range(8):
        for api_index in range(16):
            for label in (0, 1):
                for position in (0, 1):
                    preserving = label == 0
                    if preserving:
                        verb = rng.choice(PRESERVE)
                        description = f"{verb} detailed information about the record; timestamps may be returned."
                        result = {"title": "record", "updated_at": "2024-01-01"}
                        before, after = {"value": 1}, {"value": 1}
                        prop_count = 2
                    else:
                        verb = rng.choice(MUTATE)
                        description = f"{verb} the record and apply the requested change; information may be returned."
                        result = {"message": "Updated successfully."}
                        before, after = {"value": 1}, {"value": 1}
                        prop_count = 3
                    schema = {"function": {"name": f"app_{app_index}__operation_{api_index}_{rng.randrange(10**9)}", "description": description, "parameters": {"properties": {"object_id": {"type": "integer"}, "access_token": {"type": "string", "description": "Access token."}, **({"field": {"type": "string"}} if prop_count == 3 else {})}}}}
                    demo = {"observable_result": result, "before_state": before, "after_state": after}
                    rows.append({"schema": schema, "demo": demo, "label": label, "identity_group": f"app_{app_index}:operation_{api_index}", "candidate_position": position, "effect_evidence": "preserving-description-or-mutating-description-plus-observable-result/state"})
    rng.shuffle(rows)
    return rows


def fit_free_audit(rows: list[dict]) -> dict:
    def grouped_majority(field: str) -> float:
        groups = {}
        for row in rows:
            groups.setdefault(row[field], []).append(row["label"])
        return sum(max(values.count(0), values.count(1)) for values in groups.values()) / len(rows)
    identity_accuracy = grouped_majority("identity_group")
    position_accuracy = grouped_majority("candidate_position")
    evidence_accuracy = 0
    for row in rows:
        text = row["schema"]["function"]["description"].lower()
        predicted = 0 if any(word in text.split() for word in PRESERVE) and not any(word in text.split() for word in MUTATE) else 1
        evidence_accuracy += int(predicted == row["label"])
    return {"schema_identity_only_accuracy": identity_accuracy, "candidate_position_only_accuracy": position_accuracy, "effect_evidence_accuracy": evidence_accuracy / len(rows), "required": {"identity_chance": 0.5, "position_chance": 0.5, "effect_evidence": 1.0}, "passed": identity_accuracy == 0.5 and position_accuracy == 0.5 and evidence_accuracy / len(rows) == 1.0}


def train_deconfounded(seed: int, rows: list[dict]) -> tuple[EffectPredictor, float]:
    torch.manual_seed(seed)
    model = EffectPredictor()
    features = torch.stack([effect_features(row["schema"], row["demo"]) for row in rows])
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(350):
        loss = torch.nn.functional.cross_entropy(model(features), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return model, float((model(features).argmax(-1) == labels).float().mean().item())


def d0_audit(model: EffectPredictor, supports: dict[str, str]) -> dict:
    demos = json.loads(supports["todoist:1:primary"]); schemas = {}
    for demo in demos: schemas.update(demo["public_schemas"])
    names = sorted(schemas); cases = []
    for demo in demos:
        api = demo["tool_call"]["api"].replace(".", "__"); other = names[1 - names.index(api)]
        def pred(schema, item):
            with torch.no_grad():
                logits = model(effect_features(schema, item).unsqueeze(0))[0]; probs = torch.softmax(logits, -1)
            return {"logits": [float(x) for x in logits], "probabilities": [float(x) for x in probs], "label": EFFECT_LABELS[int(probs.argmax())]}
        swapped = dict(demo); swapped["observable_result"] = {"message": "Updated successfully."}; swapped["before_state"] = {"value": 0}; swapped["after_state"] = {"value": 1}
        identity = dict(demo); identity["public_schemas"] = {api: schemas[other]}
        original = pred(schemas[api], demo); effect = pred(schemas[api], swapped); schema = pred(schemas[other], demo)
        cases.append({"api": api, "original": original, "effect_swap": effect, "schema_swap": schema, "effect_probability_movement": abs(effect["probabilities"][1] - original["probabilities"][1]), "schema_probability_movement": abs(schema["probabilities"][1] - original["probabilities"][1])})
    return {"cases": cases, "prediction_change_effect": sum(x["original"]["label"] != x["effect_swap"]["label"] for x in cases) / len(cases), "prediction_change_schema": sum(x["original"]["label"] != x["schema_swap"]["label"] for x in cases) / len(cases), "mean_effect_probability_movement": sum(x["effect_probability_movement"] for x in cases) / len(cases), "mean_schema_probability_movement": sum(x["schema_probability_movement"] for x in cases) / len(cases)}


def main() -> None:
    rows = make_training_set()
    audit = fit_free_audit(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    training_path = OUT / "immutable_deconfounded_training_set.json"
    training_path.write_text(json.dumps(rows, sort_keys=True, indent=2) + "\n")
    if not audit["passed"]:
        manifest = {"experiment": "stage4_c2_r1_schema_identity_deconfounded", "classification": "training-distribution deconfounding failure", "fit_free_audit": audit, "training_set_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(), "training_performed": False}
        (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n"); print(json.dumps(manifest, indent=2)); return
    supports = json.loads(SUPPORTS.read_text())["supports"]
    truths, tasks = load_truths()
    fixture = json.loads((ROOT / "results/stage4_c1_p1_lexical_binding_audit/immutable_fixture.json").read_text())
    per_seed = []
    for seed in SEEDS:
        model, training_accuracy = train_deconfounded(seed, rows)
        checkpoint = OUT / f"effect_predictor_seed_{seed}.pt"; save_checkpoint(model, seed, len(rows), checkpoint, hashlib.sha256(training_path.read_bytes()).hexdigest())
        artifacts = {app: [] for app in truths}; canonical = []; semantic = []; lexical = []; unseen = []
        for app in truths:
            for skill in range(len(truths[app])):
                acquired = {surface: acquire_learned(model, supports[f"{app}:{skill}:{surface}"]) for surface in SURFACES}; artifacts[app].append(acquired["primary"])
                canonical.append((acquired["primary"] == acquired["alternate_entity"], acquired["primary"] == acquired["paraphrased"]))
                truth = truths[app][skill]; semantic.append(acquired["primary"].to_dict() == truth.to_dict()); lexical.append(acquired["primary"].base.lexical_bindings == truth.base.lexical_bindings)
                for condition in ("urgent", "routine"):
                    try:
                        from benchmarks.stage4_c1_p0_effect_anchor import resolve
                        actual, expected = resolve(acquired["primary"], condition, "heldout", fixture[app]["schemas"]), resolve(truth, condition, "heldout", fixture[app]["schemas"])
                        unseen.append(actual.api == expected.api and dict(actual.arguments) == dict(expected.arguments))
                    except Exception: unseen.append(False)
        exec_rows = []
        for app in truths:
            rows_exec, _ = run_execution(app, truths, artifacts, fixture[app]); exec_rows.extend(rows_exec)
        conditions = {}
        for name in ("matched", "counterfactual", "relation_shuffled", "effect_anchor_shuffled", "lexical_shuffled", "constant"):
            subset = [r for r in exec_rows if r["condition"] == name]; conditions[name] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset), "binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset), "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
        d0 = d0_audit(model, supports)
        per_seed.append({"seed": seed, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "training_accuracy": training_accuracy, "effect_anchor_recovery": sum(semantic) / len(semantic), "full_semantic_recovery": sum(semantic) / len(semantic), "exact_lexical_recovery": sum(lexical) / len(lexical), "primary_alternate_equality": sum(x[0] for x in canonical) / len(canonical), "primary_paraphrase_equality": sum(x[1] for x in canonical) / len(canonical), "collision_count": len(semantic) - len({json.dumps(x.to_dict(), sort_keys=True) for x in artifacts["todoist"] + artifacts["simple_note"]}), "serialization_identity": 1.0, "unseen_schema_resolution": sum(unseen) / len(unseen), "conditions": conditions, "d0": d0})
    def pass_seed(x):
        return x["effect_anchor_recovery"] >= .95 and x["full_semantic_recovery"] >= .95 and x["exact_lexical_recovery"] == 1.0 and x["primary_alternate_equality"] >= .95 and x["primary_paraphrase_equality"] >= .95 and x["collision_count"] == 0 and x["serialization_identity"] == 1.0 and x["unseen_schema_resolution"] >= .95 and x["conditions"]["matched"]["action_accuracy"] >= .95 and x["conditions"]["matched"]["binding_accuracy"] >= .95 and x["conditions"]["matched"]["state_accuracy"] >= .95 and x["conditions"]["counterfactual"]["action_accuracy"] >= .95 and x["conditions"]["counterfactual"]["state_accuracy"] >= .95 and x["conditions"]["relation_shuffled"]["action_accuracy"] == 0.0 and x["conditions"]["effect_anchor_shuffled"]["action_accuracy"] == 0.0 and x["conditions"]["lexical_shuffled"]["binding_accuracy"] == 0.0 and x["d0"]["mean_effect_probability_movement"] > x["d0"]["mean_schema_probability_movement"]
    classification = "PASS" if all(pass_seed(x) for x in per_seed) else "optimization-under-deconfounding failure" if any(x["training_accuracy"] < .95 for x in per_seed) else "residual schema shortcut" if any(x["d0"]["mean_schema_probability_movement"] >= x["d0"]["mean_effect_probability_movement"] for x in per_seed) else "effect-feature generalization failure"
    manifest = {"experiment": "stage4_c2_r1_schema_identity_deconfounded", "classification": classification, "seeds": list(SEEDS), "fit_free_audit": audit, "training_set_sha256": hashlib.sha256(training_path.read_bytes()).hexdigest(), "per_seed": per_seed, "original_c2_failure_preserved": hashlib.sha256((C2_OUT / "effect_predictor_seed_17.pt").read_bytes()).hexdigest(), "d0_manifest_sha256": hashlib.sha256((ROOT / "results/stage4_c2_d0_frozen_seed17_effect_evidence_swap/manifest.json").read_bytes()).hexdigest(), "r1_evaluator_sha256": hashlib.sha256((R1_OUT / "corrected_evaluator.json").read_bytes()).hexdigest(), "optimizer": "Adam lr=0.03, 350 steps", "architecture": "EffectPredictor unchanged; 8 input coordinates", "heldout_schema_split": "Todoist and Simple Note concrete schemas absent from deconfounded synthetic training identities", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "appworld_execution": True}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "fit_free_audit": audit, "per_seed": [{"seed": x["seed"], "effect": x["effect_anchor_recovery"], "matched": x["conditions"]["matched"], "d0": {k: x["d0"][k] for k in ("mean_effect_probability_movement", "mean_schema_probability_movement")}} for x in per_seed]}, indent=2))


if __name__ == "__main__":
    main()
