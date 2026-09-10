"""Stage 4-C2-D0: frozen seed-17 effect-evidence swap audit."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.stage4_c2_learned_action_effect import EffectPredictor, effect_features

ROOT = Path(__file__).parents[1]
C2_OUT = ROOT / "results/stage4_c2_learned_action_effect"
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
SUPPORTS = A0_OUT / "immutable_supports.json"
SUPPORT_KEY = "todoist:1:primary"
CHECKPOINT = C2_OUT / "effect_predictor_seed_17.pt"
OUT = ROOT / "results/stage4_c2_d0_frozen_seed17_effect_evidence_swap"


def load_model() -> EffectPredictor:
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = EffectPredictor()
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def schema_text(schema: dict) -> str:
    return str(schema["function"].get("name", "")) + " " + str(schema["function"].get("description", ""))


def effect_swap_demo(demo: dict) -> dict:
    """Swap only observable effect evidence; preserve request, schema, and call."""
    result = dict(demo.get("observable_result", {}))
    before = demo.get("before_state")
    after = demo.get("after_state")
    if before == after:
        result["message"] = "Updated successfully."
        new_before = {"value": 0}
        new_after = {"value": 1}
    else:
        result["message"] = "Shown successfully."
        new_before = {"value": 1}
        new_after = {"value": 1}
    changed = dict(demo)
    changed["observable_result"] = result
    changed["before_state"] = new_before
    changed["after_state"] = new_after
    return changed


def schema_swap_demo(demo: dict, other_schema: dict) -> dict:
    """Swap only schema/API identity cues; preserve result and state evidence."""
    changed = dict(demo)
    api = str(demo["tool_call"]["api"]).replace(".", "__")
    changed["public_schemas"] = {api: other_schema}
    changed["public_tool_schema"] = [api]
    return changed


def score(model: EffectPredictor, schema: dict, demo: dict) -> dict:
    features = effect_features(schema, demo)
    with torch.no_grad():
        logits = model(features.unsqueeze(0))[0]
        probabilities = torch.softmax(logits, dim=-1)
    return {"features": [float(x) for x in features], "logits": [float(x) for x in logits], "probabilities": [float(x) for x in probabilities], "predicted_index": int(probabilities.argmax()), "predicted_label": ("state_preserving", "state_mutating")[int(probabilities.argmax())]}


def delta(a: dict, b: dict) -> dict:
    return {"logit_delta": [b["logits"][i] - a["logits"][i] for i in range(2)], "probability_delta": [b["probabilities"][i] - a["probabilities"][i] for i in range(2)], "prediction_changed": a["predicted_label"] != b["predicted_label"]}


def main() -> None:
    supports = json.loads(SUPPORTS.read_text())["supports"]
    demos = json.loads(supports[SUPPORT_KEY])
    schemas = {}
    for demo in demos:
        schemas.update(demo["public_schemas"])
    api_names = sorted(schemas)
    if len(api_names) != 2:
        raise RuntimeError("D0 requires exactly the two failed-support APIs")
    model = load_model()
    cases = []
    for index, demo in enumerate(demos):
        api = str(demo["tool_call"]["api"]).replace(".", "__")
        other_api = api_names[1 - api_names.index(api)]
        original = score(model, schemas[api], demo)
        effect_swapped = score(model, schemas[api], effect_swap_demo(demo))
        identity_swapped = score(model, schemas[other_api], demo)
        cases.append({"api": api, "other_api": other_api, "schema_text": schema_text(schemas[api]), "other_schema_text": schema_text(schemas[other_api]), "original": original, "effect_evidence_swapped": effect_swapped, "schema_identity_swapped": identity_swapped, "original_to_effect_swap": delta(original, effect_swapped), "original_to_identity_swap": delta(original, identity_swapped), "feature_coordinate_checks": {"dimension_original": len(original["features"]) == 8, "dimension_effect_swap": len(effect_swapped["features"]) == 8, "dimension_identity_swap": len(identity_swapped["features"]) == 8, "finite": all(math.isfinite(x) for result in (original, effect_swapped, identity_swapped) for x in result["features"] + result["logits"] + result["probabilities"])}})
    effect_tracking = sum(case["original_to_effect_swap"]["prediction_changed"] for case in cases) / len(cases)
    identity_tracking = sum(case["original_to_identity_swap"]["prediction_changed"] for case in cases) / len(cases)
    effect_probability_movement = sum(abs(case["original_to_effect_swap"]["probability_delta"][1]) for case in cases) / len(cases)
    identity_probability_movement = sum(abs(case["original_to_identity_swap"]["probability_delta"][1]) for case in cases) / len(cases)
    if effect_tracking == 0.0 and identity_tracking > 0.0:
        classification = "schema-identity shortcut/memorization"
    elif effect_tracking > 0.0 and any(case["original"]["predicted_label"] != case["effect_evidence_swapped"]["predicted_label"] for case in cases):
        classification = "effect-feature distribution-transfer failure"
    elif effect_tracking == 0.0 and identity_tracking == 0.0:
        classification = "unseen-domain class collapse"
    elif not all(case["feature_coordinate_checks"]["dimension_original"] and case["feature_coordinate_checks"]["dimension_effect_swap"] and case["feature_coordinate_checks"]["dimension_identity_swap"] and case["feature_coordinate_checks"]["finite"] for case in cases):
        classification = "C2 feature-coordinate defect"
    else:
        classification = "mixed effect/schema evidence response"
    manifest = {"experiment": "stage4_c2_d0_frozen_seed17_effect_evidence_swap", "classification": classification, "preserved_failure_classification": "schema memorization/generalization failure", "support_key": SUPPORT_KEY, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), "support_sha256": hashlib.sha256(SUPPORTS.read_bytes()).hexdigest(), "cases": cases, "aggregate": {"prediction_change_under_effect_swap": effect_tracking, "prediction_change_under_schema_identity_swap": identity_tracking, "mean_mutating_probability_movement_effect_swap": effect_probability_movement, "mean_mutating_probability_movement_identity_swap": identity_probability_movement}, "counterfactuals": {"effect_swap": "schema/API identity, request, and tool call held fixed; only observable result and before/after state replaced with generic opposite-effect evidence", "schema_identity_swap": "effect-bearing result and before/after state held fixed; public schema identity replaced by the paired API schema", "feature_coordinate_convention": "frozen 8-coordinate effect_features order; no new features or reordering"}, "evaluator_truth_used_after_prediction_only": True, "training_performed": False, "seeds_run": [], "appworld_execution": False, "frozen": ["seed-17 checkpoint", "C2 feature construction", "C2 training data/split", "R1 evaluator", "immutable supports", "prior artifacts"], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"; path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "aggregate": manifest["aggregate"], "cases": [{"api": x["api"], "original": x["original"], "effect_swap": x["effect_evidence_swapped"], "identity_swap": x["schema_identity_swapped"]} for x in cases]}, indent=2))


if __name__ == "__main__":
    main()
