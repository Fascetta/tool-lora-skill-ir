"""Stage 4-C2-R1-D1: in-distribution schema-channel necessity audit."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from benchmarks.stage4_c2_learned_action_effect import EffectPredictor, effect_features

ROOT = Path(__file__).parents[1]
R1_OUT = ROOT / "results/stage4_c2_r1_schema_identity_deconfounded"
TRAINING = R1_OUT / "immutable_deconfounded_training_set.json"
OUT = ROOT / "results/stage4_c2_r1_d1_in_distribution_schema_channel_audit"
SEEDS = (17, 29, 43)


def model_for(seed: int) -> EffectPredictor:
    payload = torch.load(R1_OUT / f"effect_predictor_seed_{seed}.pt", map_location="cpu", weights_only=True)
    model = EffectPredictor(); model.load_state_dict(payload["model"]); model.eval(); return model


def prediction(model: EffectPredictor, row: dict) -> dict:
    features = effect_features(row["schema"], row["demo"])
    with torch.no_grad():
        logits = model(features.unsqueeze(0))[0]; probabilities = torch.softmax(logits, dim=-1)
    return {"features": [float(x) for x in features], "logits": [float(x) for x in logits], "probabilities": [float(x) for x in probabilities], "argmax": int(probabilities.argmax()), "label": ("state_preserving", "state_mutating")[int(probabilities.argmax())]}


def schema_without_name(schema: dict) -> dict:
    value = json.loads(json.dumps(schema)); value["function"].pop("name", None); return value


def make_pairs(rows: list[dict]) -> list[dict]:
    preserve = [r for r in rows if r["label"] == 0]
    mutate = [r for r in rows if r["label"] == 1]
    pairs = []
    for index in range(8):
        base = preserve[index]; identity_donor = preserve[index + 8]; effect_donor = mutate[index]
        identity_variant = json.loads(json.dumps(base)); identity_variant["schema"]["function"]["name"] = identity_donor["schema"]["function"]["name"]
        pairs.append({"family": "opaque_schema_identity", "source_indices": [rows.index(base), rows.index(identity_donor)], "left": base, "right": identity_variant, "mechanical_check": {"identity_changed": base["schema"]["function"]["name"] != identity_variant["schema"]["function"]["name"], "semantic_schema_fixed": schema_without_name(base["schema"]) == schema_without_name(identity_variant["schema"]), "effect_demo_fixed": base["demo"] == identity_variant["demo"]}})
        effect_variant = {"schema": json.loads(json.dumps(effect_donor["schema"])), "demo": json.loads(json.dumps(effect_donor["demo"])), "label": effect_donor["label"], "identity_group": base["identity_group"], "candidate_position": base["candidate_position"], "effect_evidence": effect_donor["effect_evidence"]}
        effect_variant["schema"]["function"]["name"] = base["schema"]["function"]["name"]
        pairs.append({"family": "effect_bearing_evidence", "source_indices": [rows.index(base), rows.index(effect_donor)], "left": base, "right": effect_variant, "mechanical_check": {"identity_fixed": base["schema"]["function"]["name"] == effect_variant["schema"]["function"]["name"], "effect_family_changed": base["schema"]["function"].get("description") != effect_variant["schema"]["function"].get("description") or base["demo"] != effect_variant["demo"], "candidate_position_fixed": base["candidate_position"] == effect_variant["candidate_position"]}})
        semantic_variant = json.loads(json.dumps(base)); semantic_variant["schema"] = json.loads(json.dumps(effect_donor["schema"])); semantic_variant["schema"]["function"]["name"] = base["schema"]["function"]["name"]
        pairs.append({"family": "public_semantic_schema_content", "source_indices": [rows.index(base), rows.index(effect_donor)], "left": base, "right": semantic_variant, "mechanical_check": {"identity_fixed": base["schema"]["function"]["name"] == semantic_variant["schema"]["function"]["name"], "demo_fixed": base["demo"] == semantic_variant["demo"], "semantic_schema_changed": schema_without_name(base["schema"]) != schema_without_name(semantic_variant["schema"])}})
    return pairs


def main() -> None:
    rows = json.loads(TRAINING.read_text()); pairs = make_pairs(rows)
    if not all(all(pair["mechanical_check"].values()) for pair in pairs):
        raise RuntimeError("D1 mechanical pair audit failed")
    diagnostic_path = OUT / "immutable_diagnostic_pairs.json"; OUT.mkdir(parents=True, exist_ok=True); diagnostic_path.write_text(json.dumps(pairs, sort_keys=True, indent=2) + "\n")
    per_seed = []
    for seed in SEEDS:
        model = model_for(seed); family_rows = []
        for pair in pairs:
            left = prediction(model, pair["left"]); right = prediction(model, pair["right"])
            family_rows.append({"family": pair["family"], "source_indices": pair["source_indices"], "left": left, "right": right, "logit_delta": [right["logits"][i] - left["logits"][i] for i in range(2)], "probability_delta": [right["probabilities"][i] - left["probabilities"][i] for i in range(2)], "argmax_flip": left["argmax"] != right["argmax"]})
        summary = {}
        for family in ("opaque_schema_identity", "effect_bearing_evidence", "public_semantic_schema_content"):
            subset = [x for x in family_rows if x["family"] == family]; summary[family] = {"count": len(subset), "argmax_flip_rate": sum(x["argmax_flip"] for x in subset) / len(subset), "mean_abs_probability_movement": sum(abs(x["probability_delta"][1]) for x in subset) / len(subset), "mean_probability_delta": [sum(x["probability_delta"][i] for x in subset) / len(subset) for i in range(2)]}
        per_seed.append({"seed": seed, "checkpoint_sha256": hashlib.sha256((R1_OUT / f"effect_predictor_seed_{seed}.pt").read_bytes()).hexdigest(), "summary": summary, "pairs": family_rows})
    identity = [x["summary"]["opaque_schema_identity"] for x in per_seed]; effect = [x["summary"]["effect_bearing_evidence"] for x in per_seed]; semantic = [x["summary"]["public_semantic_schema_content"] for x in per_seed]
    if all(x["mean_abs_probability_movement"] < 1e-9 and x["argmax_flip_rate"] == 0.0 for x in identity) and all(x["mean_abs_probability_movement"] < 1e-9 and x["argmax_flip_rate"] == 0.0 for x in effect): classification = "distributed schema shortcut"
    elif all(x["mean_abs_probability_movement"] < 1e-9 for x in identity) and any(x["mean_abs_probability_movement"] > 0.1 for x in semantic): classification = "D0 schema-semantic/identity conflation"
    elif any(x["mean_abs_probability_movement"] > 0.1 for x in effect) and all(x["mean_abs_probability_movement"] < 1e-9 for x in identity): classification = "D0 out-of-distribution intervention artifact"
    elif any(x["mean_abs_probability_movement"] > 0.1 for x in identity): classification = "non-semantic schema shortcut confirmed"
    else: classification = "inconclusive schema-channel dependence"
    manifest = {"experiment": "stage4_c2_r1_d1_in_distribution_schema_channel_audit", "classification": classification, "training_set_sha256": hashlib.sha256(TRAINING.read_bytes()).hexdigest(), "diagnostic_pairs_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(), "original_r1_manifest_sha256": hashlib.sha256((R1_OUT / "manifest.json").read_bytes()).hexdigest(), "mechanical_pair_audit": all(all(p["mechanical_check"].values()) for p in pairs), "pair_counts": {family: sum(p["family"] == family for p in pairs) for family in ("opaque_schema_identity", "effect_bearing_evidence", "public_semantic_schema_content")}, "per_seed": per_seed, "training_performed": False, "frozen_checkpoints": True, "prior_classifications_preserved": True, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    path = OUT / "manifest.json"; path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "pair_counts": manifest["pair_counts"], "per_seed": [{"seed": x["seed"], "summary": x["summary"]} for x in per_seed]}, indent=2))


if __name__ == "__main__":
    main()
