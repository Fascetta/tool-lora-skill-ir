"""Stage 4-A2: learned correspondence into the frozen P0/A1 contract."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from benchmarks.stage4_a1_learned_tool_policy import (
    P0_FIXTURE, SCHEMA, OUT as A1_OUT, fresh_execute, heldout_request,
    identifiability_audit, render_support,
)
from benchmarks.stage4_p0_oracle_tool_policy import make_constant_ir, make_ir
from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest, ToolDecision, interpret
from tool_lora.stage4_p0.learned_correspondence import (
    CorrespondenceNet, TrainedCorrespondence, acquire_with_model, train,
)

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_a2_learned_correspondence"
A1_SUPPORT = A1_OUT / "immutable_support_evaluation.json"
SEEDS = (17, 29, 43)


def save_checkpoint(trained: TrainedCorrespondence, path: Path) -> None:
    torch.save({"seed": trained.seed, "training_examples": trained.training_examples,
                "model": trained.model.state_dict()}, path)


def load_checkpoint(path: Path) -> TrainedCorrespondence:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = CorrespondenceNet()
    model.load_state_dict(payload["model"])
    model.eval()
    return TrainedCorrespondence(model, int(payload["seed"]), int(payload["training_examples"]))


def eval_seed(seed: int, truths: list[GenericPolicyIR], supports: dict[str, list[str]], out: Path) -> dict[str, object]:
    trained = train(seed)
    checkpoint = out / f"correspondence_seed_{seed}.pt"
    save_checkpoint(trained, checkpoint)
    rows = []
    canonical = []
    for skill, truth in enumerate(truths):
        acquired = {variant: acquire_with_model(trained, supports[f"{skill}:{variant}"][0])
                    for variant in ("primary", "alternate", "paraphrase")}
        canonical.append({"skill": skill, "primary_alternate_equal": acquired["primary"] == acquired["alternate"],
                          "primary_paraphrase_equal": acquired["primary"] == acquired["paraphrase"]})
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.json"
            acquired["primary"].save(path)
            reloaded = GenericPolicyIR.load(path)
            reload_identity = path.read_bytes() == (json.dumps(acquired["primary"].to_dict(), sort_keys=True, indent=2) + "\n").encode()
            del acquired["primary"]
            matched = {surface: fresh_execute(path, heldout_request(skill, surface)) for surface in ("urgent", "routine")}
        counter = acquire_with_model(trained, render_support(make_ir(skill, counterfactual=True), variant="counterfactual", skill=skill))
        relation = GenericPolicyIR(reloaded.condition_key, reloaded.condition_value, tuple(reversed(reloaded.alternatives)), reloaded.lexical_bindings)
        lexical = GenericPolicyIR(reloaded.condition_key, reloaded.condition_value, reloaded.alternatives,
                                  (("scope", truths[(skill + 1) % len(truths)].lexical_bindings[0][1]),))
        controls = {
            "relation_shuffled": relation, "lexical_shuffled": lexical, "constant": make_constant_ir(),
        }
        for surface in ("urgent", "routine"):
            target = interpret(truth, heldout_request(skill, surface), SCHEMA)
            counter_target = interpret(make_ir(skill, counterfactual=True), heldout_request(skill, surface), SCHEMA)
            cases = [("matched", matched[surface], target), ("counterfactual", interpret(counter, heldout_request(skill, surface), SCHEMA), counter_target)]
            cases.extend((name, interpret(variant, heldout_request(skill, surface), SCHEMA), target) for name, variant in controls.items())
            for condition, decision, expected in cases:
                rows.append({"skill": skill, "surface": surface, "condition": condition,
                             "action_correct": decision.action == expected.action,
                             "argument_correct": decision.arguments == expected.arguments,
                             "opaque_exact": decision.arguments[1][1] == expected.arguments[1][1],
                             "reload_identity": reload_identity if condition == "matched" else None})
    conditions = {}
    for condition in ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant"):
        subset = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset),
                                 "argument_accuracy": sum(r["argument_correct"] for r in subset) / len(subset),
                                 "opaque_exact": sum(r["opaque_exact"] for r in subset) / len(subset)}
    artifacts = [acquire_with_model(trained, supports[f"{skill}:primary"][0]).to_dict() for skill in range(len(truths))]
    return {"seed": seed, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_examples": trained.training_examples, "conditions": conditions, "rows": rows,
            "canonical_metrics": {
                "primary_alternate_equality": sum(x["primary_alternate_equal"] for x in canonical) / len(canonical),
                "primary_paraphrase_equality": sum(x["primary_paraphrase_equal"] for x in canonical) / len(canonical),
                "semantic_collision_count": len(artifacts) - len({json.dumps(x, sort_keys=True) for x in artifacts}),
                "relational_recovery": conditions["matched"]["action_accuracy"],
                "opaque_recovery": conditions["matched"]["opaque_exact"],
                "serialization_identity": sum(r["reload_identity"] is True for r in rows if r["condition"] == "matched") / (len(truths) * 2),
            }}


def main() -> None:
    fixture = json.loads(P0_FIXTURE.read_text())
    truths = [GenericPolicyIR.from_dict(item) for item in fixture["skills"]]
    saved = json.loads(A1_SUPPORT.read_text())
    supports = saved["supports"]
    audit = identifiability_audit(supports, truths)
    if not audit["all_unique"]:
        raise RuntimeError("fit-free identifiability audit failed")
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = [eval_seed(seed, truths, supports, OUT) for seed in SEEDS]
    support_hash = hashlib.sha256(A1_SUPPORT.read_bytes()).hexdigest()
    manifest = {
        "experiment": "stage4_a2_learned_correspondence", "seeds": list(SEEDS), "skill_count": len(truths),
        "p0_fixture_sha256": hashlib.sha256(P0_FIXTURE.read_bytes()).hexdigest(),
        "a1_support_evaluation_sha256": support_hash, "identifiability_audit": audit,
        "per_seed": per_seed, "learner_training_data": "synthetic anonymized structural corpus only; no A1 skill instances",
        "learner_visible": ["A1 semi-natural support", "serialized P0 IR", "held-out request", "public tool schema"],
        "evaluator_only": ["P0 truth IR", "skill IDs", "fit-free audit labels", "expected decisions"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged",
        "classification_if_failure": ["optimization failure", "surface memorization", "relational-learning failure",
                                       "canonicalization-from-learned-evidence failure", "shortcut execution"],
        "criteria": {"relational_recovery": 0.95, "opaque_lexical_recovery": 1.0, "canonical_equality": 0.95,
                     "semantic_collisions": 0, "serialization_identity": 1.0, "matched_action_argument": 0.95,
                     "counterfactual_action": 0.95},
    }
    training_manifest = {"seeds": list(SEEDS), "examples_per_seed": per_seed[0]["training_examples"], "source": "synthetic anonymized structural correspondence"}
    train_path = OUT / "immutable_training_spec.json"
    train_path.write_text(json.dumps(training_manifest, sort_keys=True, indent=2) + "\n")
    manifest["training_spec_sha256"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "per_seed": [{"seed": x["seed"], "conditions": x["conditions"], "metrics": x["canonical_metrics"]} for x in per_seed]}, indent=2))


if __name__ == "__main__":
    main()
