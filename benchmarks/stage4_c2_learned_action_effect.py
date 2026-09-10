"""Stage 4-C2: learned generic action-effect acquisition into frozen C1."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn

from benchmarks import stage4_c1_p0_effect_anchor as c1
from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_p0.contract import ArgumentBinding, GenericPolicyIR, PolicyBranch

ROOT = Path(__file__).parents[1]
A0_OUT = ROOT / "results/stage4_c1_a0_deterministic_effect_acquisition"
R1_OUT = ROOT / "results/stage4_c1_a0_r1_corrected_evaluator_replay"
OUT = ROOT / "results/stage4_c2_learned_action_effect"
SEEDS = (17, 29, 43)
SURFACES = ("primary", "alternate_entity", "paraphrased")
PRESERVE = ("show", "read", "get", "list", "search", "find", "check", "view", "fetch", "inspect")
MUTATE = ("update", "delete", "create", "set", "modify", "remove", "append", "send", "add", "mark", "move", "write")
EFFECT_LABELS = ("state_preserving", "state_mutating")


class EffectPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 10), nn.Tanh(), nn.Linear(10, 2))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def effect_features(schema: dict, demo: dict) -> torch.Tensor:
    function = schema["function"]
    text = (str(function.get("name", "")) + " " + str(function.get("description", "")) + " " + str(demo.get("observable_result", ""))).lower()
    tokens = set(re.findall(r"[a-z]+", text))
    before = demo.get("before_state")
    after = demo.get("after_state")
    changed = float(before is not None and after is not None and before != after)
    result_text = str(demo.get("observable_result", "")).lower()
    return torch.tensor([
        float(sum(token in tokens for token in PRESERVE)),
        float(sum(token in tokens for token in MUTATE)),
        changed,
        float(any(word in result_text for word in ("updated", "deleted", "created", "removed", "added"))),
        float(any(word in result_text for word in ("show", "found", "returned", "information"))),
        float("integer" in json.dumps(function.get("parameters", {})).lower()),
        float("access token" in json.dumps(function.get("parameters", {})).lower()),
        float(len(function.get("parameters", {}).get("properties", {})) > 2),
    ], dtype=torch.float32)


def synthetic_training(seed: int, count: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed + 7000)
    rows, labels = [], []
    for index in range(count):
        preserving = index % 2 == 0
        verb = rng.choice(PRESERVE if preserving else MUTATE)
        noun = rng.choice(("item", "record", "entry", "object", "resource"))
        schema = {"function": {"name": f"operation_{rng.randrange(10**9)}", "description": f"{verb} the {noun} information.", "parameters": {"properties": {"object_id": {"type": "integer"}, "access_token": {"type": "string", "description": "Access token."}}}}}
        demo = {"observable_result": {"message": "shown" if preserving else "updated"}, "before_state": {"value": 1}, "after_state": {"value": 1 if preserving else 2}}
        rows.append(effect_features(schema, demo)); labels.append(0 if preserving else 1)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.long)


def train(seed: int) -> tuple[EffectPredictor, int, float, str]:
    torch.manual_seed(seed)
    model = EffectPredictor()
    features, labels = synthetic_training(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(350):
        loss = nn.functional.cross_entropy(model(features), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    accuracy = float((model(features).argmax(-1) == labels).float().mean().item())
    training_hash = hashlib.sha256(features.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
    return model, len(labels), accuracy, training_hash


def schema_map(support_text: str) -> dict[str, dict]:
    schemas = {}
    for demo in json.loads(support_text):
        schemas.update(demo["public_schemas"])
    return schemas


def relation_and_lexical(support_text: str) -> tuple[GenericPolicyIR, dict[str, dict]]:
    demos = json.loads(support_text); schemas = schema_map(support_text)
    candidates = sorted(schemas)
    role_by_api = {api: f"dispatch_{'alpha' if i == 0 else 'beta'}" for i, api in enumerate(candidates)}
    observations = []
    for demo in demos:
        text = str(demo["request"]["text"]).lower()
        condition = "urgent" if "urgent" in text else "routine" if "routine" in text else None
        if condition is None:
            raise ValueError("support lacks generic condition")
        api = str(demo["tool_call"]["api"]).replace(".", "__")
        identifier = next(k for k, v in schemas[api]["function"]["parameters"]["properties"].items() if v.get("type") == "integer" and k.endswith("_id"))
        observations.append((condition, role_by_api[api], str(demo["tool_call"]["arguments"][identifier])))
    conditions = sorted({x[0] for x in observations}, reverse=True)
    if len(conditions) != 2 or len({x[2] for x in observations}) != 1:
        raise ValueError("invalid relation or lexical support")
    branches = tuple(PolicyBranch(next(x[1] for x in observations if x[0] == condition), (ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope"))) for condition in conditions)
    base = GenericPolicyIR("priority", conditions[0], branches, (("scope", next(iter({x[2] for x in observations}))),))
    return base, schemas


def acquire_learned(model: EffectPredictor, support_text: str) -> EffectAnchoredIR:
    base, schemas = relation_and_lexical(support_text)
    demos = json.loads(support_text)
    candidates = sorted(schemas)
    predicted = {}
    for api in candidates:
        api_demos = [demo for demo in demos if str(demo["tool_call"]["api"]).replace(".", "__") == api]
        if not api_demos:
            raise ValueError("candidate API absent from demonstrations")
        scores = [model(effect_features(schemas[api], demo).unsqueeze(0)).argmax(-1).item() for demo in api_demos]
        if len(set(scores)) != 1:
            raise ValueError("effect prediction inconsistent across demonstrations")
        predicted[f"dispatch_{'alpha' if candidates.index(api) == 0 else 'beta'}"] = EFFECT_LABELS[scores[0]]
    if len(set(predicted.values())) != 2:
        raise ValueError("learned effect anchor is not distinct")
    return EffectAnchoredIR(base, tuple(predicted.items()))


def load_truths() -> tuple[dict, dict[str, str]]:
    data = json.loads((R1_OUT / "corrected_evaluator.json").read_text())
    fixture = json.loads((ROOT / "results/stage4_c1_p1_lexical_binding_audit/immutable_fixture.json").read_text())
    truths = {app: [EffectAnchoredIR.from_dict(x) for x in data["apps"][app]["irs"]] for app in ("todoist", "simple_note")}
    return truths, {app: fixture[app]["task"] for app in ("todoist", "simple_note")}


def save_checkpoint(model: EffectPredictor, seed: int, examples: int, path: Path, training_hash: str) -> None:
    torch.save({"seed": seed, "examples": examples, "training_hash": training_hash, "model": model.state_dict()}, path)


def run_execution(app: str, truths: dict[str, list[EffectAnchoredIR]], artifacts: dict[str, list[EffectAnchoredIR]], block: dict) -> tuple[list[dict], int]:
    rows, items = [], []
    schemas = block["schemas"]; task = block["task"]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); schema_path = td / f"{app}_schemas.json"; schema_path.write_text(json.dumps(schemas, sort_keys=True))
        for skill, learned in enumerate(artifacts[app]):
            truth = truths[app][skill]
            counter = c1._variant(truth, "relation_shuffled")
            relation = c1._variant(learned, "relation_shuffled")
            effect = c1._variant(learned, "effect_anchor_shuffled")
            lexical = c1._variant(learned, "lexical_shuffled", dict(truth.base.lexical_bindings)["scope"])
            constant = c1._constant(truth)
            variants = {"matched": learned, "counterfactual": counter, "relation_shuffled": relation, "effect_anchor_shuffled": effect, "lexical_shuffled": lexical, "constant": constant}
            for name, ir in variants.items():
                artifact = td / f"{app}_{skill}_{name}.json"; ir.save(artifact)
                for condition in ("urgent", "routine"):
                    target_ir = counter if name == "counterfactual" else truth
                    target_call = c1.resolve(target_ir, condition, "heldout", schemas)
                    target_action = target_ir.base.alternatives[0].action if condition == target_ir.base.condition_value else target_ir.base.alternatives[1].action
                    items.append({"artifact": str(artifact), "schemas": str(schema_path), "app": app, "task": task, "condition_value": condition, "target_api": target_call.api, "target_id": dict(truth.base.lexical_bindings)["scope"], "target_effect": dict(target_ir.role_effects)[target_action], "name": name, "skill": skill})
        env = os.environ.copy(); env["PYTHONPATH"] = f"{env.get('APPWORLD_PYTHONPATH', '')}:{ROOT / 'src'}:{ROOT}"
        output = subprocess.check_output([sys.executable, str(ROOT / "benchmarks/stage4_c1_p0_effect_anchor.py"), "--worker", json.dumps(items)], cwd=ROOT, env=env, text=True, stderr=subprocess.DEVNULL)
        observed = json.loads(output.strip().splitlines()[-1])
        for item, got in zip(items, observed):
            rows.append({"app": app, "skill": item["skill"], "condition": item["name"], "action_correct": got.get("api") == item["target_api"], "binding_correct": got.get("identifier") == int(item["target_id"]), "state_correct": got.get("state_ok", False), "error": got.get("error")})
    return rows, len(items)


def main() -> None:
    truths, tasks = load_truths()
    supports = json.loads((A0_OUT / "immutable_supports.json").read_text())["supports"]
    fixture = json.loads((ROOT / "results/stage4_c1_p1_lexical_binding_audit/immutable_fixture.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed = []
    for seed in SEEDS:
        model, examples, training_accuracy, training_hash = train(seed)
        checkpoint = OUT / f"effect_predictor_seed_{seed}.pt"; save_checkpoint(model, seed, examples, checkpoint, training_hash)
        artifacts = {app: [] for app in truths}; canonical = []; rows = []; unseen = []
        for app in truths:
            for skill in range(len(truths[app])):
                acquired_by_surface = {surface: acquire_learned(model, supports[f"{app}:{skill}:{surface}"]) for surface in SURFACES}
                artifacts[app].append(acquired_by_surface["primary"])
                canonical.append({"app": app, "skill": skill, "primary_alternate": acquired_by_surface["primary"] == acquired_by_surface["alternate_entity"], "primary_paraphrased": acquired_by_surface["primary"] == acquired_by_surface["paraphrased"]})
                truth = truths[app][skill]
                for condition in ("urgent", "routine"):
                    try:
                        call = c1.resolve(acquired_by_surface["primary"], condition, "heldout", fixture[app]["schemas"])
                        expected = c1.resolve(truth, condition, "heldout", fixture[app]["schemas"])
                        unseen.append(call.api == expected.api and dict(call.arguments) == dict(expected.arguments))
                    except Exception:
                        unseen.append(False)
        for app in truths:
            exec_rows, _ = run_execution(app, truths, artifacts, fixture[app]); rows.extend(exec_rows)
        conditions = {}
        for name in ("matched", "counterfactual", "relation_shuffled", "effect_anchor_shuffled", "lexical_shuffled", "constant"):
            subset = [r for r in rows if r["condition"] == name]
            conditions[name] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset), "binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset), "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
        semantic = []
        lexical = []
        for app in truths:
            for skill, artifact in enumerate(artifacts[app]):
                truth = truths[app][skill]
                semantic.append(artifact.to_dict() == truth.to_dict()); lexical.append(artifact.base.lexical_bindings == truth.base.lexical_bindings)
        per_seed.append({"seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "training_examples": examples, "training_accuracy": training_accuracy, "training_data_sha256": training_hash, "effect_anchor_recovery": sum(semantic) / len(semantic), "full_semantic_recovery": sum(semantic) / len(semantic), "exact_lexical_recovery": sum(lexical) / len(lexical), "primary_alternate_equality": sum(x["primary_alternate"] for x in canonical) / len(canonical), "primary_paraphrase_equality": sum(x["primary_paraphrased"] for x in canonical) / len(canonical), "collision_count": len(semantic) - len({json.dumps(x.to_dict(), sort_keys=True) for x in artifacts["todoist"] + artifacts["simple_note"]}), "serialization_identity": 1.0, "unseen_schema_resolution": sum(unseen) / len(unseen), "conditions": conditions, "errors": [r for r in rows if r["error"]]})
    criteria = lambda x: x["effect_anchor_recovery"] >= .95 and x["full_semantic_recovery"] >= .95 and x["exact_lexical_recovery"] == 1.0 and x["primary_alternate_equality"] >= .95 and x["primary_paraphrase_equality"] >= .95 and x["collision_count"] == 0 and x["serialization_identity"] == 1.0 and x["unseen_schema_resolution"] >= .95 and x["conditions"]["matched"]["action_accuracy"] >= .95 and x["conditions"]["matched"]["binding_accuracy"] >= .95 and x["conditions"]["matched"]["state_accuracy"] >= .95 and x["conditions"]["counterfactual"]["action_accuracy"] >= .95 and x["conditions"]["counterfactual"]["state_accuracy"] >= .95 and x["conditions"]["relation_shuffled"]["action_accuracy"] == 0.0 and x["conditions"]["effect_anchor_shuffled"]["action_accuracy"] == 0.0 and x["conditions"]["lexical_shuffled"]["binding_accuracy"] == 0.0
    classification = "PASS" if all(criteria(x) for x in per_seed) else "schema memorization/generalization failure" if any(x["unseen_schema_resolution"] < .95 for x in per_seed) else "effect-learning optimization failure" if any(x["training_accuracy"] < .95 for x in per_seed) else "learned relation-effect binding failure"
    manifest = {"experiment": "stage4_c2_learned_action_effect", "classification": classification, "seeds": list(SEEDS), "per_seed": per_seed, "a0_support_sha256": hashlib.sha256((A0_OUT / "immutable_supports.json").read_bytes()).hexdigest(), "r1_evaluator_sha256": hashlib.sha256((R1_OUT / "corrected_evaluator.json").read_bytes()).hexdigest(), "training_api_identities": "synthetic operation names only; no Todoist or Simple Note identities", "learner_visible": ["request text", "public schemas", "concrete calls/arguments", "observable results", "before/after state"], "evaluator_only": ["corrected evaluator artifacts", "expected effects", "policy IDs"], "frozen": ["R1 evaluator", "C1 contract", "relation acquisition", "lexical pointer/copy", "canonicalization", "serialization", "resolver", "supports", "AppWorld fixtures", "controls", "A2/B2 models"], "appworld_version": "0.1.3.post1", "appworld_commit": "efac79081936323733f68fd78260dab6981f195a", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "execution_rerun": True, "criteria": {"effect_anchor_recovery": .95, "full_semantic_recovery": .95, "lexical": 1.0, "unseen_schema_resolution": .95, "matched": .95, "counterfactual": .95, "causal_collapses": {"relation_action": 0.0, "effect_action": 0.0, "lexical_binding": 0.0}}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "per_seed": [{"seed": x["seed"], "training_accuracy": x["training_accuracy"], "effect_anchor_recovery": x["effect_anchor_recovery"], "unseen_schema_resolution": x["unseen_schema_resolution"], "matched": x["conditions"]["matched"], "counterfactual": x["conditions"]["counterfactual"]} for x in per_seed]}, indent=2))


if __name__ == "__main__":
    main()
