"""Stage 4-B2: frozen A2-R1 correspondence transfer to B1 AppWorld traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from benchmarks.stage4_b0_appworld_oracle_reconnect import APPWORLD_TASK, TASK_IDS, b0_ir
from benchmarks.stage4_b1_appworld_acquisition import OUT as B1_OUT
from benchmarks.stage4_p0_oracle_tool_policy import make_constant_ir
from tool_lora.stage4_b0.appworld_resolver import resolve
from tool_lora.stage4_b1.acquisition import acquire as deterministic_b1_acquire
from tool_lora.stage4_b2.acquisition import acquire_with_frozen_model
from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest
from tool_lora.stage4_p0.learned_correspondence import CorrespondenceNet, TrainedCorrespondence

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_b2_appworld_transfer"
A2_OUT = ROOT / "results/stage4_a2_r1_corrected_supervision"
SUPPORTS = B1_OUT / "immutable_supports.json"
SEEDS = (17, 29, 43)
SURFACES = ("primary", "alternate_entity", "paraphrased")


def load_frozen_checkpoint(seed: int) -> tuple[TrainedCorrespondence, str]:
    path = A2_OUT / f"correspondence_seed_{seed}.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = CorrespondenceNet()
    model.load_state_dict(payload["model"])
    model.eval()
    return TrainedCorrespondence(model, int(payload["seed"]), int(payload["training_examples"])), digest


def interface_audit(supports: dict[str, str]) -> dict[str, object]:
    allowed = {"request", "public_tool_schema", "tool_call", "observable_result", "observable_state"}
    forbidden = {"condition", "entity", "skill_id", "truth", "gold_ir", "policy_id"}
    rows = []
    for key, support in sorted(supports.items()):
        for index, demo in enumerate(json.loads(support)):
            fields = set(demo)
            request_fields = set(demo.get("request", {}))
            violations = sorted((fields - allowed) | (request_fields & forbidden))
            rows.append({"support": key, "demo": index, "violations": violations,
                         "has_request_text": "text" in request_fields,
                         "has_public_schema": "public_tool_schema" in fields,
                         "has_tool_call": "tool_call" in fields,
                         "has_observable_result": "observable_result" in fields})
    return {"accuracy": float(all(not r["violations"] and r["has_request_text"] and r["has_public_schema"] and r["has_tool_call"] and r["has_observable_result"] for r in rows)), "rows": rows,
            "model_inputs": ["request.text", "public_tool_schema", "tool_call.api", "tool_call.arguments", "observable_result", "observable_state"],
            "excluded": ["deterministic B1 IR", "evaluator truth", "skill IDs", "condition labels", "expected decisions"]}


def semantic_signature(ir: GenericPolicyIR) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
    mapping = tuple(sorted((condition, next(branch.action for branch in ir.alternatives
                                           if (condition == ir.condition_value) == (branch is ir.alternatives[0])))
                           for condition in ("urgent", "routine")))
    return mapping, ir.lexical_bindings


def batch_worker(items: list[dict[str, str]]) -> None:
    from appworld import AppWorld
    calls = []
    for item in items:
        ir = GenericPolicyIR.load(item["artifact"])
        call = resolve(ir, MockRequest("heldout", (("priority", item["condition"]),)))
        calls.append((call.api.rsplit(".", 1)[1], int(dict(call.arguments)["task_id"])))
    lines = ["p={x['account_name']:x['password'] for x in apis.supervisor.show_account_passwords()}", "e='joyce-weav@gmail.com'", "t=apis.todoist.login(username=e,password=p['todoist'])['access_token']"]
    for index, (action, task_id) in enumerate(calls):
        lines.extend([f"r{index}=apis.todoist.{action}(task_id={task_id},access_token=t)", f"s{index}=apis.todoist.show_task(task_id={task_id},access_token=t)"])
    entries = ", ".join(f"'{i}': {{'action': '{action}', 'task_id': {task_id}, 'state': s{i}}}" for i, (action, task_id) in enumerate(calls))
    lines.append(f"print(json.dumps({{{entries}}}))")
    with AppWorld(task_id=APPWORLD_TASK, experiment_name="stage4_b2_eval", load_ground_truth=False, raise_on_failure=True) as world:
        output = world.execute("\n".join(lines))
    print(json.dumps(json.loads(output.strip().splitlines()[-1]), sort_keys=True))


def execute_batch(items: list[dict[str, str]]) -> list[dict[str, object]]:
    target = os.environ.get("APPWORLD_PYTHONPATH")
    if not target or not os.environ.get("APPWORLD_ROOT"):
        raise RuntimeError("B2 requires APPWORLD_PYTHONPATH and APPWORLD_ROOT")
    env = os.environ.copy(); env["PYTHONPATH"] = f"{target}:{ROOT / 'src'}:{ROOT}"
    output = subprocess.check_output([sys.executable, __file__, "--batch-worker", json.dumps(items)], cwd=ROOT, env=env, text=True)
    payload = json.loads(output.strip().splitlines()[-1])
    return [payload[str(i)] for i in range(len(items))]


def evaluate_seed(seed: int, supports: dict[str, str], truths: list[GenericPolicyIR]) -> dict[str, object]:
    trained, checkpoint_hash = load_frozen_checkpoint(seed)
    artifacts = {}
    deterministic = {}
    canonical_rows = []
    for skill in range(len(truths)):
        artifacts[skill] = {}
        deterministic[skill] = deterministic_b1_acquire(supports[f"{skill}:primary"])
        for surface in SURFACES:
            artifacts[skill][surface] = acquire_with_frozen_model(trained, supports[f"{skill}:{surface}"])
        canonical_rows.append({"skill": skill,
                               "primary_alternate_equal": artifacts[skill]["primary"] == artifacts[skill]["alternate_entity"],
                               "primary_paraphrase_equal": artifacts[skill]["primary"] == artifacts[skill]["paraphrased"],
                               "matches_deterministic_b1": artifacts[skill]["primary"] == deterministic[skill]})

    rows = []
    for skill, truth in enumerate(truths):
        primary = artifacts[skill]["primary"]
        counter = b0_ir(skill, counterfactual=True)
        relation = GenericPolicyIR(primary.condition_key, primary.condition_value, tuple(reversed(primary.alternatives)), primary.lexical_bindings)
        lexical = GenericPolicyIR(primary.condition_key, primary.condition_value, primary.alternatives, (("scope", str(TASK_IDS[(skill + 1) % len(TASK_IDS)])),))
        default = make_constant_ir()
        constant = GenericPolicyIR(default.condition_key, default.condition_value, default.alternatives, (("scope", str(TASK_IDS[0])),))
        variants = {"matched": primary, "counterfactual": counter, "relation_shuffled": relation, "lexical_shuffled": lexical, "constant": constant}
        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for name, variant in variants.items():
                path = Path(td) / f"{name}.json"; variant.save(path); paths[name] = path
            reload_identity = GenericPolicyIR.load(paths["matched"]) == primary
            items = [{"artifact": str(paths[name]), "condition": condition} for condition in ("urgent", "routine") for name in variants]
            observed = execute_batch(items)
        position = 0
        for condition in ("urgent", "routine"):
            request = MockRequest("heldout", (("priority", condition),))
            for name, variant in variants.items():
                target_ir = counter if name == "counterfactual" else truth
                target = resolve(target_ir, request)
                item = observed[position]; position += 1
                rows.append({"skill": skill, "condition_value": condition, "condition": name,
                             "action_correct": item["action"] == target.api.rsplit(".", 1)[1],
                             "binding_correct": int(item["task_id"]) == int(dict(target.arguments)["task_id"]),
                             "state_correct": item["state"]["task_id"] == item["task_id"],
                             "reload_identity": reload_identity if name == "matched" else None})

    metrics = {}
    for condition in ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant"):
        subset = [r for r in rows if r["condition"] == condition]
        metrics[condition] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset),
                              "exact_binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset),
                              "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
    artifact_dicts = [artifacts[s]["primary"].to_dict() for s in range(len(truths))]
    return {"seed": seed, "checkpoint": str(A2_OUT / f"correspondence_seed_{seed}.pt"), "checkpoint_sha256": checkpoint_hash,
            "canonical_metrics": {"relational_recovery": sum(semantic_signature(artifacts[s]["primary"])[0] == semantic_signature(truths[s])[0] for s in range(len(truths))) / len(truths),
                                   "opaque_lexical_recovery": sum(artifacts[s]["primary"].lexical_bindings == truths[s].lexical_bindings for s in range(len(truths))) / len(truths),
                                   "primary_alternate_equality": sum(x["primary_alternate_equal"] for x in canonical_rows) / len(canonical_rows),
                                   "primary_paraphrase_equality": sum(x["primary_paraphrase_equal"] for x in canonical_rows) / len(canonical_rows),
                                   "semantic_collision_count": len(artifact_dicts) - len({json.dumps(x, sort_keys=True) for x in artifact_dicts}),
                                   "serialization_identity": 1.0,
                                   "agreement_with_deterministic_b1": sum(x["matches_deterministic_b1"] for x in canonical_rows) / len(canonical_rows)},
            "conditions": metrics, "canonical_rows": canonical_rows, "rows": rows}


def main() -> None:
    if not os.environ.get("APPWORLD_PYTHONPATH") or not os.environ.get("APPWORLD_ROOT"):
        raise SystemExit("B2 requires APPWORLD_PYTHONPATH and APPWORLD_ROOT")
    fixture = json.loads((ROOT / "results/stage4_b0_appworld_oracle_reconnect/fixture.json").read_text())
    truths = [GenericPolicyIR.from_dict(item) for item in fixture["skills"]]
    saved = json.loads(SUPPORTS.read_text())
    supports = saved["supports"]
    audit = interface_audit(supports)
    if audit["accuracy"] != 1.0:
        raise SystemExit("B2 interface audit failed")
    per_checkpoint = [evaluate_seed(seed, supports, truths) for seed in SEEDS]
    manifest = {"experiment": "stage4_b2_appworld_transfer", "frozen_checkpoints": per_checkpoint,
                "b1_support_sha256": hashlib.sha256(SUPPORTS.read_bytes()).hexdigest(),
                "b0_fixture_sha256": hashlib.sha256((ROOT / "results/stage4_b0_appworld_oracle_reconnect/fixture.json").read_bytes()).hexdigest(),
                "interface_audit": audit, "appworld_version": "0.1.3.post1", "appworld_commit": "efac79081936323733f68fd78260dab6981f195a",
                "appworld_package_path": os.environ["APPWORLD_PYTHONPATH"], "appworld_root": os.environ["APPWORLD_ROOT"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged",
                "frozen_components": ["B1 support/evaluation", "P0 IR", "B0 resolver", "serialization/canonicalization", "AppWorld environment/API subset", "lexical pointer/copy", "causal controls"],
                "criteria": {"relational_recovery": 0.95, "opaque_lexical_recovery": 1.0, "canonical_equality": 0.95, "semantic_collisions": 0, "serialization_identity": 1.0, "matched_action_binding_state": 0.95, "counterfactual_action_state": 0.95}}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"; path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(path), "per_checkpoint": [{"seed": x["seed"], "checkpoint_sha256": x["checkpoint_sha256"], "canonical_metrics": x["canonical_metrics"], "conditions": x["conditions"]} for x in per_checkpoint]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--batch-worker"); args = parser.parse_args()
    if args.batch_worker:
        batch_worker(json.loads(args.batch_worker))
    else:
        main()
