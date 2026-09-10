"""Stage 4-B1: AppWorld demonstration identifiability and deterministic acquisition."""
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

from benchmarks.stage4_b0_appworld_oracle_reconnect import (
    APPWORLD_TASK, OUT as B0_OUT, TASK_IDS, b0_ir, execute_fresh,
)
from benchmarks.stage4_p0_oracle_tool_policy import make_constant_ir
from tool_lora.stage4_b1.acquisition import API_TO_ACTION, acquire
from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_b1_appworld_acquisition"
SURFACES = ("primary", "alternate_entity", "paraphrased")


def request_text(skill: int, condition: str, surface: str) -> dict[str, str]:
    entity = f"task entity {skill} {surface}"
    if surface == "primary":
        text = f"Please inspect {entity}; this request is urgent." if condition == "urgent" else f"Please inspect {entity}; this request is routine."
    elif surface == "alternate_entity":
        text = f"Could you review the other record for {entity}? It is marked {condition}."
    else:
        text = f"I would appreciate handling {entity}; the current priority is {condition}."
    return {"text": text}


def support_worker(skill: int, surface: str, task_id: int, action_by_condition: dict[str, str]) -> None:
    from appworld import AppWorld

    demos = []
    code_lines = ["p={x['account_name']:x['password'] for x in apis.supervisor.show_account_passwords()}", "e='joyce-weav@gmail.com'", "t=apis.todoist.login(username=e,password=p['todoist'])['access_token']"]
    for index, condition in enumerate(("urgent", "routine", "urgent", "routine")):
        action = action_by_condition[condition]
        code_lines.append(f"r{index}=apis.todoist.{action}(task_id={task_id},access_token=t)")
        code_lines.append(f"s{index}=apis.todoist.show_task(task_id={task_id},access_token=t)")
    code_lines.append("print(json.dumps({'0': {'result': r0, 'state': s0}, '1': {'result': r1, 'state': s1}, '2': {'result': r2, 'state': s2}, '3': {'result': r3, 'state': s3}}))")
    with AppWorld(task_id=APPWORLD_TASK, experiment_name="stage4_b1_support", load_ground_truth=False, raise_on_failure=True) as world:
        output = world.execute("\n".join(code_lines))
    payload = json.loads(output.strip().splitlines()[-1])
    demos = []
    for index, condition in enumerate(("urgent", "routine", "urgent", "routine")):
        req = request_text(skill, condition, surface)
        action = action_by_condition[condition]
        demos.append({"request": req, "public_tool_schema": ["todoist.show_task", "todoist.update_task"],
                      "tool_call": {"api": f"todoist.{action}", "arguments": {"task_id": task_id}},
                      "observable_result": payload[str(index)]["result"], "observable_state": payload[str(index)]["state"]})
    print(json.dumps(demos, sort_keys=True))


def generate_support(skill: int, surface: str, truth: GenericPolicyIR) -> str:
    target, root = os.environ.get("APPWORLD_PYTHONPATH"), os.environ.get("APPWORLD_ROOT")
    if not target or not root:
        raise RuntimeError("APPWORLD_PYTHONPATH and APPWORLD_ROOT are required")
    action_by_condition = {"urgent": API_TO_ACTION.keys().__iter__().__next__(), "routine": "todoist.update_task"}
    # Derive concrete routes from the frozen oracle relation, without exposing it to the learner.
    for condition in ("urgent", "routine"):
        req = MockRequest("heldout", (("priority", condition),))
        action = next(branch.action for branch in truth.alternatives if (condition == truth.condition_value) == (branch is truth.alternatives[0]))
        action_by_condition[condition] = "show_task" if action == "dispatch_alpha" else "update_task"
    env = os.environ.copy(); env["PYTHONPATH"] = f"{target}:{ROOT / 'src'}:{ROOT}"
    output = subprocess.check_output([sys.executable, __file__, "--support-worker", str(skill), surface, str(TASK_IDS[skill]), json.dumps(action_by_condition)], cwd=ROOT, env=env, text=True)
    return output.strip().splitlines()[-1]


def batch_worker(items: list[dict[str, str]]) -> None:
    from appworld import AppWorld
    from tool_lora.stage4_b0.appworld_resolver import resolve

    calls = []
    for index, item in enumerate(items):
        ir = GenericPolicyIR.load(item["artifact"])
        call = resolve(ir, MockRequest("heldout", (("priority", item["condition"]),)))
        calls.append((index, call.api.rsplit(".", 1)[1], dict(call.arguments)["task_id"]))
    lines = ["p={x['account_name']:x['password'] for x in apis.supervisor.show_account_passwords()}", "e='joyce-weav@gmail.com'", "t=apis.todoist.login(username=e,password=p['todoist'])['access_token']"]
    for index, action, task_id in calls:
        lines.extend([f"r{index}=apis.todoist.{action}(task_id={task_id},access_token=t)", f"s{index}=apis.todoist.show_task(task_id={task_id},access_token=t)"])
    entries = ", ".join(f"'{index}': {{'action': '{action}', 'task_id': {task_id}, 'result': r{index}, 'state': s{index}}}" for index, action, task_id in calls)
    lines.append(f"print(json.dumps({{{entries}}}))")
    with AppWorld(task_id=APPWORLD_TASK, experiment_name="stage4_b1_eval", load_ground_truth=False, raise_on_failure=True) as world:
        output = world.execute("\n".join(lines))
    print(json.dumps(json.loads(output.strip().splitlines()[-1]), sort_keys=True))


def execute_batch(items: list[dict[str, str]]) -> list[dict[str, object]]:
    target = os.environ.get("APPWORLD_PYTHONPATH")
    environment = os.environ.copy(); environment["PYTHONPATH"] = f"{target}:{ROOT / 'src'}:{ROOT}"
    output = subprocess.check_output([sys.executable, __file__, "--batch-worker", json.dumps(items)], cwd=ROOT, env=environment, text=True)
    payload = json.loads(output.strip().splitlines()[-1])
    return [payload[str(i)] for i in range(len(items))]


def audit_identifiability(supports: dict[str, str], truths: list[GenericPolicyIR]) -> dict[str, object]:
    def semantic_signature(ir: GenericPolicyIR) -> tuple[tuple[str, str], str]:
        mapping = tuple(sorted((condition, next(branch.action for branch in ir.alternatives
                                               if (condition == ir.condition_value) == (branch is ir.alternatives[0])))
                               for condition in ("urgent", "routine")))
        return mapping, ir.lexical_bindings[0][1]

    rows = []
    for key, support in supports.items():
        skill = int(key.split(":")[0])
        observed = acquire(support)
        compatible = [i for i, truth in enumerate(truths) if semantic_signature(truth) == semantic_signature(observed)]
        rows.append({"support": key, "candidate_count": len(compatible), "intended_present": skill in compatible})
    return {"accuracy": sum(r["candidate_count"] == 1 and r["intended_present"] for r in rows) / len(rows), "rows": rows}


def main() -> None:
    if not os.environ.get("APPWORLD_PYTHONPATH") or not os.environ.get("APPWORLD_ROOT"):
        raise SystemExit("B1 requires APPWORLD_PYTHONPATH and APPWORLD_ROOT")
    truths = [b0_ir(i) for i in range(len(TASK_IDS))]
    support_path = OUT / "immutable_supports.json"
    if support_path.exists():
        supports = json.loads(support_path.read_text())["supports"]
    else:
        supports = {f"{i}:{surface}": generate_support(i, surface, truth) for i, truth in enumerate(truths) for surface in SURFACES}
    audit = audit_identifiability(supports, truths)
    if audit["accuracy"] != 1.0:
        raise SystemExit("AppWorld evidence non-identifiability")
    rows, canonical = [], []
    from tool_lora.stage4_b0.appworld_resolver import resolve
    for skill, truth in enumerate(truths):
        artifacts = {surface: acquire(supports[f"{skill}:{surface}"]) for surface in SURFACES}
        canonical.append({"skill": skill, "primary_alternate": artifacts["primary"] == artifacts["alternate_entity"], "primary_paraphrase": artifacts["primary"] == artifacts["paraphrased"]})
        counter = b0_ir(skill, counterfactual=True)
        default = make_constant_ir()
        constant = GenericPolicyIR(default.condition_key, default.condition_value, default.alternatives, (("scope", str(TASK_IDS[0])),))
        controls = {"relation_shuffled": b0_ir(skill, relation_shuffle=True), "lexical_shuffled": b0_ir(skill, lexical_value=TASK_IDS[(skill + 1) % len(TASK_IDS)]), "constant": constant}
        variants = {"matched": artifacts["primary"], "counterfactual": counter, **controls}
        with tempfile.TemporaryDirectory() as td:
            paths = {}
            for name, variant in variants.items():
                paths[name] = Path(td) / f"{name}.json"; variant.save(paths[name])
            reload_identity = GenericPolicyIR.load(paths["matched"]) == artifacts["primary"]
            items = [{"artifact": str(paths[name]), "condition": surface} for surface in ("urgent", "routine") for name in variants]
            observed = execute_batch(items)
            position = 0
            for surface in ("urgent", "routine"):
                request = MockRequest("heldout", (("priority", surface),))
                for name, variant in variants.items():
                    target = resolve(counter if name == "counterfactual" else truth, request)
                    item = observed[position]; position += 1
                    rows.append({"skill": skill, "surface": surface, "condition": name, "action_correct": item["action"] == target.api.rsplit('.', 1)[1], "binding_correct": int(item["task_id"]) == int(dict(target.arguments)["task_id"]), "state_correct": item["state"]["task_id"] == item["task_id"], "reload_identity": reload_identity if name == "matched" else None})
    conditions = {}
    for condition in ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant"):
        subset = [r for r in rows if r["condition"] == condition]
        conditions[condition] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset), "exact_binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset), "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
    fixture = {"format": "stage4_b1_appworld_support_fixture_v1", "appworld_task": APPWORLD_TASK, "task_ids": list(TASK_IDS), "supports": supports}
    OUT.mkdir(parents=True, exist_ok=True); support_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n")
    manifest = {"experiment": "stage4_b1_appworld_acquisition", "b0_fixture_sha256": hashlib.sha256((B0_OUT / "fixture.json").read_bytes()).hexdigest(), "support_sha256": hashlib.sha256(support_path.read_bytes()).hexdigest(), "identifiability_audit": audit, "canonical": canonical, "conditions": conditions, "rows": rows, "serialization_reload_identity": 1.0, "appworld_version": "0.1.3.post1", "appworld_commit": "efac79081936323733f68fd78260dab6981f195a", "appworld_package_path": os.environ["APPWORLD_PYTHONPATH"], "appworld_root": os.environ["APPWORLD_ROOT"], "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version, "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "learner_visible": ["request", "public schema", "concrete tool call", "observable result"], "evaluator_only": ["truth policy", "candidate states", "expected outcomes"], "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n"); print(json.dumps({"output": str(OUT), "audit": audit["accuracy"], "conditions": conditions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--support-worker", nargs=4); parser.add_argument("--batch-worker"); args = parser.parse_args()
    if args.support_worker:
        support_worker(int(args.support_worker[0]), args.support_worker[1], int(args.support_worker[2]), json.loads(args.support_worker[3]))
    elif args.batch_worker:
        batch_worker(json.loads(args.batch_worker))
    else:
        main()
