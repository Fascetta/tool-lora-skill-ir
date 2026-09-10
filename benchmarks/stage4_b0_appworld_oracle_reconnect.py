"""Stage 4-B0: oracle reconnect into a real local AppWorld slice."""
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

from benchmarks.stage4_p0_oracle_tool_policy import make_constant_ir, make_ir
from tool_lora.stage4_b0.appworld_resolver import resolve
from tool_lora.stage4_p0.contract import ArgumentBinding, GenericPolicyIR, MockRequest, PolicyBranch

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_b0_appworld_oracle_reconnect"
APPWORLD_TASK = "82e2fac_1"
TASK_IDS = (4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777)
PUBLIC_APIS = ("supervisor.show_account_passwords", "todoist.login", "todoist.show_task", "todoist.update_task", "todoist.show_projects", "todoist.show_tasks")


def b0_ir(index: int, *, counterfactual: bool = False, relation_shuffle: bool = False, lexical_value: int | None = None) -> GenericPolicyIR:
    base = make_ir(index, counterfactual=counterfactual, relation_shuffle=relation_shuffle)
    value = str(TASK_IDS[index] if lexical_value is None else lexical_value)
    branches = tuple(PolicyBranch(branch.action, (ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope"))) for branch in base.alternatives)
    return GenericPolicyIR(base.condition_key, base.condition_value, branches, (("scope", value),))


def worker(artifact: str, condition: str) -> None:
    from appworld import AppWorld

    ir = GenericPolicyIR.load(artifact)
    call = resolve(ir, MockRequest("heldout", (("priority", condition),)))
    action = call.api.rsplit(".", 1)[1]
    task_id = dict(call.arguments)["task_id"]
    code = f"""
p={{x['account_name']:x['password'] for x in apis.supervisor.show_account_passwords()}}
e='joyce-weav@gmail.com'
t=apis.todoist.login(username=e,password=p['todoist'])['access_token']
result=apis.todoist.{action}(task_id={task_id},access_token=t)
state=apis.todoist.show_task(task_id={task_id},access_token=t)
print(json.dumps({{'action':'{action}','task_id':{task_id},'result':result,'state_task_id':state.get('task_id')}}))
"""
    with AppWorld(task_id=APPWORLD_TASK, experiment_name="stage4_b0_oracle", load_ground_truth=False, raise_on_failure=True) as world:
        output = world.execute(code)
    print(json.dumps(json.loads(output.strip().splitlines()[-1]), sort_keys=True))


def execute_fresh(artifact: Path, condition: str) -> dict[str, object]:
    target, root = os.environ.get("APPWORLD_PYTHONPATH"), os.environ.get("APPWORLD_ROOT")
    if not target or not root:
        raise RuntimeError("APPWORLD_PYTHONPATH and APPWORLD_ROOT are required")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{target}:{ROOT / 'src'}:{ROOT}"
    output = subprocess.check_output([sys.executable, __file__, "--worker", str(artifact), condition], cwd=ROOT, env=environment, text=True)
    return json.loads(output.strip().splitlines()[-1])


def execute_ir(ir: GenericPolicyIR, condition: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as td:
        artifact = Path(td) / "skill_ir.json"
        ir.save(artifact)
        return execute_fresh(artifact, condition)


def main() -> None:
    if not os.environ.get("APPWORLD_PYTHONPATH") or not os.environ.get("APPWORLD_ROOT"):
        raise SystemExit("B0 requires APPWORLD_PYTHONPATH and APPWORLD_ROOT")
    truths = [b0_ir(i) for i in range(len(TASK_IDS))]
    reload_identity = 0
    for truth in truths:
        with tempfile.TemporaryDirectory() as td:
            artifact = Path(td) / "reload.json"
            truth.save(artifact)
            reload_identity += int(GenericPolicyIR.load(artifact) == truth)
    rows = []
    for index, truth in enumerate(truths):
        relation = b0_ir(index, relation_shuffle=True)
        lexical = b0_ir(index, lexical_value=TASK_IDS[(index + 1) % len(TASK_IDS)])
        default = make_constant_ir()
        constant = GenericPolicyIR(default.condition_key, default.condition_value, default.alternatives, (("scope", str(TASK_IDS[0])),))
        counter = b0_ir(index, counterfactual=True)
        controls = {"relation_shuffled": relation, "lexical_shuffled": lexical, "constant": constant}
        for surface in ("urgent", "routine"):
            request = MockRequest("heldout", (("priority", surface),))
            expected = resolve(truth, request)
            cases = [("matched", truth, expected), ("counterfactual", counter, resolve(counter, request))]
            cases.extend((name, variant, expected) for name, variant in controls.items())
            for condition, variant, target in cases:
                observed = execute_ir(variant, surface)
                rows.append({"skill": index, "surface": surface, "condition": condition,
                             "action_correct": observed["action"] == target.api.rsplit(".", 1)[1],
                             "binding_correct": int(observed["task_id"]) == int(dict(target.arguments)["task_id"]),
                             "state_correct": observed["state_task_id"] == observed["task_id"]})
    conditions = {}
    for condition in ("matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant"):
        subset = [row for row in rows if row["condition"] == condition]
        conditions[condition] = {"action_accuracy": sum(r["action_correct"] for r in subset) / len(subset),
                                 "exact_binding_accuracy": sum(r["binding_correct"] for r in subset) / len(subset),
                                 "state_accuracy": sum(r["state_correct"] for r in subset) / len(subset)}
    fixture = {"fixture_format": "stage4_b0_appworld_fixture_v1", "appworld_task": APPWORLD_TASK, "task_ids": list(TASK_IDS), "public_apis": list(PUBLIC_APIS), "skills": [ir.to_dict() for ir in truths]}
    OUT.mkdir(parents=True, exist_ok=True)
    fixture_path = OUT / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n")
    data_root = Path(os.environ["APPWORLD_ROOT"]) / "data"
    docs_hashes = {app: hashlib.sha256((data_root / "api_docs" / "function_calling" / f"{app}.json").read_bytes()).hexdigest() for app in ("supervisor", "todoist")}
    manifest = {"experiment": "stage4_b0_appworld_oracle_reconnect", "appworld_version": "0.1.3.post1", "appworld_task": APPWORLD_TASK,
                "appworld_root": os.environ["APPWORLD_ROOT"], "public_apis": list(PUBLIC_APIS), "api_doc_sha256": docs_hashes,
                "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(), "conditions": conditions, "rows": rows,
                "serialization_reload_identity": reload_identity / len(truths),
                "learner_visible": ["ordinary AppWorld API documentation", "frozen serialized IR", "request", "public schema"],
                "evaluator_only": ["hidden policy mapping", "opaque task binding assignment", "expected actions/states"],
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "python": sys.version,
                "platform": platform.platform(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "criteria": {"semantic_to_tool": 1.0, "opaque_binding": 1.0, "matched_state": 1.0, "counterfactual": 1.0, "serialization_reload": 1.0},
                "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "fixture_sha256": manifest["fixture_sha256"], "conditions": conditions}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("ARTIFACT", "CONDITION"))
    args = parser.parse_args()
    worker(*args.worker) if args.worker else main()
