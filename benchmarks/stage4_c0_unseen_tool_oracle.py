"""Stage 4-C0: oracle transfer to an unseen AppWorld application/API set."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from benchmarks.stage4_b0_appworld_oracle_reconnect import b0_ir
from tool_lora.stage4_b0.appworld_resolver import resolve
from tool_lora.stage4_p0.contract import GenericPolicyIR, MockRequest

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_c0_unseen_tool_oracle"
APPWORLD_ROOT = Path(os.environ.get("APPWORLD_ROOT", "/tmp/appworld-root-b0-ohMzCT"))
TASK = "cf6abd2_2"
APPLICATION = "simple_note"
API_DOC = APPWORLD_ROOT / "data/api_docs/function_calling/simple_note.json"
API_NAMES = ("simple_note__show_note", "simple_note__delete_note")
NOTE_IDS = tuple(range(3042, 3050))
PREVIOUS_APIS = (
    "todoist.show_task", "todoist.update_task",
    "supervisor.show_account_passwords", "todoist.login",
)


def selected_schema() -> dict[str, object]:
    if API_DOC.exists():
        docs = json.loads(API_DOC.read_text())
        return {item["function"]["name"]: item for item in docs if item["function"]["name"] in API_NAMES}
    fixture = json.loads((OUT / "immutable_fixture.json").read_text())
    return fixture["public_schema"]


def c0_ir(index: int, *, counterfactual: bool = False, relation_shuffle: bool = False,
          lexical_value: int | None = None) -> GenericPolicyIR:
    # The P0 relation remains generic. Only the evaluator's abstract action
    # names and exact opaque literal are stored in the frozen contract.
    show_first = not (index % 2) ^ counterfactual ^ relation_shuffle
    if not show_first:
        show_first = False
    actions = ("dispatch_alpha", "dispatch_beta") if show_first else ("dispatch_beta", "dispatch_alpha")
    value = str(NOTE_IDS[index] if lexical_value is None else lexical_value)
    from tool_lora.stage4_p0.contract import ArgumentBinding, PolicyBranch
    branches = tuple(PolicyBranch(action, (
        ArgumentBinding("entity", "entity"), ArgumentBinding("scope", "literal", "scope")
    )) for action in actions)
    return GenericPolicyIR("priority", "urgent", branches, (("scope", value),))


def expected_api(ir: GenericPolicyIR, condition: str) -> str:
    action = ir.alternatives[0].action if condition == ir.condition_value else ir.alternatives[1].action
    return "simple_note.show_note" if action == "dispatch_alpha" else "simple_note.delete_note"


def build_fixture() -> dict[str, object]:
    schema = selected_schema()
    skills = [c0_ir(i).to_dict() for i in range(len(NOTE_IDS))]
    return {
        "format": "stage4_c0_unseen_tool_oracle_fixture_v1",
        "appworld_task": TASK,
        "application": APPLICATION,
        "selected_apis": list(API_NAMES),
        "selected_api_identities_absent_from_prior_stage4": all(api not in PREVIOUS_APIS for api in API_NAMES),
        "fresh_opaque_bindings": list(NOTE_IDS),
        "skills": skills,
        "public_schema": schema,
        "selection_rationale": {
            "application": "Simple Note is absent from P0/A1/A2/B0/B1/B2, unlike the prior Todoist/ Supervisor slice.",
            "semantics": "show_note reads detailed structured content; delete_note mutates persistent note records.",
            "arguments": "Both actions use integer note_id, which is structurally compatible with the frozen opaque literal binding while differing from Todoist task_id.",
            "task_fixture": "cf6abd2_2 contains the Simple Note account and note population; IDs 3042-3049 are fresh relative to prior Stage-4 fixtures.",
        },
        "provenance": {
            "api_doc_sha256": (
                hashlib.sha256(API_DOC.read_bytes()).hexdigest()
                if API_DOC.exists()
                else json.loads((OUT / "immutable_fixture.json").read_text())["provenance"]["api_doc_sha256"]
            ),
            "appworld_version": "0.1.3.post1",
            "appworld_commit": "efac79081936323733f68fd78260dab6981f195a",
        },
    }


def resolver_audit(fixture: dict[str, object]) -> dict[str, object]:
    rows = []
    truths = [GenericPolicyIR.from_dict(item) for item in fixture["skills"]]
    for skill, truth in enumerate(truths):
        for condition in ("urgent", "routine"):
            expected = expected_api(truth, condition)
            try:
                call = resolve(truth, MockRequest(f"note-{skill}", (("priority", condition),)))
                actual = call.api
                error = None
            except Exception as exc:  # intentional boundary audit
                actual = None
                error = f"{type(exc).__name__}: {exc}"
            rows.append({"skill": skill, "condition": condition, "expected_api": expected,
                         "actual_api": actual, "error": error,
                         "api_correct": actual == expected})
    return {"rows": rows, "resolution_accuracy": sum(row["api_correct"] for row in rows) / len(rows),
            "all_expected_calls_resolved": all(row["actual_api"] is not None for row in rows),
            "failure_classification": "generic-resolver failure"}


def serialization_audit(fixture: dict[str, object]) -> float:
    return sum(GenericPolicyIR.from_dict(item).to_dict() == item for item in fixture["skills"]) / len(fixture["skills"])


def main() -> None:
    if not API_DOC.exists():
        raise SystemExit(f"missing pinned AppWorld schema: {API_DOC}")
    fixture = build_fixture()
    audit = resolver_audit(fixture)
    serialization_identity = serialization_audit(fixture)
    OUT.mkdir(parents=True, exist_ok=True)
    fixture_path = OUT / "immutable_fixture.json"
    fixture_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n")
    manifest = {
        "experiment": "stage4_c0_unseen_tool_oracle",
        "status": "blocked_at_frozen_resolver_boundary" if audit["resolution_accuracy"] < 1.0 else "ready_for_execution",
        "classification": audit["failure_classification"] if audit["resolution_accuracy"] < 1.0 else "not_applicable",
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "resolver_audit": audit,
        "representability": {"oracle_ir_instances": len(fixture["skills"]), "all_contract_valid": True, "serialization_identity": serialization_identity},
        "selected_application": APPLICATION,
        "selected_apis": list(API_NAMES),
        "prior_stage4_api_exclusion": fixture["selected_api_identities_absent_from_prior_stage4"],
        "appworld_task": TASK,
        "appworld_version": "0.1.3.post1",
        "appworld_commit": "efac79081936323733f68fd78260dab6981f195a",
        "appworld_root": str(APPWORLD_ROOT),
        "api_doc_sha256": fixture["provenance"]["api_doc_sha256"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frozen_components": ["P0 IR", "relational/lexical decomposition", "serialization/canonicalization", "B0 resolver", "evaluator methodology", "support-removal semantics", "causal controls"],
        "learned_acquisition_used": False,
        "required_outcome_conditions": ["matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant", "serialization_reload"],
        "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged",
    }
    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"fixture": str(fixture_path), "manifest": str(manifest_path), "fixture_sha256": manifest["fixture_sha256"], "serialization_identity": serialization_identity, "resolution_accuracy": audit["resolution_accuracy"], "classification": manifest["classification"]}, indent=2))


if __name__ == "__main__":
    main()
