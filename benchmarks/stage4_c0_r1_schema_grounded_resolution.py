"""Stage 4-C0-R1: audit schema-grounded generic resolution before execution."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from tool_lora.stage4_c0_r1.schema_audit import audit_role_identifiability

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/stage4_c0_r1_schema_grounded_resolution"
C0_OUT = ROOT / "results/stage4_c0_unseen_tool_oracle"


def main() -> None:
    fixture_path = C0_OUT / "immutable_fixture.json"
    fixture = json.loads(fixture_path.read_text())
    audit = audit_role_identifiability(fixture["skills"], fixture["public_schema"])
    manifest = {
        "experiment": "stage4_c0_r1_schema_grounded_resolution",
        "status": "blocked_before_execution" if not audit["unique_mapping"] else "ready_for_execution",
        "classification": "resolver-interface non-identifiability" if not audit["unique_mapping"] else "not_applicable",
        "c0_fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "schema_audit": audit,
        "execution_performed": False,
        "controls_performed": False,
        "learned_acquisition_used": False,
        "frozen_components": ["P0 Skill IR", "C0 Simple Note fixture", "hidden-policy semantics", "serialization", "lexical representation", "AppWorld environment/version", "evaluator", "prior acquisition models/artifacts"],
        "required_controls": ["matched", "counterfactual", "relation_shuffled", "lexical_shuffled", "constant", "serialization_reload", "candidate_order_permutation", "irrelevant_tool_distractors"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "typeguard_workaround": "temporary --target distribution on PYTHONPATH; system site-packages unchanged",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "manifest.json"
    output.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"manifest": str(output), "classification": manifest["classification"], "unique_mapping": audit["unique_mapping"], "candidate_mapping_count": audit["candidate_mapping_count"]}, indent=2))


if __name__ == "__main__":
    main()
