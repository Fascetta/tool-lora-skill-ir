import json
from pathlib import Path

from tool_lora.stage4_c0_r1.schema_audit import audit_role_identifiability


def test_c0_r1_detects_non_identifiable_role_mapping():
    fixture = json.loads(Path("results/stage4_c0_unseen_tool_oracle/immutable_fixture.json").read_text())
    audit = audit_role_identifiability(fixture["skills"], fixture["public_schema"])
    assert audit["candidate_mapping_count"] == 2
    assert audit["unique_mapping"] is False
    assert audit["accuracy"] == 0.0
