import json

from benchmarks.stage4_c0_unseen_tool_oracle import build_fixture, resolver_audit
from tool_lora.stage4_p0.contract import GenericPolicyIR


def test_c0_fixture_is_representable_and_unseen():
    fixture = build_fixture()
    assert fixture["application"] == "simple_note"
    assert fixture["selected_api_identities_absent_from_prior_stage4"] is True
    assert len(fixture["skills"]) == 8
    assert all(GenericPolicyIR.from_dict(item).lexical_bindings[0][1] for item in fixture["skills"])


def test_frozen_resolver_boundary_is_explicitly_audited():
    audit = resolver_audit(build_fixture())
    assert audit["resolution_accuracy"] == 0.0
    assert audit["failure_classification"] == "generic-resolver failure"
