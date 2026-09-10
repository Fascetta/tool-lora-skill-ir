from __future__ import annotations

from benchmarks.stage4_p0_oracle_tool_policy import make_ir, request, SCHEMA
from tool_lora.stage4_p0.contract import GenericPolicyIR, interpret


def test_oracle_reconnect_and_controls(tmp_path):
    for i in range(12):
        ir = make_ir(i)
        path = tmp_path / f"skill-{i}.json"
        ir.save(path)
        assert GenericPolicyIR.load(path) == ir
        decision = interpret(ir, request(i), SCHEMA)
        assert decision.arguments[0][1] == f"entity-heldout-{i}-x"
        assert decision.arguments[1][1] == f"org-{i:02d}-opaque-9f{i:02d}"


def test_counterfactual_changes_action_but_preserves_literal():
    for i in range(12):
        matched = interpret(make_ir(i), request(i), SCHEMA)
        counterfactual = interpret(make_ir(i, counterfactual=True), request(i), SCHEMA)
        assert matched.action != counterfactual.action
        assert matched.arguments[1] == counterfactual.arguments[1]
