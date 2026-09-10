from __future__ import annotations

from tool_lora.stage4_b1.acquisition import acquire


def test_appworld_trace_acquisition_is_generic_and_exact():
    support = '[{"request":{"text":"Please handle the urgent item."},"tool_call":{"api":"todoist.show_task","arguments":{"task_id":4766}},"observable_result":{"task_id":4766},"observable_state":{"task_id":4766}},{"request":{"text":"Please handle the routine item."},"tool_call":{"api":"todoist.update_task","arguments":{"task_id":4766}},"observable_result":{},"observable_state":{"task_id":4766}}]'
    ir = acquire(support)
    assert ir.lexical_bindings == (("scope", "4766"),)
    assert [branch.action for branch in ir.alternatives] == ["dispatch_alpha", "dispatch_beta"]
