from __future__ import annotations

from benchmarks.stage4_b0_appworld_oracle_reconnect import b0_ir
from tool_lora.stage4_b0.appworld_resolver import resolve
from tool_lora.stage4_p0.contract import MockRequest


def test_frozen_ir_maps_to_concrete_appworld_calls():
    request = MockRequest("heldout", (("priority", "urgent"),))
    for index in range(12):
        call = resolve(b0_ir(index), request)
        assert call.api in {"todoist.show_task", "todoist.update_task"}
        assert call.arguments[0][0] == "task_id"
