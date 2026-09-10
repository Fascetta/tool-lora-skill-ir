import json

from tool_lora.stage4_b2.acquisition import acquire_with_frozen_model
from tool_lora.stage4_p0.learned_correspondence import CorrespondenceNet, TrainedCorrespondence


def test_b2_adapter_uses_public_trace_and_frozen_model():
    model = CorrespondenceNet()
    model.eval()
    support = json.dumps([
        {"request": {"text": "Handle this urgent item."},
         "public_tool_schema": ["todoist.show_task", "todoist.update_task"],
         "tool_call": {"api": "todoist.show_task", "arguments": {"task_id": 9001}},
         "observable_result": {"task_id": 9001}},
        {"request": {"text": "Handle this routine item."},
         "public_tool_schema": ["todoist.show_task", "todoist.update_task"],
         "tool_call": {"api": "todoist.update_task", "arguments": {"task_id": 9001}},
         "observable_result": {}},
    ])
    # A deterministic forward stub makes this unit test independent of a
    # randomly initialized model; the benchmark itself loads frozen A2-R1.
    class Stub:
        def __call__(self, features):
            import torch
            return torch.tensor([[1.0, 0.0]])
    ir = acquire_with_frozen_model(TrainedCorrespondence(Stub(), 0, 0), support)
    assert ir.lexical_bindings == (("scope", "9001"),)
    assert {branch.action for branch in ir.alternatives} == {"dispatch_alpha", "dispatch_beta"}
