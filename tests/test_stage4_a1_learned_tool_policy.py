from __future__ import annotations

from benchmarks.stage4_a1_learned_tool_policy import render_support
from benchmarks.stage4_p0_oracle_tool_policy import SCHEMA, make_ir, request
from tool_lora.stage4_p0.acquisition import acquire
from tool_lora.stage4_p0.contract import interpret


def test_acquisition_recovers_each_blinded_support_variant():
    for skill in range(12):
        truth = make_ir(skill)
        artifacts = [acquire(render_support(truth, variant=variant, skill=skill))
                     for variant in ("primary", "alternate", "paraphrase")]
        assert artifacts[0] == artifacts[1] == artifacts[2]
        for condition in ("urgent", "routine"):
            assert interpret(artifacts[0], request(skill, condition), SCHEMA) == interpret(truth, request(skill, condition), SCHEMA)
