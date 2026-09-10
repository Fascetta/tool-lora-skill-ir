from __future__ import annotations

from benchmarks.stage4_a1_learned_tool_policy import render_support
from benchmarks.stage4_p0_oracle_tool_policy import make_ir
from tool_lora.stage4_p0.learned_correspondence import acquire_with_model, train


def test_learned_correspondence_recovers_blinded_supports():
    model = train(17, steps=500)
    for skill in range(12):
        truth = make_ir(skill)
        artifacts = [acquire_with_model(model, render_support(truth, variant=variant, skill=skill))
                     for variant in ("primary", "alternate", "paraphrase")]
        assert artifacts[0] == artifacts[1] == artifacts[2]
