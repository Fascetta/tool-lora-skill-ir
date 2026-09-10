import json
from pathlib import Path

from tool_lora.stage4_c1.contract import EffectAnchoredIR
from tool_lora.stage4_c1.schema_resolver import audit_identifiability


def test_c1_effect_anchor_is_identifiable_and_ambiguity_is_rejected():
    fixture = json.loads(Path("results/stage4_c1_p0_effect_anchor/immutable_fixture.json").read_text())
    for family in ("todoist", "simple_note"):
        irs = [EffectAnchoredIR.from_dict(x) for x in fixture[family]["irs"]]
        assert audit_identifiability(irs, fixture[family]["schemas"])["unique_mapping"]
    ambiguity = fixture["ambiguity_control"]
    assert not audit_identifiability([EffectAnchoredIR.from_dict(ambiguity["ir"])], ambiguity["schemas"])["unique_mapping"]
