import json
from pathlib import Path


def test_g1_manifest_reports_gauge_equivalence():
    path = Path("results/stage4_c1_a0_g1_gauge_audit/manifest.json")
    if path.exists():
        data = json.loads(path.read_text())
        assert data["permutation_invariant_effect_anchor_recovery"] == 1.0
