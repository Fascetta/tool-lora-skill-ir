from __future__ import annotations

import hashlib
from pathlib import Path
import torch

from tool_lora.skill_ir.acquisition import PersistentSkillIR, acquire, execute
from tool_lora.skill_ir.correspondence_matcher import CorrespondenceMatcher, load_p6
from tool_lora.skill_ir.resolver import resolve_values

ROOT = Path(__file__).parents[1]
CHECKPOINT = ROOT / "results/stage3d_p6_r1/p6_r1_checkpoint.pt"
SUITE = ROOT / "results/stage3d_p6_r1/immutable_suite.json"

def test_frozen_reference_hashes_and_load():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == "d2721f3684ec7f0dc16bf46834098a35cf9e07d42499e89fe439ea47ac5cb827"
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == "8a8b1d71b1a7dd61973086c1041d4503887caeddced3f1562ab91fd22e6263a5"
    model = load_p6(CHECKPOINT)
    assert isinstance(model, CorrespondenceMatcher)

def test_persistent_ir_support_removal_and_reload(tmp_path):
    support = """Demonstration 0 request:\nGet the current weather for Rome.\nDemonstration 0 response:\n{\"weather\": {\"city\": \"Rome\"}}\nDemonstration 1 request:\nGet the Fahrenheit forecast for Paris.\nDemonstration 1 response:\n{\"forecast\": {\"city\": \"Paris\", \"units\": \"ev_F\"}}\n"""
    ir = acquire(support)
    path = tmp_path / "skill_ir.json"; ir.save(path)
    reloaded = PersistentSkillIR.load(path)
    assert "support" not in path.read_text().lower()
    assert execute(reloaded, "Get the Fahrenheit forecast for Berlin.")["forecast"]["city"] == "Berlin"

def test_resolver_keeps_generic_persisted_semantics():
    record = type("Record", (), {})()
    record.observations = (("exact", "city"), ("enum", "units"), ("transformed", "date"))
    record.output_key = "weather"; record.predicate = "current weather"
    record.transformations = (((1, 2, 0), "-", "/"),)
    record.literal_bindings = (("Celsius", "units", "opaque-c"), ("Fahrenheit", "units", "opaque-f"))
    other = type("Record", (), {})(); other.observations = (("exact", "city"), ("enum", "units")); other.output_key = "forecast"; other.predicate = "forecast"; other.transformations = (); other.literal_bindings = record.literal_bindings
    class IR: records = (other, record)
    assert resolve_values(IR())[7:] == ("MDY", "/")
