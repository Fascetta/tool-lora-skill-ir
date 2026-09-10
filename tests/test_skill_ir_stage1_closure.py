"""Focused, dependency-light invariants for the closed Stage-1 path."""
from __future__ import annotations

import itertools
import json
import random

import torch

from tool_lora.skill_ir.field_permutation import PERMS
from tool_lora.skill_ir.protocol_ir import (
    NINE_PROTOCOL_IR_COMPONENTS,
    ExplicitSkillIR,
    ProtocolIR,
    old_ir_component,
)
from tool_lora.skill_ir.protocol_schema import ProtocolValues
from tool_lora.skill_ir.separator_pointer import pointer_index


def test_legacy_ir_has_nine_exact_additive_contributions():
    values = tuple(f"value_{i}" for i in range(9))
    expected = sum((old_ir_component(i, value) for i, value in enumerate(values)), torch.zeros(8, 256))
    assert torch.equal(expected, ExplicitSkillIR.from_values(values).legacy_ir_tensor)
    assert len(NINE_PROTOCOL_IR_COMPONENTS) == 9
    assert tuple(ProtocolValues.from_tuple(values).as_tuple()) == values


def test_legacy_ir_reconstructs_one_hundred_random_protocol_vectors():
    rng = random.Random(20260818)
    for _ in range(100):
        values = tuple(f"token_{rng.randrange(100000)}" for _ in range(9))
        ir = ExplicitSkillIR.from_values(values)
        reference = ProtocolIR(values[:2], values[2:5], values[5:7], (values[7], values[8])).vector()
        assert torch.equal(ir.legacy_ir_tensor, reference)


def test_persistent_skill_ir_round_trip(tmp_path):
    values = tuple(f"v{i}" for i in range(9))
    path = tmp_path / "skill_ir.json"
    original = ExplicitSkillIR.from_values(values, {"date_order": 1.0})
    original.save(path)
    restored = ExplicitSkillIR.load(path)
    assert restored.protocol_values == original.protocol_values
    assert torch.equal(restored.legacy_ir_tensor, original.legacy_ir_tensor)
    assert restored.confidence == original.confidence
    assert "support" not in json.dumps(restored.to_dict()).lower()


class Tokenizer:
    def __call__(self, value, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in value]}


def test_separator_pointer_and_candidate_order_randomization():
    tokenizer = Tokenizer()
    candidates = ("sx_b", "sx_a")
    assert pointer_index(tokenizer, "sx_a", candidates) == 1
    assert pointer_index(tokenizer, "sx_b", candidates) == 0


def test_field_permutation_space_is_exactly_six_legal_permutations():
    assert PERMS == tuple(itertools.permutations(range(3)))
    assert len(PERMS) == 6
    assert all(sorted(perm) == [0, 1, 2] for perm in PERMS)
