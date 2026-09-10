from __future__ import annotations

import json
from pathlib import Path

import torch

from tool_lora.functional_hypernet.functional_lora import TargetLinearSpec
from tool_lora.skill_ir.composition import SeparableSemanticComposer
from tool_lora.skill_ir.compression import hash_coordinate, protocol_vector, reachable_mask
from tool_lora.skill_ir.compiler import OracleCompiler, OracleIA3Compiler
from tool_lora.skill_ir.discrete_hybrid import discrete_metrics
from tool_lora.skill_ir.protocol_ir import ExplicitSkillIR, ProtocolIR
from tool_lora.skill_ir.registry import ExecutionSpec
from tool_lora.skill_ir.structured_hybrid import (
    LexicalSidecar, artifact_bytes, reconstruct_values, relation_input,
)


def test_protocol_ir_hash_and_explicit_round_trip_are_exact() -> None:
    values = ("fn_a", "fn_b", "field_a", "field_b", "field_c", "enum_a", "enum_b", "MDY", "sx_a")
    protocol = ProtocolIR(values[:2], values[2:5], values[5:7], values[7:])
    expected = torch.zeros(2048)
    for index, token in enumerate(values):
        expected[hash_coordinate(index, token)] += 1
    assert torch.equal(protocol.vector().reshape(-1), expected)
    explicit = ExplicitSkillIR.from_values(values)
    assert torch.equal(explicit.legacy_ir_tensor.reshape(-1), expected)


def test_structured_relation_round_trip_covers_all_72_states() -> None:
    side = LexicalSidecar(("fn_a", "fn_b"), ("field_a", "field_b", "field_c"), ("enum_a", "enum_b"), "sx_a")
    seen = set()
    for function in range(2):
        for field in range(6):
            for enum in range(2):
                for order in range(3):
                    values = reconstruct_values(side, (function, field, enum, order))
                    assert relation_input(values).shape == (20,)
                    assert torch.equal(protocol_vector(values), ProtocolIR(values[:2], values[2:5], values[5:7], values[7:]).vector().reshape(-1))
                    seen.add(values)
    assert len(seen) == 72


def test_lexical_sidecar_is_canonical_and_byte_accounting_is_explicit() -> None:
    values = ("fn_b", "fn_a", "field_c", "field_a", "field_b", "enum_b", "enum_a", "YMD", "sx_a")
    side = LexicalSidecar.from_values(values)
    assert side.functions == ("fn_a", "fn_b")
    assert side.fields == ("field_a", "field_b", "field_c")
    assert side.enums == ("enum_a", "enum_b")
    artifact = artifact_bytes(side, torch.zeros(8))
    assert artifact["semantic_code_bytes"] == 32
    assert artifact["total_bytes"] == artifact["lexical_bytes"] + 32 + 16


def test_reachable_mask_is_static_and_slot_indexed() -> None:
    pools = {"functions": ("fn_a",), "arguments": ("field_a",), "enums": ("enum_a",), "separators": ("sx_a",)}
    mask = reachable_mask(pools)
    assert mask.dtype == torch.bool
    assert bool(mask[hash_coordinate(0, "fn_a")])
    assert bool(mask[hash_coordinate(7, "DMY")])
    assert int(mask.sum()) >= 1


def test_vq_injectivity_metric_distinguishes_aliases() -> None:
    labels = torch.tensor([(a, b, c, d) for a in range(2) for b in range(6) for c in range(2) for d in range(3)])
    unique = torch.arange(72)
    metrics = discrete_metrics(unique, labels, 128)
    assert metrics["codes_with_multiple_states"] == 0
    assert metrics["states_with_multiple_codes"] == 0
    aliased = unique.clone(); aliased[1] = aliased[0]
    assert discrete_metrics(aliased, labels, 128)["codes_with_multiple_states"] == 1


def test_factor_separable_outputs_only_depend_on_typed_inputs() -> None:
    torch.manual_seed(3)
    composer = SeparableSemanticComposer()
    left = torch.tensor([[0, 0, 0, 0]])
    right = torch.tensor([[1, 5, 1, 2]])
    first = composer(left)
    second = composer(right)
    assert not torch.equal(first[0], second[0])
    assert not torch.equal(first[1], second[1])
    # Changing only factor A cannot alter B/C/D outputs, and vice versa.
    changed_a = left.clone(); changed_a[:, 0] = 1
    changed_b = left.clone(); changed_b[:, 1] = 1
    out_a = composer(changed_a); out_b = composer(changed_b)
    assert all(torch.equal(first[index], out_a[index]) for index in (1, 2, 3))
    assert all(torch.equal(first[index], out_b[index]) for index in (0, 2, 3))


def test_compiler_path_order_and_ia3_shapes_are_self_describing() -> None:
    specs = [
        TargetLinearSpec("model.layers.0.self_attn.q_proj", 0, "q_proj", 8, 8),
        TargetLinearSpec("model.layers.0.self_attn.v_proj", 0, "v_proj", 8, 8),
        TargetLinearSpec("model.layers.1.self_attn.q_proj", 1, "q_proj", 8, 8),
    ]
    paths = [spec.module_path for spec in specs]
    shapes = {spec.module_path: (spec.out_features, spec.in_features) for spec in specs}
    lora = OracleCompiler(paths, shapes, 4, 16)
    ia3_paths = ["model.layers.0.self_attn.k_proj", "model.layers.0.mlp.down_proj"]
    ia3_shapes = {ia3_paths[0]: (8, 8), ia3_paths[1]: (8, 16)}
    ia3 = OracleIA3Compiler(ia3_paths, ia3_shapes)
    lora_weights = lora.fast(torch.zeros(2048), ExecutionSpec("lora", 4, "T1"))
    ia3_weights = ia3.fast(torch.zeros(2048), ExecutionSpec("ia3", 1, "IA3_STANDARD"))
    assert list(lora_weights.factors) == paths
    assert ia3_weights.scales[ia3_paths[0]].numel() == 8
    assert ia3_weights.scales[ia3_paths[1]].numel() == 16
    round_trip = json.loads(json.dumps(ExecutionSpec("ia3", 1, "IA3_STANDARD").to_dict()))
    assert round_trip["mechanism"] == "ia3"


def test_stage2_checkpoint_metadata_round_trip(tmp_path: Path) -> None:
    metadata = {
        "stage2_format_version": "stage2b_b1_hybrid_v1",
        "ir_schema_version": "legacy_protocol_ir_v1",
        "K": 8,
        "architecture": "structured_hybrid_mlp",
        "loss": {"name": "four_relation_cross_entropy"},
        "seed": 20260817,
        "generator": {"seed": 20260817, "split": "train"},
        "train_reachable_mask": [True, False, True],
        "compiler_checkpoint": "stage1/compiler.pt",
    }
    path = tmp_path / "checkpoint.pt"
    torch.save({"format": "stage2_checkpoint_v1", "model_config": metadata}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["format"] == "stage2_checkpoint_v1"
    assert loaded["model_config"] == metadata
