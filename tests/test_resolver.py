from itertools import permutations

from tool_lora.skill_ir.resolver import apply_permutation


def test_all_generic_three_component_permutations_use_target_index_convention():
    source = ("A", "B", "C")
    for permutation in permutations(range(3)):
        assert apply_permutation(source, permutation) == tuple(source[index] for index in permutation)


def test_known_executor_orders_use_persisted_convention():
    source = ("Y", "M", "D")
    assert apply_permutation(source, (0, 1, 2)) == ("Y", "M", "D")
    assert apply_permutation(source, (1, 2, 0)) == ("M", "D", "Y")
    assert apply_permutation(source, (2, 1, 0)) == ("D", "M", "Y")
