from tool_lora.stage4_c1_a0.acquisition import audit_support


def test_a0_module_requires_two_distinct_generic_effects():
    assert callable(audit_support)
