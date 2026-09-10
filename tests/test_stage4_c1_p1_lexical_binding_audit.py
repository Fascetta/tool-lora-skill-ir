from benchmarks.stage4_c1_p1_lexical_binding_audit import build_fixture


def test_p1_bindings_are_unique_within_each_application():
    fixture = build_fixture()
    for app in ("todoist", "simple_note"):
        values = fixture[app]["binding_assignment"]
        assert len(values) == len(set(values))
        assert fixture[app]["constant_binding"] in {str(x) for x in values}
