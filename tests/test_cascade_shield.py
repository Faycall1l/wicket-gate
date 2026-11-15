from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.shield import check
from network_orch.cascade import slots


def test_scale_in_bounds_ok():
    assert check(Action(ActionKind.SCALE, replicas=4)).ok


def test_scale_out_of_bounds_blocked():
    v = check(Action(ActionKind.SCALE, replicas=99))
    assert not v.ok and not v.incomplete


def test_protected_namespace_blocked():
    a = Action(ActionKind.SCALE, namespace="kube-system", replicas=2)
    assert not check(a).ok


def test_missing_params_are_incomplete_not_unsafe():
    v = check(Action(ActionKind.SCALE))
    assert v.incomplete and not v.ok


def test_migrate_noop_blocked_and_valid_ok():
    a = Action(ActionKind.MIGRATE, to_zone="edge-a")
    assert not check(a, current_zone="edge-a").ok
    assert check(a, current_zone="edge-b").ok


def test_monitor_always_ok_refuse_never():
    assert check(Action(ActionKind.MONITOR)).ok
    assert not check(Action(ActionKind.REFUSE)).ok


def test_migrate_unknown_source_is_incomplete_not_unsafe():
    a = Action(ActionKind.MIGRATE, to_zone="edge-a")
    v = check(a, current_zone="?")
    assert not v.ok and v.incomplete
    assert "source zone unresolved" in v.violations


def test_underspecified_multiplier_phrases():
    assert slots.underspecified("increase streamer replicas to 10x current")
    assert slots.underspecified("triple streamer then double it to 96")
    assert slots.underspecified("scale streamer to 8 replicas twice") is not None
    assert slots.underspecified("double streamer twice to 4x and once more")


def test_underspecified_ignores_absolute_counts():
    assert slots.underspecified("scale streamer to 4 replicas") is None
    assert slots.underspecified("migrate streamer from edge-a to edge-b") is None
    assert slots.underspecified("set streamer replicas to 8") is None
