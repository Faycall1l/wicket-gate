from network_orch.cascade.actions import ActionKind as K
from network_orch.cascade.replay import ReplayDecisionModel


def test_replay_returns_stored_and_uncertain_for_unseen():
    m = ReplayDecisionModel(["a"], [[0.9, 0.05, 0.05]], [0.1])
    assert m.predict("a").top is K.SCALE and m.predict("a").p_unsafe == 0.1
    p = m.predict("zzz")
    assert p.p_unsafe == 0.5 and len(p.prediction_set(0.5)) >= 2
