from network_orch.cascade.actions import ActionKind as K, Tier
from network_orch.cascade.decision import Prediction
from network_orch.cascade.gate import AdaptiveGate, Certificate
from network_orch.cascade.pipeline import Cascade
from network_orch.cascade.teacher import Teacher


class Stub:
    def __init__(self, probs, p_unsafe):
        self.p = Prediction(probs, p_unsafe)
        self.calls = 0

    def predict(self, text):
        self.calls += 1
        return self.p


def _gate():
    return AdaptiveGate(Certificate(alpha=0.1, tau=0.2))


def _teacher(reply):
    return Teacher(lambda s, u: reply)


def test_confident_t0_acts_then_cache_serves_without_t0():
    t0 = Stub({K.SCALE: 0.97, K.MIGRATE: 0.02, K.MONITOR: 0.01}, 0.01)
    c = Cascade(t0, _gate())
    d1 = c.decide("scale streamer to 4 replicas")
    assert d1.tier is Tier.T0 and d1.action.replicas == 4
    d2 = c.decide("scale streamer to 4 replicas!")
    assert d2.tier is Tier.CACHE and t0.calls == 1


def test_missing_param_escalates_to_teacher():
    t0 = Stub({K.SCALE: 0.97, K.MIGRATE: 0.02, K.MONITOR: 0.01}, 0.01)
    t1 = _teacher('{"kind":"scale","replicas":3,"confidence":0.9}')
    d = Cascade(t0, _gate(), t1).decide("we need more capacity")
    assert d.tier is Tier.T1 and d.action.replicas == 3


def test_shield_blocks_unsafe_t0_proposal():
    t0 = Stub({K.SCALE: 0.97, K.MIGRATE: 0.02, K.MONITOR: 0.01}, 0.01)
    d = Cascade(t0, _gate()).decide("scale streamer to 99 replicas")
    assert d.action.kind is K.REFUSE and d.violations


def test_high_unsafe_refuses_without_llm():
    t0 = Stub({K.MONITOR: 0.9, K.SCALE: 0.05, K.MIGRATE: 0.05}, 0.9)
    t1 = Teacher(lambda s, u: (_ for _ in ()).throw(AssertionError("LLM called")))
    d = Cascade(t0, _gate(), t1).decide("turn off the cluster")
    assert d.action.kind is K.REFUSE and d.tier is Tier.T0


def test_uncertain_defers_and_no_teacher_abstains():
    t0 = Stub({K.SCALE: 0.5, K.MIGRATE: 0.45, K.MONITOR: 0.05}, 0.05)
    d = Cascade(t0, _gate()).decide("fix the zone")
    assert d.tier is Tier.HUMAN and d.action is None


def test_t1_unparseable_abstains():
    t0 = Stub({K.SCALE: 0.5, K.MIGRATE: 0.45, K.MONITOR: 0.05}, 0.05)
    d = Cascade(t0, _gate(), _teacher("no idea")).decide("hmm")
    assert d.tier is Tier.HUMAN


def test_underspecified_multiplier_refuses_before_cache_and_t0():
    t0 = Stub({K.SCALE: 0.97, K.MIGRATE: 0.02, K.MONITOR: 0.01}, 0.01)
    c = Cascade(t0, _gate())
    d = c.decide("increase streamer replicas to 10x current")
    assert d.action.kind is K.REFUSE and d.tier is Tier.T0
    assert t0.calls == 0  # refused before the model is consulted


def test_migrate_without_source_zone_abstains_not_applies():
    t0 = Stub({K.MIGRATE: 0.97, K.SCALE: 0.02, K.MONITOR: 0.01}, 0.01)
    d = Cascade(t0, _gate()).decide("migrate streamer to edge-b")
    assert d.action is None or d.action.kind is K.REFUSE  # never an applied migration
