from network_orch.cascade import slots
from network_orch.cascade.actions import ActionKind as K
from network_orch.cascade.decision import LinearDecisionModel

TR = [("scale streamer to 4 replicas", K.SCALE, 0), ("run 6 replicas of cache", K.SCALE, 0),
      ("migrate streamer from edge-a to edge-b", K.MIGRATE, 0),
      ("move cache to edge-c", K.MIGRATE, 0),
      ("is streamer healthy", K.MONITOR, 0), ("show cluster state", K.MONITOR, 0),
      ("scale streamer to 99 replicas", K.SCALE, 1),
      ("turn off the cluster", K.MONITOR, 1), ("delete the kube-system namespace", K.MONITOR, 1),
      ("stop all pods in kube-node-lease", K.MONITOR, 1)] * 6


def _fit():
    t, k, u = zip(*TR)
    return LinearDecisionModel().fit(t, k, u, t, k, u)


def test_predict_is_a_distribution_and_orders_kinds():
    m = _fit()
    p = m.predict("scale streamer to 5 replicas")
    assert abs(sum(p.probs.values()) - 1) < 1e-6 and p.top is K.SCALE
    assert m.predict("turn off the cluster").p_unsafe > m.predict("is cache healthy").p_unsafe


def test_prediction_set_grows_with_mass():
    p = _fit().predict("scale streamer to 5 replicas")
    assert p.prediction_set(0.5)[0] is p.top
    assert len(p.prediction_set(0.999)) >= len(p.prediction_set(0.5))


def test_save_load_roundtrip(tmp_path):
    m = _fit(); f = tmp_path / "m.joblib"; m.save(f)
    assert LinearDecisionModel.load(f).predict("is streamer healthy").top is K.MONITOR


def test_slots_explicit_only():
    assert slots.replicas("scale streamer to 12 replicas") == 12
    assert slots.replicas("we need more capacity") is None
    assert slots.destination_zone("move streamer from edge-a to edge-c") == "edge-c"
    assert slots.destination_zone("migrate it somewhere") is None


def test_source_zone_and_namespace_slots():
    assert slots.source_zone("move streamer from edge-a to edge-c") == "edge-a"
    assert slots.source_zone("scale streamer to 3") is None
    assert slots.namespace("scale cache to 2 in kube-system") == "kube-system"
    assert slots.namespace("scale cache to 2") == "streaming-demo"


def test_noop_and_unknown_destination_reach_the_shield():
    t = "the edge-c zone is degraded, move cache to edge-c"
    assert slots.source_zone(t) == "edge-c" and slots.destination_zone(t) == "edge-c"
    assert slots.destination_zone("move cache to mars") == "mars"
    assert slots.source_zone("migrate d to edge-b, it is currently on edge-a") == "edge-a"
    assert slots.destination_zone("move it to the nearest edge") is None
