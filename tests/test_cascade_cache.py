from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.cache import VerifiedDecisionCache, normalize

A = Action(ActionKind.SCALE, replicas=3)


def test_normalize_strips_ticket_and_punct():
    assert normalize("Scale streamer to 3, please (ticket 4812)!") == "scale streamer to 3 please"


def test_hit_miss_and_rate():
    c = VerifiedDecisionCache()
    assert c.get("scale streamer to 3") is None
    c.put("Scale streamer to 3.", A)
    assert c.get("scale streamer to 3 (ticket 9999)") == A
    assert c.hit_rate == 0.5


def test_ttl_expiry_and_lru_eviction():
    c = VerifiedDecisionCache(max_size=2, ttl_s=10)
    c.put("a b", A, now=0)
    assert c.get("a b", now=11) is None
    c.put("one", A, now=0); c.put("two", A, now=0); c.put("three", A, now=0)
    assert len(c) == 2 and c.get("one", now=1) is None


def test_numbers_are_not_collapsed():
    assert normalize("scale to 100 replicas") != normalize("scale to 200 replicas")
