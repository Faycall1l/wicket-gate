from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import ALLOW, ASK, DENY, ESCALATE, REPAIR, decide, guard


def scale(n, ns="streaming-demo"):
    return Action(ActionKind.SCALE, namespace=ns, replicas=n)


def mig(z, ns="streaming-demo"):
    return Action(ActionKind.MIGRATE, namespace=ns, to_zone=z)


def test_verdict_classes():
    assert decide(scale(3))[0] == ALLOW
    assert decide(scale(0))[0] == ESCALATE
    assert decide(scale(50))[0] == REPAIR
    assert decide(scale(None))[0] == ASK
    assert decide(scale(3, "kube-system"))[0] == DENY
    assert decide(mig("mars"))[0] == REPAIR
    assert decide(mig(None))[0] == ASK
    assert decide(mig("edge-b"))[0] == ALLOW
    assert decide(Action(ActionKind.REFUSE))[0] == DENY
    assert decide(Action(ActionKind.MONITOR))[0] == ALLOW
    assert decide(scale(3), "multiplier")[0] == ASK


def test_guard_repairs_once():
    seen = []

    def propose(t):
        seen.append(t)
        return scale(50) if len(seen) == 1 else scale(8)
    d = guard("scale the streamer to fifty", propose)
    assert d.verdict == ALLOW and d.retries == 1 and d.action.replicas == 8
    assert "outside the allowed range" in seen[1]


def test_guard_retry_bound_and_no_proposal():
    d = guard("scale up a lot", lambda t: scale(50), retries=2)
    assert d.verdict == REPAIR and d.retries == 2 and d.action is None
    assert guard("x", lambda t: None).verdict == DENY


def test_custom_policy():
    from network_orch.cascade.graded import Policy
    p = Policy.from_dict({"max_replicas": 20, "zones": ["site-1", "site-2"], "namespaces": ["media"]})
    assert decide(scale(12, "media"), "", p)[0] == ALLOW
    assert decide(scale(12), "", p)[0] == REPAIR          # namespace not in allowlist
    assert decide(mig("site-2", "media"), "", p)[0] == ALLOW
    assert decide(mig("edge-a", "media"), "", p)[0] == REPAIR
    assert decide(scale(12))[0] == REPAIR                 # default policy unchanged
