"""Put the graded checks in front of any agent. `propose` is your agent: text in, typed Action out.

Run: uv run python examples/quickstart.py
"""
from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import Policy, guard

replies = iter([Action(ActionKind.SCALE, namespace="media", replicas=50),   # out of range
                Action(ActionKind.SCALE, namespace="media", replicas=12)])  # the agent's corrected answer


def propose(text: str) -> Action | None:
    print("agent sees:", text.splitlines()[-1][:100])
    return next(replies)


policy = Policy(max_replicas=20, zones=("site-1", "site-2"), namespaces=("media",))
d = guard("Scale the media streamer for tonight's event", propose, policy=policy)
print(d.verdict, d.reason, d.action, f"after {d.retries} repair")
