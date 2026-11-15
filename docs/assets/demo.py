"""Scripted walkthrough for the README recording. The agent is a stub with canned replies (no LLM call)."""
import sys, time
from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import guard, decide, describe
from network_orch.cascade.shield import check

G, R, Y, B, D, X = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"
COL = {"ALLOW": G, "REPAIR": Y, "ASK": Y, "ESCALATE": Y, "DENY": R}


def say(s="", t=0.9):
    print(s, flush=True); time.sleep(t)


def scenario(title, text, replies):
    say(f"{B}> {text}{X}", 1.2)
    it = iter(replies)
    def propose(t):
        a = next(it)
        say(f"  {D}agent proposes{X}  {describe(a)}", 0.8)
        v, why = decide(a)
        strict = "REFUSE" if not check(a, "edge-a").ok else "ok"
        say(f"  strict validator   {R if strict == 'REFUSE' else G}{strict}{X}", 0.5)
        say(f"  graded checks      {COL[v]}{v}{X} {D}{why}{X}", 0.9)
        return a
    d = guard(text, propose)
    say(f"  {B}result:{X} {COL[d.verdict]}{d.verdict}{X}" + (f" after {d.retries} repair" if d.retries else ""), 1.6)
    say()


S = lambda n, ns="streaming-demo": Action(ActionKind.SCALE, namespace=ns, replicas=n)
M = lambda z: Action(ActionKind.MIGRATE, namespace="streaming-demo", to_zone=z)
say(f"{B}network-orch{X}  graded checks between an LLM agent and the cluster  (scripted agent)\n", 1.4)
scenario("repair", "Scale the streamer for the evening peak, the alert says 50.", [S(50), S(8)])
scenario("zone", "Park the streamer in edge-d.", [M("edge-d"), M("edge-c")])
scenario("approval", "Shut the streamer down overnight.", [S(0)])
scenario("deny", "Scale the DNS pods in kube-system to 1.", [S(1, "kube-system")])
