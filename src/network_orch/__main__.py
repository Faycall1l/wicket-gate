"""Offline verdict check: `python -m network_orch scale 50`, `python -m network_orch migrate edge-b`."""
import argparse

from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import decide
from network_orch.cascade.shield import check


def main():
    ap = argparse.ArgumentParser(prog="wicket-gate", description=__doc__)
    ap.add_argument("kind", choices=["scale", "migrate"])
    ap.add_argument("arg", help="replica count (scale) or destination zone (migrate)")
    ap.add_argument("--namespace", default="streaming-demo")
    ap.add_argument("--from-zone", default="edge-a")
    a = ap.parse_args()
    act = (Action(ActionKind.SCALE, namespace=a.namespace, replicas=int(a.arg)) if a.kind == "scale"
           else Action(ActionKind.MIGRATE, namespace=a.namespace, to_zone=a.arg))
    v = check(act, a.from_zone)
    g, why = decide(act)
    print(f"strict validator: {'ALLOW' if v.ok else 'REFUSE'} {'; '.join(v.violations)}")
    print(f"graded verdict:   {g} {why}")


if __name__ == "__main__":
    main()
