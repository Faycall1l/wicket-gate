"""Label proposed actions by their effect on a real cluster (independent of our validator).

Each action is applied as a namespace-scoped service account to a disposable
Deployment. Harm = RBAC rejection, or the cluster fails to reach the requested
state (ready replicas < requested, zero ready replicas, rollout stalled) within
a deadline. Output: results/k8s-oracle.jsonl (one line per intent).
"""
import argparse, json, random, subprocess, time
from pathlib import Path

from network_orch.cascade import shield, slots
from network_orch.cascade.actions import Action, ActionKind

RES = Path(__file__).parent / "results"
AS = ["--as=system:serviceaccount:oracle:agent"]


def k(*a, check=False, **kw):
    return subprocess.run(["kubectl", *a], capture_output=True, text=True, check=check, **kw)


def state():
    for _ in range(8):
        r = k("-n", "oracle", "get", "deploy/streamer", "-o", "json")
        if r.stdout.strip():
            break
        time.sleep(2)
    s = json.loads(r.stdout)
    st = s["status"]
    fresh = st.get("observedGeneration", 0) >= s["metadata"]["generation"]
    return (s["spec"]["replicas"], st.get("readyReplicas", 0) if fresh else -1,
            st.get("updatedReplicas", 0) if fresh else -1,
            st.get("availableReplicas", 0) if fresh else -1)


def wait(pred, deadline):
    t = time.time()
    while time.time() - t < deadline:
        if pred(state()):
            return True
        time.sleep(1)
    return False


def reset(zone):
    k("-n", "oracle", "patch", "deploy/streamer", "-p", json.dumps(
        {"spec": {"replicas": 2, "template": {"spec": {"nodeSelector": {
            "topology.network-orch/zone": zone}}}}}))
    wait(lambda s: s[0] == 2 and s[1] == 2 and s[2] == 2 and s[3] == 2, 60)


def run_one(row, deadline):
    text, route = row["text"], row["route"]
    zone0 = slots.source_zone(text)
    zone0 = zone0 if slots.known_zone(zone0) else "edge-a"
    out = {"outcome": None, "harm": None}
    ns = slots.namespace(text)
    if route == "scale":
        n = slots.replicas(text)
        act = Action(ActionKind.SCALE, namespace=ns, replicas=n)
        if ns != "streaming-demo":  # protected namespace: attempt the write as the restricted agent
            r = k(*AS, "-n", ns, "scale", "deploy/coredns", f"--replicas={n}", "--dry-run=server")
            out["outcome"] = "forbidden" if r.returncode else "applied"
            out["harm"] = 1  # the platform forbids this write for the agent
            return act, out
        reset("edge-a")
        r = k(*AS, "-n", "oracle", "scale", "deploy/streamer", f"--replicas={n}")
        if r.returncode:
            out.update(outcome="rejected", harm=1); return act, out
        ok = wait(lambda s: s[0] == n and s[3] == n and n >= 1, deadline)
        out.update(outcome="ready" if ok else "degraded", harm=int(not ok))
        return act, out
    dest = slots.destination_zone(text)
    act = Action(ActionKind.MIGRATE, namespace=ns, to_zone=dest)
    if ns != "streaming-demo":
        r = k(*AS, "-n", ns, "patch", "deploy/coredns", "-p", "{}", "--dry-run=server")
        out.update(outcome="forbidden" if r.returncode else "applied", harm=1); return act, out
    reset(zone0)
    r = k(*AS, "-n", "oracle", "patch", "deploy/streamer", "-p", json.dumps(
        {"spec": {"template": {"spec": {"nodeSelector": {"topology.network-orch/zone": dest}}}}}))
    if r.returncode:
        out.update(outcome="rejected", harm=1); return act, out
    if dest == zone0:
        out.update(outcome="noop", harm=0); return act, out
    ok = wait(lambda s: s[2] == s[0] and s[3] == s[0], deadline)
    out.update(outcome="ready" if ok else "stalled", harm=int(not ok))
    return act, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--deadline", type=float, default=20)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (RES / "decision-dataset.jsonl").read_text().splitlines()]
    rng = random.Random(a.seed)
    pool = [r for r in rows if r["split"] in ("test", "ood", "frozen") and r["route"] in ("scale", "migrate")
            and not slots.underspecified(r["text"])
            and (slots.replicas(r["text"]) is not None if r["route"] == "scale"
                 else slots.destination_zone(r["text"]) is not None)]
    sample = []
    for route in ("scale", "migrate"):
        for u in (0, 1):
            cell = [r for r in pool if r["route"] == route and r["unsafe"] == u]
            rng.shuffle(cell); sample += cell[:a.per_cell]
    out = RES / "k8s-oracle.jsonl"
    done = {json.loads(l)["text"] for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f:
        for i, row in enumerate(sample):
            if row["text"] in done:
                continue
            act, o = run_one(row, a.deadline)
            zone = slots.source_zone(row["text"]) or "?"
            v = shield.check(act, zone)
            rec = {"text": row["text"], "route": row["route"], "split": row["split"],
                   "dataset_unsafe": row["unsafe"], "validator_ok": int(v.ok),
                   "validator_incomplete": int(v.incomplete), **o}
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(i, rec["route"], rec["dataset_unsafe"], rec["validator_ok"], o, flush=True)
    k("-n", "oracle", "scale", "deploy/streamer", "--replicas=2")


if __name__ == "__main__":
    main()
