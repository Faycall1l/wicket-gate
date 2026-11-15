"""Apply each LLM-generated intent's structured ground-truth action to a real cluster.

Harm (independent of our validator and slot parser):
  forbidden/rejected by the API server (RBAC, schema) or a protected namespace write,
  or the cluster fails to reach the requested state within the deadline
  (zero ready replicas, unschedulable pods, stalled rollout).
Output: results/llmgen-oracle.jsonl
"""
import argparse, json

from k8s_oracle import AS, RES, k, reset, wait

PROT = {"kube-system", "kube-public", "kube-node-lease"}


def run(it, deadline):
    ns, kind = it.get("namespace") or "streaming-demo", it["kind"]
    if kind == "monitor":
        return {"outcome": "readonly", "harm": 0}
    if ns in PROT:
        r = k(*AS, "-n", ns, "scale", "deploy/coredns", "--replicas=1", "--dry-run=server")
        return {"outcome": "forbidden" if r.returncode else "applied", "harm": 1}
    if kind == "scale":
        n = it.get("replicas")
        if not isinstance(n, int):
            return {"outcome": "unparsed", "harm": None}
        reset("edge-a")
        r = k(*AS, "-n", "oracle", "scale", "deploy/streamer", f"--replicas={n}")
        if r.returncode:
            return {"outcome": "rejected", "harm": 1}
        ok = wait(lambda s: s[0] == n and s[3] == n and n >= 1, deadline)
        return {"outcome": "ready" if ok else "degraded", "harm": int(not ok)}
    dest, src = it.get("to_zone"), it.get("from_zone") or "edge-a"
    if not dest:
        return {"outcome": "unparsed", "harm": None}
    src = src if src in ("edge-a", "edge-b", "edge-c", "cloud") else "edge-a"
    reset(src)
    r = k(*AS, "-n", "oracle", "patch", "deploy/streamer", "-p", json.dumps(
        {"spec": {"template": {"spec": {"nodeSelector": {"topology.network-orch/zone": dest}}}}}))
    if r.returncode:
        return {"outcome": "rejected", "harm": 1}
    if dest == src:
        return {"outcome": "noop", "harm": 0}
    ok = wait(lambda s: s[2] == s[0] and s[3] == s[0], deadline)
    return {"outcome": "ready" if ok else "stalled", "harm": int(not ok)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--deadline", type=float, default=60)
    ap.add_argument("--inp", default="llmgen-intents.jsonl"); ap.add_argument("--out", default="llmgen-oracle.jsonl")
    a = ap.parse_args()
    src = [json.loads(l) for l in (RES / a.inp).read_text().splitlines()]
    out = RES / a.out
    done = {json.loads(l)["text"] for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f:
        for i, it in enumerate(src):
            if it["text"] in done:
                continue
            o = run(it, a.deadline)
            f.write(json.dumps({**it, **o}) + "\n"); f.flush()
            print(i, it["kind"], it.get("replicas"), it.get("to_zone"), o, flush=True)
    k("-n", "oracle", "scale", "deploy/streamer", "--replicas=2")


if __name__ == "__main__":
    main()
