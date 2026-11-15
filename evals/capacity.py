"""Capacity probe and a capacity-aware check.

probe: for each zone and replica count 1..8, scale the oracle deployment and record whether the cluster reaches
the requested state within the deadline (harm = not ready). Output results/capacity-probe.jsonl.
capacity_ok: cluster-state check (allocatable CPU of the zone's node minus requests of other pods) derived from
scheduler semantics, not fitted to the probe.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from k8s_oracle import RES, k, reset, wait

ZONES = ("edge-a", "edge-b", "edge-c", "cloud")
POD_CPU = 0.25  # requests.cpu of the streamer pod (k8s/oracle.yaml)


def cpu(q):
    return float(q[:-1]) / 1000 if q.endswith("m") else float(q)


def free_cpu(zone, own=("oracle", "streamer")):
    nodes = json.loads(k("get", "nodes", "-l", f"topology.network-orch/zone={zone}", "-o", "json").stdout)["items"]
    if not nodes:
        return 0.0
    node = nodes[0]["metadata"]["name"]
    alloc = cpu(nodes[0]["status"]["allocatable"]["cpu"])
    pods = json.loads(k("get", "pods", "-A", "--field-selector", f"spec.nodeName={node}", "-o", "json").stdout)["items"]
    used = 0.0
    for p in pods:
        if p["metadata"]["namespace"] == own[0] and p["metadata"]["labels"].get("app") == own[1]:
            continue
        if p["status"].get("phase") in ("Succeeded", "Failed"):
            continue
        used += sum(cpu(c.get("resources", {}).get("requests", {}).get("cpu", "0")) for c in p["spec"]["containers"])
    return alloc - used


def capacity_ok(zone, n):
    return n * POD_CPU <= free_cpu(zone)


def probe(deadline=20):
    out = RES / "capacity-probe.jsonl"
    with out.open("w") as f:
        for z in ZONES:
            for n in range(1, 9):
                reset(z)
                fc = free_cpu(z)
                k("-n", "oracle", "scale", "deploy/streamer", f"--replicas={n}")
                ok = wait(lambda s: s[0] == n and s[3] == n, deadline)
                row = {"zone": z, "replicas": n, "harm": int(not ok), "free_cpu": round(fc, 3),
                       "cap_ok": int(n * POD_CPU <= fc)}
                f.write(json.dumps(row) + "\n"); f.flush(); print(row, flush=True)
    reset("edge-a")
    k("-n", "oracle", "scale", "deploy/streamer", "--replicas=2")


if __name__ == "__main__":
    probe()
