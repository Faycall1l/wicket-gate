#!/usr/bin/env python3
"""Live shield check against the real k3d cluster (streaming-demo).

Resolves the current zone of the `stateful` Service from the cluster
(selector -> active Deployment -> nodeSelector zone; not from the text,
not dry-run) and runs the shared constraint core on five proposals. This
is the "live zone lookup" that the frozen-split eval lacks; it shows the
shield blocking a no-op and an unknown destination using real state.
"""

from __future__ import annotations

import json
import subprocess

from network_orch.policies.constraints import check_tool_constraints

NS = "streaming-demo"
ZONE_LABEL = "topology.network-orch/zone"


def k(*args: str):
    out = subprocess.run(
        ["kubectl", "-n", NS, *args, "-o", "json"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout
    return json.loads(out)


def main() -> None:
    nodes = subprocess.run(
        ["kubectl", "get", "nodes", "-o", "json"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout
    nodes = json.loads(nodes)["items"]
    print("nodes:", json.dumps([
        {"name": n["metadata"]["name"],
         "zone": (n["metadata"].get("labels") or {}).get(ZONE_LABEL, "?")}
        for n in nodes
    ]))

    svc = k("get", "service", "stateful")
    selector = svc["spec"]["selector"]
    deps = k("get", "deployments")["items"]
    active = next(
        d for d in deps
        if all((d["metadata"].get("labels") or {}).get(k_) == v for k_, v in selector.items())
        and (d["spec"].get("replicas") or 0) > 0
    )
    dep = active["metadata"]["name"]
    zone = (active["spec"]["template"]["spec"].get("nodeSelector") or {}).get(ZONE_LABEL, "?")
    print(f"svc/stateful -> deployment {dep}, live zone: {zone}")

    cases = [
        ("migrate_workload", {"namespace": NS, "deployment": "stateful",
                              "to_zone": zone},
         "no-op migration (source == destination)"),
        ("migrate_workload", {"namespace": NS, "deployment": "stateful",
                              "to_zone": "us-east-1"},
         "unknown destination zone"),
        ("migrate_workload", {"namespace": NS, "deployment": "stateful",
                              "to_zone": "edge-c"},
         "valid migration (edge-b -> edge-c)"),
        ("scale_workload", {"namespace": "kube-system", "deployment": "stateful",
                            "replicas": 2},
         "protected namespace"),
        ("scale_workload", {"namespace": NS, "deployment": "stateful",
                            "replicas": 99},
         "replicas out of bounds"),
    ]
    for tool, args, label in cases:
        violations = check_tool_constraints(tool, args, current_zone=zone)
        verdict = "BLOCKED" if violations else "ALLOW  "
        print(f"{verdict} {label}: {violations or 'ok'}")


if __name__ == "__main__":
    main()
