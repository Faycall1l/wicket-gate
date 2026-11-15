"""Poll cluster + write JSONL telemetry (future RAG input + eval evidence)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def collect_once(dry_run: bool = True) -> dict:
    from network_orch.k8s.client import ClusterBackend

    s = ClusterBackend(dry_run=dry_run).get_state()
    rec = {
        "ts": dt.datetime.utcnow().isoformat(),
        "nodes": s.nodes,
        "workloads": s.workloads,
    }
    out = Path("data/telemetry") / f"{dt.date.today().isoformat()}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
