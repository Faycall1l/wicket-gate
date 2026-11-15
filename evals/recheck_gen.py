"""3x majority rerun of validator-passed harm=1 items in a labeled intent file (arg: file name)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from k8s_oracle import RES, k
from recheck_harm import maj, DEADLINE
from network_orch.cascade import shield
from llmgen_eval import gt_action

p = RES / sys.argv[1]
rows = [json.loads(l) for l in p.read_text().splitlines()]
for r in rows:
    r.setdefault("harm_first", r.get("harm"))
    if r.get("harm") != 1 or r.get("deadline") == DEADLINE or r["kind"] not in ("scale", "migrate"):
        continue
    act = gt_action(r)
    zone = r.get("from_zone") if r.get("from_zone") in ("edge-a", "edge-b", "edge-c", "cloud") else "edge-a"
    if not shield.check(act, zone).ok:
        continue
    it = {x: r.get(x) for x in ("kind", "replicas", "to_zone", "namespace", "from_zone")}
    h, hs = maj(it); print(r["kind"], r.get("replicas"), r.get("to_zone"), 1, "->", h, hs, flush=True)
    r["harm"], r["reruns"], r["deadline"] = h, hs, DEADLINE
    p.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
k("-n", "oracle", "scale", "deploy/streamer", "--replicas=2")
