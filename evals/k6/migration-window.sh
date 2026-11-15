#!/usr/bin/env bash
set -euo pipefail
# k6 migration-window (issue #1 acceptance): 20 rps into svc/streamer via
# port-forward, reschedule the streamer pods onto another edge zone MID-RUN
# (nodeSelector patch), then split --out json into steady / migration /
# steady-after phases.
#
# Run on a QUIET host: scripts/quiet-host.sh FIRST (the contended host once
# showed an endpoint blackhole + TLS-handshake port-forward deaths; a quiet
# host reproduced 100% continuity in the smoke test below).
#
# Usage:
#   evals/k6/migration-window.sh [duration] [patch-at-sec] [outdir]
export KUBECONFIG="${KUBECONFIG:-$HOME/.config/k3d/kubeconfig-network-orch.yaml}"

DUR="${1:-5m}"
PATCH_AT="${2:-120}"
OUT="${3:-evals/k6/out/migration-$(date +%Y%m%d-%H%M%S)}"
LOCAL_PORT=18080
PF_PID=""
K6_PID=""
mkdir -p "$OUT"

cleanup() {
    [[ -n "$K6_PID" ]] && kill "$K6_PID" 2>/dev/null || true
    [[ -n "$PF_PID" ]] && kill "$PF_PID" 2>/dev/null || true
}
trap cleanup EXIT

# 0) In-cluster attribution channel (prober-py -> http://streamer/): same
#    0.05s cadence as k6, but the data plane only — no port-forward. If this
#    shows 100% while the k6/pf channel blackholes, that's pf lifecycle, not
#    endpoint starvation.
python3 - "$OUT" <<'PY'
import subprocess, time
out = "evals/k6/out"
dur = 6 * 60            # seconds of load
N = int(dur / 0.05)     # 20 rps for 6 min
cm = subprocess.run(["kubectl","-n","streaming-demo","create","configmap","prober-src",
                     "--from-file=loader.py=evals/loader.py","--dry-run=client","-o","yaml"],
                    capture_output=True, text=True)
subprocess.run(["kubectl","-n","streaming-demo","apply","-f","-"], input=cm.stdout,
               capture_output=True, text=True)
subprocess.run(["kubectl","-n","streaming-demo","delete","pod","prober-py","--wait=false"],
               capture_output=True, text=True)
tpl = open("k8s/demo/prober.yaml").read().replace("__URL__","http://streamer/").replace("__N__",str(N)).replace("__GAP__","0.05")
subprocess.run(["kubectl","-n","streaming-demo","apply","-f","-"], input=tpl,
               capture_output=True, text=True)
for _ in range(90):
    st = subprocess.run(["kubectl","-n","streaming-demo","get","pod","prober-py",
                         "-o","jsonpath={.status.phase}"],capture_output=True,
                        text=True, timeout=20).stdout.strip()
    if st in ("Running","Succeeded"):
        print(f"[driver] in-cluster prober running (N={N})", flush=True)
        break
    time.sleep(2)
else:
    print("[driver] WARN: prober-py did not start; external channel only")
PY

# 1) Own the port-forward (pf deaths strangled earlier runs).
kubectl port-forward -n streaming-demo svc/streamer "$LOCAL_PORT":80 >/dev/null 2>&1 &
PF_PID=$!

# 2) Hard-fail if the pf never serves 200; a dead pf = a garbage run.
python3 - "$LOCAL_PORT" <<'PY'
import sys, time, urllib.request
port = int(sys.argv[1])
url = f"http://localhost:{port}/"
t0 = time.time()
while time.time() - t0 < 45:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            if r.status == 200:
                print(f"[driver] k6 target ready after {time.time()-t0:.1f}s", flush=True)
                sys.exit(0)
    except Exception:
        pass
    time.sleep(1)
print("[driver] FATAL: no 200 on k6 target within 45s; aborting", flush=True)
sys.exit(3)
PY

# 3) Launch k6; it now runs across both the steady and migration windows.
K6_TEST_DURATION="$DUR" k6 run --no-thresholds \
  --out json="$OUT/raw.jsonl" evals/k6/continuity.js >"$OUT/k6-stdout.txt" 2>&1 &
K6_PID=$!

# 3b) Watch the pf: record liveness epochs, so a dead pf is attributed.
(
    while kill -0 "$PF_PID" 2>/dev/null; do
        python3 -c "import time; print(time.time(), flush=True)" >>"$OUT/pf-alive.tsv"
        sleep 2
    done
    echo "died epoch $(python3 -c 'import time; print(time.time())')" >>"$OUT/pf-alive.tsv"
) &
WATCH_PID=$!

# 4) Sleep until patch time (aligned to real k6 start), then reschedule pods
#    onto a different edge zone. Patch happens while k6 is hammering the svc.
python3 - "$PATCH_AT" "$OUT" <<'PY'
import json, sys, time
patch_at = float(sys.argv[1])
time.sleep(patch_at)
open(f"{sys.argv[2]}/patch-at.json", "w").write(json.dumps({"epoch": time.time()}))
print(f"[driver] patch epoch={time.time():.1f}", flush=True)
PY

CUR_NODE=$(kubectl -n streaming-demo get pod -l app=streamer \
  -o jsonpath='{.items[0].spec.nodeName}')
CUR_ZONE=$(kubectl get node "$CUR_NODE" -o jsonpath='{.metadata.labels.topology\.network-orch/zone}')
case "$CUR_ZONE" in
    edge-a) NEW_ZONE="edge-b" ;;
    edge-b) NEW_ZONE="edge-c" ;;
    *)      NEW_ZONE="edge-a" ;;
esac
echo "[driver] pods on $CUR_NODE (zone=$CUR_ZONE) -> rescheduling to zone=$NEW_ZONE"
kubectl -n streaming-demo patch deploy streamer \
  --type=merge -p "{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"topology.network-orch/zone\":\"$NEW_ZONE\"}}}}}" >/dev/null

kubectl -n streaming-demo rollout status deploy/streamer --timeout=300s >"$OUT/rollout.txt" 2>&1
python3 - "$OUT" <<'PY'
import json, sys, time
p = json.load(open(f"{sys.argv[1]}/patch-at.json"))
done = time.time()
print(f"[driver] rollout window {done-p['epoch']:.0f}s")
json.dump({"patch_epoch": p["epoch"], "rollout_done_epoch": done},
          open(f"{sys.argv[1]}/windows.json", "w"))
PY

# 5) Let k6 finish, clean up, summarize per-phase + in-cluster attribution.
wait "$K6_PID"
kill "$WATCH_PID" 2>/dev/null || true
echo "[driver] k6 done; splitting phases and summarizing"
python3 evals/k6/split_phases.py "$OUT/raw.jsonl" "$OUT/windows.json" >"$OUT/summary.txt"
cat "$OUT/summary.txt"
echo "--- in-cluster (data-plane) attribution ---"
python3 - "$OUT/prober.result.json" <<'PY'
import json, subprocess, sys
out = sys.argv[1]
try:
    logs = subprocess.run(["kubectl","-n","streaming-demo","logs","prober-py"],
                          capture_output=True, text=True, timeout=120).stdout
except Exception as e:
    logs = f"ERR {e}"
ok = fail = 0
lats = []
for ln in logs.splitlines():
    p = ln.split()
    if len(p) != 3:
        continue
    try:
        ms = (int(p[1])-int(p[0]))/1e6
    except ValueError:
        continue
    if p[2] == "200":
        ok += 1; lats.append(ms)
    else:
        fail += 1
lats.sort()
q = lambda f: lats[min(int(f*len(lats)), len(lats)-1)] if lats else None
m = {"requests": ok+fail, "success_rate": ok/(ok+fail) if ok+fail else 0.0,
     "p50_ms": round(q(0.5),2) if lats else None, "p95_ms": round(q(0.95),2) if lats else None,
     "p99_ms": round(q(0.99),2) if lats else None, "max_ms": round(lats[-1],2) if lats else None}
json.dump(m, open(out,"w"))
print(json.dumps(m, indent=2))
PY
echo "--- pf liveness (attribution) ---"
python3 - "$OUT/pf-alive.tsv" "$OUT/windows.json" <<'PY'
import json, sys
lines = [l.strip() for l in open(sys.argv[1]) if l.strip()]
w = json.load(open(sys.argv[2]))
epochs = []
for l in lines:
    if l.startswith("died"):
        epochs.append(float(l.split()[2])); continue
    try: epochs.append(float(l))
    except ValueError: pass
print(f"pf alive samples={len(epochs)} last_alive={epochs[-1] if epochs else None}")
patch, done = w["patch_epoch"], w["rollout_done_epoch"]
print(f"patch={patch:.0f} rollout_done={done:.0f} last_pf_alive={epochs[-1] if epochs else -1}")
PY