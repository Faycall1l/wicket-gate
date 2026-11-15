#!/usr/bin/env bash
# Package src/ + cascade_eval.py + dataset as a Kaggle dataset (create or version).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=$(mktemp -d)
mkdir -p "$OUT/src"
cp -R src/network_orch "$OUT/src/"
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
cp evals/cascade_eval.py evals/drift_sim.py "$OUT/"
cp evals/results/decision-dataset.jsonl "$OUT/"
cat > "$OUT/dataset-metadata.json" <<J
{"title":"netorch-src","id":"faycal00/netorch-src","licenses":[{"name":"CC0-1.0"}]}
J
if kaggle datasets status faycal00/netorch-src >/dev/null 2>&1; then
  kaggle datasets version -p "$OUT" -m "${1:-update}" --dir-mode zip
else
  kaggle datasets create -p "$OUT" --dir-mode zip
fi
