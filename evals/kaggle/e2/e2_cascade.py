"""E2 on Kaggle (CPU): full cascade eval per T0 featurizer x teacher x calibration."""
import glob, itertools, os, subprocess, sys

root = os.path.dirname(glob.glob("/kaggle/input/**/cascade_eval.py", recursive=True)[0])
sys.path.insert(0, os.path.join(root, "src"))
os.makedirs("/kaggle/working/results", exist_ok=True)
data = os.path.join(root, "decision-dataset.jsonl")
env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"))

for t0, teacher, calib, alpha in itertools.product(
        ["tfidf", "minilm", "bge-small"], ["none", "oracle"],
        ["pooled", "worst_group"], [0.10, 0.05]):
    out = f"/kaggle/working/results/cascade-{t0}-{teacher}-{calib}-a{alpha}.json"
    cmd = [sys.executable, os.path.join(root, "cascade_eval.py"), "--data", data,
           "--t0", t0, "--teacher", teacher, "--calib", calib, "--alpha", str(alpha),
           "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(t0, teacher, calib, alpha, "rc", r.returncode, r.stderr[-300:] if r.returncode else "")
print("done")
