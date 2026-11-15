"""Serve two OpenAI-compatible LLMs (one per T4) behind Cloudflare quick tunnels.
URLs are announced on a private ntfy topic; endpoints require an API key."""
import subprocess, time, re, os, urllib.request, sys
TOPIC, KEY = "netorch-23d76f2472b3047c", "sk-3b0d4b52b166dcc58a21fa08ee2667ec"
MODELS = [("Qwen/Qwen2.5-14B-Instruct-AWQ", 8000, "0"), ("Qwen/Qwen2.5-7B-Instruct-AWQ", 8001, "1")]
sh = lambda c: subprocess.run(c, shell=True, check=True)
sh("pip install -q vllm 2>&1 | tail -2")
sh("curl -sL -o /kaggle/working/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /kaggle/working/cloudflared")
def notify(msg):
    req = urllib.request.Request(f"https://ntfy.sh/{TOPIC}", data=msg.encode()); urllib.request.urlopen(req, timeout=20)
procs = []
for model, port, gpu in MODELS:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    procs.append(subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port),
         "--dtype", "float16", "--max-model-len", "4096", "--gpu-memory-utilization", "0.90",
         "--api-key", KEY, "--served-model-name", model.split("/")[1]],
        env=env, stdout=open(f"/kaggle/working/vllm{port}.log", "w"), stderr=subprocess.STDOUT))
for model, port, _ in MODELS:
    for _ in range(120):
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": f"Bearer {KEY}"}), timeout=5); break
        except Exception: time.sleep(10)
    else:
        notify(f"FAIL {model}: " + open(f"/kaggle/working/vllm{port}.log").read()[-800:]); raise SystemExit(1)
for model, port, _ in MODELS:
    log = open(f"/kaggle/working/cf{port}.log", "w")
    procs.append(subprocess.Popen(["/kaggle/working/cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"], stdout=log, stderr=subprocess.STDOUT))
    for _ in range(30):
        time.sleep(2)
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(f"/kaggle/working/cf{port}.log").read())
        if m: notify(f"READY {model.split('/')[1]} {m.group(0)}/v1"); break
time.sleep(11 * 3600)
