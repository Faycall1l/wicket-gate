import json, sys, time, urllib.request
url, n, gap = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
for i in range(n):
    t = time.time()
    try:
        with urllib.request.urlopen(url + "/inc", timeout=5) as r:
            print(json.dumps({"t": t, "ok": True, "body": r.read().decode()}), flush=True)
    except Exception as e:
        print(json.dumps({"t": t, "ok": False, "err": str(e)[:120]}), flush=True)
    time.sleep(gap)
