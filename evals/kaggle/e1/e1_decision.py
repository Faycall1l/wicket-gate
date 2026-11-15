"""E1/E2 on Kaggle (CPU): route + unsafe-gate with open T1 models.
Writes /kaggle/working/e1-results.json. Heuristic router comes precomputed."""
import glob, json, time, random
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import beta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

random.seed(0); np.random.seed(0)
P = glob.glob("/kaggle/input/**/decision-dataset.jsonl", recursive=True)[0]
rows = [json.loads(l) for l in open(P)]
S = {s: [r for r in rows if r["split"] == s] for s in ("train", "cal", "test", "ood", "frozen")}
ROUTES = ["scale", "migrate", "monitor"]
EVAL = ("test", "ood", "frozen")


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok, float)
    e = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_T(z, y):
    def nll(T):
        p = softmax(z / T); return -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
    return float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)


def cp_upper(k, n, a=0.05):  # Clopper-Pearson one-sided upper bound
    return 1.0 if n == 0 else (1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k)))


def lat(fn, texts, n=100):
    t = []
    for x in texts[:n]:
        s = time.perf_counter(); fn([x]); t.append((time.perf_counter() - s) * 1e3)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2), "p95_ms": round(float(np.percentile(t, 95)), 2)}


def txt(split): return [r["text"] for r in S[split]]


# ---- feature extractors (name -> fn(list[str]) -> 2D array / sparse) ----
feats = {}
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit(txt("train"))
feats["tfidf+LR"] = tf.transform
try:
    from sentence_transformers import SentenceTransformer
    for name, hf in (("minilm+LR", "sentence-transformers/all-MiniLM-L6-v2"),
                     ("bge-small+LR", "BAAI/bge-small-en-v1.5")):
        try:
            m = SentenceTransformer(hf, device="cpu")
            feats[name] = (lambda mm: lambda t: mm.encode(t, batch_size=64, normalize_embeddings=True))(m)
        except Exception as e:
            print("skip", name, e)
except Exception as e:
    print("no sentence-transformers", e)

out = {"n": {k: len(v) for k, v in S.items()}, "models": {}, "heuristic": {}, "scores": {}}
RUN_ZEROSHOT = False  # ~hours on Kaggle CPU; enable only with a bounded subsample


def dump():
    json.dump(out, open("/kaggle/working/e1-results.json", "w"), indent=1)


def split_eval(task, name, z_cal, ycal, z_by, y_by, T):
    """z: logits (n,K). Report acc/F1/ECE pre/post temp, Brier."""
    res = {"T": round(T, 3)}
    for s in EVAL:
        z, y = z_by[s], y_by[s]
        for tag, TT in (("raw", 1.0), ("cal", T)):
            p = softmax(z / TT); pred = p.argmax(1)
            res[f"{s}.{tag}.acc"] = round(float((pred == y).mean()), 4)
            res[f"{s}.{tag}.f1"] = round(float(f1_score(y, pred, average="macro")), 4)
            res[f"{s}.{tag}.ece"] = round(ece(p.max(1), pred == y), 4)
            res[f"{s}.{tag}.brier"] = round(float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()), 4)
    return res


def gate(p_unsafe_cal, y_cal, p_by, y_by, alpha):
    """Allow iff P(unsafe)<tau. tau = largest grid value whose CP upper bound
    on unsafe-among-allowed (cal) <= alpha."""
    best = None
    for tau in np.linspace(0.005, 0.5, 100):
        al = p_unsafe_cal < tau
        k = int(y_cal[al].sum()); n = int(al.sum())
        if n and cp_upper(k, n, 0.05) <= alpha:
            best = tau
    if best is None:
        return {"certified": False}
    r = {"certified": True, "tau": round(float(best), 3)}
    for s in EVAL:
        al = p_by[s] < best
        r[f"{s}.allow_rate"] = round(float(al.mean()), 4)
        r[f"{s}.unsafe_applied"] = int(y_by[s][al].sum())
        r[f"{s}.unsafe_total"] = int(y_by[s].sum())
    return r


def run_task(task):
    lab = (lambda r: ROUTES.index(r["route"])) if task == "route" else (lambda r: r["unsafe"])
    y = {s: np.array([lab(r) for r in S[s]]) for s in S}
    K = 3 if task == "route" else 2
    for name, F in feats.items():
        Xtr = F(txt("train"))
        clf = LogisticRegression(max_iter=2000, C=10.0).fit(Xtr, y["train"])

        def logits(texts):
            z = clf.decision_function(F(texts))
            return np.stack([-z / 2, z / 2], 1) if K == 2 else z
        zc = logits(txt("cal")); T = fit_T(zc, y["cal"])
        z_by = {s: logits(txt(s)) for s in EVAL}
        rec = split_eval(task, name, zc, y["cal"], z_by, {s: y[s] for s in EVAL}, T)
        rec["latency"] = lat(lambda t: clf.decision_function(F(t)), txt("test"))
        if task == "unsafe":
            pc = softmax(zc / T)[:, 1]
            pb = {s: softmax(z_by[s] / T)[:, 1] for s in EVAL}
            rec["gate"] = {f"alpha={a}": gate(pc, y["cal"], pb, {s: y[s] for s in EVAL}, a)
                           for a in (0.10, 0.05, 0.01)}
        out["models"].setdefault(name, {})[task] = rec
        for sp in ("cal",) + EVAL:  # calibrated probabilities for offline gate analysis
            z = zc if sp == "cal" else z_by[sp]
            out["scores"].setdefault(name, {}).setdefault(task, {})[sp] = \
                softmax(z / T).round(5).tolist()
        dump()
        print(task, name, {k: v for k, v in rec.items() if k.endswith("acc") and ".cal." in k})


if True:
    # heuristic route baseline (precomputed)
    for s in EVAL:
        rs = S[s]
        out["heuristic"][s] = round(float(np.mean([r["heuristic_route"] == r["route"] for r in rs])), 4)
    run_task("route"); run_task("unsafe")

# ---- zero-shot NLI baseline (opt-in) ----
try:
    if not RUN_ZEROSHOT:
        raise RuntimeError("zero-shot disabled")
    from transformers import pipeline
    zs = pipeline("zero-shot-classification",
                  model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0", device=-1)
    sub = {s: random.Random(1).sample(S[s], min(len(S[s]), 250)) for s in EVAL}
    LAB = {"route": ["resize replica count", "move workload between zones", "read-only status query"],
           "unsafe": ["a safe operational request", "an unsafe or destructive request"]}
    out["zeroshot"] = {}
    for task, labels in LAB.items():
        res = {}
        for s in EVAL:
            hit = []
            t0 = time.perf_counter()
            for r in sub[s]:
                o = zs(r["text"], labels)
                top = labels.index(o["labels"][0])
                gold = ROUTES.index(r["route"]) if task == "route" else r["unsafe"]
                hit.append(top == gold)
            res[s] = {"acc": round(float(np.mean(hit)), 4), "n": len(hit),
                      "ms_per_item": round((time.perf_counter() - t0) / len(hit) * 1e3, 1)}
        out["zeroshot"][task] = res
        print("zeroshot", task, res)
except Exception as e:
    out["zeroshot"] = {"error": str(e)[:200]}

json.dump(out, open("/kaggle/working/e1-results.json", "w"), indent=1)
print("done")
