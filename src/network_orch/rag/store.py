"""ChromaDB RAG: curated runbooks + topology knowledge, cited retrieval."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

DATA_ROOT = Path("data")

COLLECTIONS = {
    "runbooks": "data/runbooks",
    "knowledge": "data/knowledge",
}

# ── warm-cache singletons ────────────────────────────────────────────────
# Re-creating the Chroma client / embedding function per call was ~60 ms of
# setup, and embedding the query once per collection doubled the CPU forward
# pass. Cache both, embed once, and memoize results so repeats are free.
_LOCK = threading.Lock()
_CLIENTS: dict[str, object] = {}
_EMBEDDING = None
_COLLECTIONS: dict[tuple[str, str], object] = {}
_RESULT_CACHE: "OrderedDict[tuple, list[dict]]" = OrderedDict()
_RESULT_CACHE_MAX = 512
_CACHE_STATS = {"hits": 0, "misses": 0}


def _client(persist_dir: str):
    c = _CLIENTS.get(persist_dir)
    if c is None:
        c = get_client(persist_dir)
        _CLIENTS[persist_dir] = c
    return c


def _embedding():
    global _EMBEDDING
    if _EMBEDDING is None:
        _EMBEDDING = _build_embedding_fn()
    return _EMBEDDING


def cache_info() -> dict:
    """Return retrieval-cache counters: hits, misses, size, hit_rate."""
    with _LOCK:
        hits, misses = _CACHE_STATS["hits"], _CACHE_STATS["misses"]
        total = hits + misses
        return {"hits": hits, "misses": misses, "size": len(_RESULT_CACHE),
                "hit_rate": round(hits / total, 4) if total else 0.0}


def clear_cache() -> None:
    """Drop memoized results and reset counters (tests / fresh measurement)."""
    with _LOCK:
        _RESULT_CACHE.clear()
        _CACHE_STATS.update(hits=0, misses=0)


def _invalidate_collections() -> None:
    with _LOCK:
        _COLLECTIONS.clear()



def _build_embedding_fn():
    """Real OSS embeddings: sentence-transformers all-MiniLM-L6-v2, local CPU.

    Upgrade path from the v1 hashed fallback (kept in git history): genuine
    semantic vectors, no API calls. Model pre-downloaded once; set
    SENTENCE_TRANSFORMERS_HOME to pin the cache location.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2", device="cpu", normalize_embeddings=True
    )


def get_client(persist_dir: str = "data/chroma"):
    import chromadb

    return chromadb.PersistentClient(path=persist_dir)


def _sections(md_path, text: str) -> list[tuple[str, str]]:
    """Split markdown into ## sections (TelcoAI-style section-aware chunking).

    Returns [(section_title, chunk_text)]. Doc title is prepended to every
    chunk so retrievals keep document-level context.
    """
    import re

    title = md_path.stem.replace("-", " ").replace("_", " ")
    parts = re.split(r"(?m)^##\s+(.*)$", text)
    if len(parts) < 3:
        return [(title, text[:1500])]
    out: list[tuple[str, str]] = []
    preamble = parts[0].strip()
    if preamble:
        out.append((f"{title} :: overview", f"{title}\n{preamble}"[:1500]))
    for i in range(1, len(parts) - 1, 2):
        head, body = parts[i].strip(), parts[i + 1].strip()
        if body:
            out.append((f"{title} :: {head}", f"{title} — {head}\n{body}"[:1500]))
    return out or [(title, text[:1500])]


def ingest(persist_dir: str = "data/chroma",
           section_frac: float = 1.0) -> dict:
    """(Re)build the collections from markdown sections. Returns chunk counts.

    `section_frac` keeps the first fraction of sections per doc (B6 corpus
    scaling; default 1.0 reproduces the exact current index byte-for-byte).
    """
    client = get_client(persist_dir)
    counts: dict[str, int] = {}
    for name, rel in COLLECTIONS.items():
        col = client.get_or_create_collection(name, embedding_function=_build_embedding_fn())
        docs, ids, metas = [], [], []
        for md in sorted((DATA_ROOT / rel.split("/", 1)[1]).glob("*.md")):
            text = md.read_text()
            secs = _sections(md, text)
            keep = max(1, int(len(secs) * section_frac))
            for j, (sec, chunk) in enumerate(secs[:keep]):
                docs.append(chunk)
                ids.append(f"{name}/{md.name}#{j}")
                metas.append({"source": f"{rel}/{md.name}", "section": sec})
        if docs:
            try:
                col.upsert(documents=docs, ids=ids, metadatas=metas)
            except Exception:
                col.add(documents=docs, ids=ids, metadatas=metas)
        counts[name] = len(docs)
    _invalidate_collections()
    clear_cache()
    return counts


def _collections(persist_dir: str, client):
    """Cached {name: collection} handles for a persist_dir."""
    out = {}
    for name in COLLECTIONS:
        key = (persist_dir, name)
        col = _COLLECTIONS.get(key)
        if col is None:
            try:
                col = client.get_collection(name, embedding_function=_embedding())
            except Exception:
                continue
            _COLLECTIONS[key] = col
        out[name] = col
    return out


def query(q: str, top_k: int = 4, persist_dir: str = "data/chroma") -> list[dict]:
    """Return [{source, excerpt}] merged across collections by vector distance.

    Both collections use the same embedding function, so distances are
    comparable and the merge is a genuine global top-k (previously the
    runbooks collection always filled the quota and knowledge never surfaced).

    Warm-path optimizations (issue #12): the client/embedding are singletons
    and the query is embedded ONCE, then matched against both collections via
    ``query_embeddings``; results are memoized so repeated intents are free.
    :func:`cache_info` reports the hit/miss mix.
    """
    key = (q, int(top_k), persist_dir)
    with _LOCK:
        if key in _RESULT_CACHE:
            _RESULT_CACHE.move_to_end(key)
            _CACHE_STATS["hits"] += 1
            return [dict(h) for h in _RESULT_CACHE[key]]
        _CACHE_STATS["misses"] += 1

    client = _client(persist_dir)
    vec = _embedding()([q])
    scored: list[tuple[float, dict]] = []
    for name, col in _collections(persist_dir, client).items():
        r = col.query(query_embeddings=vec, n_results=top_k,
                      include=["documents", "metadatas", "distances"])
        docs = r.get("documents", [[]])[0]
        metas = r.get("metadatas", [[]])[0]
        dists = r.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            scored.append((float(dist), {
                "source": (meta or {}).get("source", name),
                "section": (meta or {}).get("section", ""),
                "excerpt": doc[:800],
            }))
    scored.sort(key=lambda t: t[0])
    hits = [h for _, h in scored[:top_k]]
    with _LOCK:
        _RESULT_CACHE[key] = [dict(h) for h in hits]
        _RESULT_CACHE.move_to_end(key)
        while len(_RESULT_CACHE) > _RESULT_CACHE_MAX:
            _RESULT_CACHE.popitem(last=False)
    return hits
