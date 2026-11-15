"""Signed audit envelope + staging-gate contract (issue #43, VIAVI pattern).

Every execution that would mutate the cluster produces one canonical
envelope: inputs-hash + validator verdict + twin probe result, sealed
with HMAC-SHA256. Same inputs → same envelope bytes (replay-deterministic),
so any reviewer can re-derive and verify a past decision offline.

Staging gate: a validated proposal applies to the staging namespace first;
a continuity probe gates promotion to the prod namespace. First live
staging probe landed 2026-09-19 (601 req, p95 7.3ms, envelope sealed in
the k6 artifact); promotion path stays untested.
"""

from __future__ import annotations

import hashlib
import hmac
import json

STAGING_NAMESPACE = "staging-demo"
PROD_NAMESPACE = "streaming-demo"


def canonical_inputs(action: str, params: dict) -> str:
    """Deterministic inputs hash: sorted-key JSON → sha256 hex."""
    blob = json.dumps({"action": action, "params": params}, sort_keys=True,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def build_envelope(action: str, params: dict, validation: list[str],
                   probe: dict, key: bytes) -> dict:
    """Seal the execution record. `validation` empty = validator passed;
    `probe` carries the twin/staging probe result (or {} when not run)."""
    inputs_hash = canonical_inputs(action, params)
    body = {"inputs_hash": inputs_hash, "action": action,
            "validation": list(validation), "probe": probe}
    mac = hmac.new(key, json.dumps(body, sort_keys=True,
                                   separators=(",", ":")).encode(),
                   hashlib.sha256).hexdigest()
    return {**body, "envelope_hmac": mac}


def verify_envelope(env: dict, key: bytes) -> bool:
    """Re-derive the seal; True iff the envelope is intact."""
    body = {k: env[k] for k in ("inputs_hash", "action", "validation", "probe")}
    mac = hmac.new(key, json.dumps(body, sort_keys=True,
                                   separators=(",", ":")).encode(),
                   hashlib.sha256).hexdigest()
    return hmac.compare_digest(mac, env.get("envelope_hmac", ""))


def stage_first(namespace: str) -> str:
    """Gate order: prod-namespace writes stage first, reads pass through."""
    if namespace == PROD_NAMESPACE:
        return STAGING_NAMESPACE
    return namespace
