"""Minimal OpenAI-compatible chat client for T1 with explicit endpoint policy.

`configure(only=...)` pins the backend: "primary" or "groq" never fall back;
"auto" tries primary then Groq. Every successful call is tallied in `SERVED`
by the model that actually answered, so experiments can report provenance.
"""

from __future__ import annotations

from collections import Counter

import httpx

from network_orch.config import SETTINGS

SERVED: Counter = Counter()  # "<model>" -> successful calls
_ONLY = "auto"

_CLIENT: httpx.Client | None = None


def _client() -> httpx.Client:
    """One pooled client per process: connection reuse avoids the socket
    churn that exhausted macOS network buffers on long arm runs."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            http2=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            timeout=httpx.Timeout(60.0),
        )
    return _CLIENT


def configure(only: str = "auto") -> None:
    if only not in {"auto", "primary", "groq"}:
        raise ValueError(f"unknown backend policy: {only}")
    global _ONLY
    _ONLY = only


def _endpoints() -> list[tuple[str, str, str]]:
    primary, groq = SETTINGS.llm_endpoint(), SETTINGS.groq_endpoint()
    if _ONLY == "primary":
        return [primary]
    if _ONLY == "groq":
        return [groq]
    return [primary] + ([groq] if SETTINGS.groq_available else [])


def _payload(model: str, system: str, user: str) -> dict:
    body = {"model": model, "temperature": 0.0, "max_tokens": 400,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if "gpt-oss" in model:  # reasoning model: keep reasoning short so content fits
        body["reasoning_effort"] = "low"
    return body


def chat(system: str, user: str, timeout: float = 60.0) -> str:
    """One completion; tries allowed endpoints in order and tallies who served."""
    last: Exception | None = None
    for base_url, api_key, model in _endpoints():
        try:
            r = _client().post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=_payload(model, system, user),
                timeout=timeout)
            r.raise_for_status()
            SERVED[model] += 1
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001 - fall through per policy
            last = exc
    raise RuntimeError(f"no LLM endpoint answered (policy={_ONLY}): {last}")
