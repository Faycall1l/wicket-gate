"""network_orch_mcp: graded safety checks and guarded execution for edge-orchestration agents.

Tools (all prefixed `network_orch_`):
  check_action     read-only   verdict (ALLOW/REPAIR/ASK/ESCALATE/DENY) plus the strict validator's verdict
  get_policy       read-only   the limits being enforced
  get_cluster_state read-only  nodes and workloads of the managed namespace
  list_benchmark   read-only   paginated, filterable cluster-labeled intents
  query_runbook    read-only   retrieval over the runbooks (needs the RAG store)
  apply_action     write       applies an action only when the verdict is ALLOW; dry-run by default

Real writes additionally require NETWORK_ORCH_ALLOW_APPLY=1 in the server's environment.
Transports: stdio (default) or streamable HTTP bound to 127.0.0.1.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

from network_orch.cascade import shield, slots
from network_orch.cascade.actions import Action, ActionKind
from network_orch.cascade.graded import ALLOW, DEFAULT, Policy, decide
from network_orch.config import SETTINGS

SERVER_NAME = "network_orch_mcp"
CHARACTER_LIMIT = 25_000
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
DNS_LABEL = r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$"
ZONE_FALLBACK = "edge-a"
AUDIT_PATH = Path(os.getenv("NETWORK_ORCH_AUDIT", "data/traces/mcp-audit.jsonl"))
BENCH_DIR = Path(os.getenv("NETWORK_ORCH_BENCHMARK", Path(__file__).resolve().parents[3] / "benchmark"))
BENCH_SETS = {"set1": "set1-oracle.jsonl", "set2": "set2-oracle.jsonl"}
NEXT_STEP = {
    "ALLOW": "Safe to apply with network_orch_apply_action.",
    "REPAIR": "Correct the parameter named in the reason and call network_orch_check_action again (at most 2 times).",
    "ASK": "Ask the operator for the missing or ambiguous parameter; do not guess.",
    "ESCALATE": "Request operator approval before any further step; do not apply.",
    "DENY": "Do not apply. This target is protected or the request was refused.",
}

mcp = FastMCP(SERVER_NAME)


class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


class ActionKindIn(str, Enum):
    SCALE = "scale"
    MIGRATE = "migrate"


class _Base(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")


class ActionInput(_Base):
    kind: ActionKindIn = Field(..., description="Action type: 'scale' (set replica count) or 'migrate' (move to another zone).")
    deployment: str = Field(default="streamer", pattern=DNS_LABEL, max_length=63,
                            description="Deployment name (DNS label), e.g. 'streamer'.")
    namespace: str = Field(default="streaming-demo", min_length=1, max_length=253,
                           description="Kubernetes namespace, e.g. 'streaming-demo'. Protected namespaces such as 'kube-system' are denied.")
    replicas: Optional[int] = Field(default=None, ge=0, le=100_000,
                                    description="Target replica count for 'scale', e.g. 3. Omit for 'migrate'.")
    to_zone: Optional[str] = Field(default=None, max_length=63,
                                   description="Destination zone for 'migrate', e.g. 'edge-b'. Omit for 'scale'.")
    from_zone: Optional[str] = Field(default=None, max_length=63,
                                     description="Current zone of the workload, if known, e.g. 'edge-a'. Defaults to 'edge-a'.")
    intent_text: Optional[str] = Field(default=None, max_length=2000,
                                       description="The operator's original request. Used to detect ambiguous phrasing such as 'double it'.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN,
                                            description="'markdown' for reading, 'json' for programmatic use.")


class ApplyInput(ActionInput):
    dry_run: bool = Field(default=True, description="When true (default) only report what would be applied. "
                                                    "Real writes also need NETWORK_ORCH_ALLOW_APPLY=1 on the server.")


class EmptyInput(_Base):
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'.")


class BenchmarkInput(_Base):
    dataset: str = Field(default="set2", pattern="^(set1|set2)$",
                         description="'set1' (used during design) or 'set2' (generated after the design was frozen).")
    kind: Optional[str] = Field(default=None, pattern="^(scale|migrate|monitor)$", description="Filter by action kind.")
    harm: Optional[int] = Field(default=None, ge=0, le=1, description="Filter by cluster-measured harm: 1 harmful, 0 safe.")
    verdict: Optional[str] = Field(default=None, pattern="^(ALLOW|REPAIR|ASK|ESCALATE|DENY)$",
                                   description="Keep only items whose ground-truth action gets this graded verdict.")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Items per page (1-50).")
    offset: int = Field(default=0, ge=0, description="Items to skip, from the previous response's next_offset.")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="'markdown' or 'json'.")


class RunbookInput(_Base):
    question: str = Field(..., min_length=3, max_length=500, description="Operational question, e.g. 'what to do when edge-b is saturated'.")
    top_k: int = Field(default=SETTINGS.rag_top_k, ge=1, le=10, description="Passages to return.")


# ---------------------------------------------------------------- helpers
def _policy() -> Policy:
    raw = os.getenv("NETWORK_ORCH_POLICY")
    if not raw:
        return DEFAULT
    try:
        return Policy.from_dict(json.loads(Path(raw).read_text()))
    except (OSError, ValueError, TypeError) as e:
        raise ValueError(f"NETWORK_ORCH_POLICY={raw} is not a readable JSON policy file ({type(e).__name__}).") from e


def _to_action(p: ActionInput) -> Action:
    return Action(ActionKind(p.kind.value), deployment=p.deployment, namespace=p.namespace,
                  replicas=p.replicas, to_zone=p.to_zone)


def _evaluate(p: ActionInput) -> dict[str, Any]:
    act = _to_action(p)
    pol = _policy()
    zone = p.from_zone or ZONE_FALLBACK
    why = slots.underspecified(p.intent_text) if p.intent_text else ""
    verdict, reason = decide(act, why, pol)
    strict = shield.check(act, zone)
    return {
        "verdict": verdict, "reason": reason, "next_step": NEXT_STEP[verdict],
        "strict_validator": {"ok": strict.ok, "violations": list(strict.violations)},
        "action": {"kind": act.kind.value, "deployment": act.deployment, "namespace": act.namespace,
                   "replicas": act.replicas, "to_zone": act.to_zone, "from_zone": zone},
    }


def _truncate(text: str) -> str:
    if len(text) <= CHARACTER_LIMIT:
        return text
    return text[:CHARACTER_LIMIT] + "\n\n[truncated: lower `limit` or add filters]"


def _error(e: Exception) -> str:
    """Actionable, non-leaky error text."""
    if isinstance(e, FileNotFoundError):
        return ("Error: benchmark data not found. Set NETWORK_ORCH_BENCHMARK to the directory containing "
                "set1-oracle.jsonl and set2-oracle.jsonl.")
    if isinstance(e, ValueError):
        return f"Error: {e}"
    if isinstance(e, PermissionError):
        return "Error: permission denied by the cluster. Check the server's service account."
    return f"Error: unexpected failure ({type(e).__name__}). Check the server log."


def _audit(tool: str, args: dict, result: dict) -> None:
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": tool, "args": args, "result": result}
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _fmt_eval(r: dict, fmt: ResponseFormat) -> str:
    if fmt is ResponseFormat.JSON:
        return json.dumps(r, indent=2)
    a, s = r["action"], r["strict_validator"]
    tgt = f"replicas={a['replicas']}" if a["kind"] == "scale" else f"to_zone={a['to_zone']} (from {a['from_zone']})"
    return "\n".join([
        f"## Verdict: {r['verdict']}", f"- Action: {a['kind']} {a['deployment']} in `{a['namespace']}`, {tgt}",
        f"- Reason: {r['reason'] or 'none'}", f"- Strict validator: {'ok' if s['ok'] else 'REFUSE'}"
        + (f" ({'; '.join(s['violations'])})" if s["violations"] else ""),
        f"- Next step: {r['next_step']}"])


def _load_bench(dataset: str) -> list[dict]:
    path = BENCH_DIR / BENCH_SETS[dataset]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if r.get("harm") is not None]


def _bench_verdict(it: dict) -> str:
    k = it["kind"]
    if k == "monitor":
        return ALLOW
    ns = it.get("namespace") or "streaming-demo"
    act = (Action(ActionKind.SCALE, namespace=ns, replicas=it.get("replicas")) if k == "scale"
           else Action(ActionKind.MIGRATE, namespace=ns, to_zone=it.get("to_zone")))
    return decide(act, slots.underspecified(it["text"]), _policy())[0]


# ------------------------------------------------------------------ tools
@mcp.tool(name="network_orch_check_action", annotations={
    "title": "Check an orchestration action", "readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False})
async def network_orch_check_action(params: ActionInput) -> str:
    """Check a proposed scale or migrate action against the safety policy without touching the cluster.

    Returns one of five verdicts: ALLOW (safe to apply), REPAIR (a parameter is out of range or unknown;
    the reason says how to fix it), ASK (missing or ambiguous parameter: ask the operator), ESCALATE
    (scale to zero: needs approval) or DENY (protected namespace or refusal). Also reports whether the
    stricter validator would have refused it. Use before network_orch_apply_action. Does not modify anything.

    Args:
        params (ActionInput): kind ('scale'|'migrate'), deployment, namespace, replicas (scale),
            to_zone (migrate), from_zone, intent_text (the operator's words), response_format.

    Returns:
        str: markdown, or JSON {verdict, reason, next_step, strict_validator:{ok, violations[]},
        action:{kind, deployment, namespace, replicas, to_zone, from_zone}}.
        On bad configuration: "Error: <message>".

    Examples:
        - "Is scaling streamer to 50 allowed?" -> kind='scale', replicas=50 -> REPAIR (range is 1-8)
        - "Move it to edge-d" -> kind='migrate', to_zone='edge-d' -> REPAIR (unknown zone)
        - "Shut it down" -> kind='scale', replicas=0 -> ESCALATE
        - Don't use to execute anything; use network_orch_apply_action for that.
    """
    try:
        return _fmt_eval(_evaluate(params), params.response_format)
    except Exception as e:  # noqa: BLE001 - converted to an actionable tool error
        return _error(e)


@mcp.tool(name="network_orch_get_policy", annotations={
    "title": "Get the enforced policy", "readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False})
async def network_orch_get_policy(params: EmptyInput) -> str:
    """Return the limits network_orch_check_action enforces.

    Args:
        params (EmptyInput): response_format ('markdown'|'json').

    Returns:
        str: JSON {min_replicas, max_replicas, zones[], protected_namespaces[], namespaces[]|null}
        (`namespaces` null means no allowlist), or markdown. "Error: ..." if NETWORK_ORCH_POLICY is unreadable.
    """
    try:
        p = _policy()
    except Exception as e:  # noqa: BLE001
        return _error(e)
    d = {"min_replicas": p.min_replicas, "max_replicas": p.max_replicas, "zones": list(p.zones),
         "protected_namespaces": list(p.protected_namespaces),
         "namespaces": list(p.namespaces) if p.namespaces is not None else None}
    if params.response_format is ResponseFormat.JSON:
        return json.dumps(d, indent=2)
    return "\n".join(["## Policy", f"- Replicas: {d['min_replicas']} to {d['max_replicas']}",
                      f"- Zones: {', '.join(d['zones'])}", f"- Protected namespaces: {', '.join(d['protected_namespaces'])}",
                      f"- Namespace allowlist: {', '.join(d['namespaces']) if d['namespaces'] else 'none'}"])


@mcp.tool(name="network_orch_get_cluster_state", annotations={
    "title": "Get cluster state", "readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": True})
async def network_orch_get_cluster_state(params: EmptyInput) -> str:
    """Read a snapshot of nodes and workloads in the managed namespace. Does not modify anything.

    Args:
        params (EmptyInput): response_format ('markdown'|'json').

    Returns:
        str: JSON {"nodes": [...], "workloads": [...]} or a markdown summary.
        "Error: ..." when the cluster is unreachable (check kubeconfig, or set K8S_DRY_RUN=true for a canned state).
    """
    from network_orch.k8s.client import ClusterBackend
    try:
        st = await asyncio.to_thread(lambda: ClusterBackend(namespace=SETTINGS.namespace, dry_run=SETTINGS.dry_run).get_state())
    except Exception as e:  # noqa: BLE001
        return "Error: cannot read the cluster (" + type(e).__name__ + "). Check kubeconfig and cluster status, or set K8S_DRY_RUN=true."
    if params.response_format is ResponseFormat.JSON:
        return _truncate(json.dumps({"nodes": st.nodes, "workloads": st.workloads}, indent=2))
    lines = [f"## Cluster state (`{SETTINGS.namespace}`)", f"- Nodes: {len(st.nodes)}", f"- Workloads: {len(st.workloads)}"]
    lines += [f"  - {w.get('deployment')}: zone {w.get('zone')}, replicas {w.get('replicas')}" for w in st.workloads]
    return _truncate("\n".join(lines))


@mcp.tool(name="network_orch_list_benchmark", annotations={
    "title": "List labeled benchmark intents", "readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False})
async def network_orch_list_benchmark(params: BenchmarkInput) -> str:
    """List intents from the cluster-labeled benchmark, with the graded verdict of each ground-truth action.

    Each item has the operator text, the structured action, and `harm`, measured by applying the action to a
    real Deployment. Use it to see what kinds of requests the checks allow, repair or block.

    Args:
        params (BenchmarkInput): dataset ('set1'|'set2'), kind, harm (0|1), verdict, limit (1-50, default 20),
            offset, response_format.

    Returns:
        str: JSON {total, count, offset, has_more, next_offset, items:[{text, kind, replicas, to_zone,
        namespace, harm, verdict}]} or a markdown list. "Error: ..." if the data directory is missing.

    Examples:
        - "Which harmful scale intents does the policy only repair?" -> kind='scale', harm=1, verdict='REPAIR'
        - Don't use to judge a new action; use network_orch_check_action.
    """
    try:
        rows = await asyncio.to_thread(_load_bench, params.dataset)
        items = []
        for r in rows:
            if params.kind and r["kind"] != params.kind:
                continue
            if params.harm is not None and r["harm"] != params.harm:
                continue
            v = _bench_verdict(r)
            if params.verdict and v != params.verdict:
                continue
            items.append({"text": r["text"], "kind": r["kind"], "replicas": r.get("replicas"), "to_zone": r.get("to_zone"),
                          "namespace": r.get("namespace"), "harm": r["harm"], "verdict": v})
    except Exception as e:  # noqa: BLE001
        return _error(e)
    page = items[params.offset: params.offset + params.limit]
    nxt = params.offset + len(page)
    out = {"total": len(items), "count": len(page), "offset": params.offset, "has_more": nxt < len(items),
           "next_offset": nxt if nxt < len(items) else None, "items": page}
    if params.response_format is ResponseFormat.JSON:
        return _truncate(json.dumps(out, indent=2))
    if not page:
        return "No items match these filters. Remove a filter or lower `offset`."
    lines = [f"## Benchmark {params.dataset}: {len(page)} of {len(items)} (offset {params.offset})"]
    lines += [f"- [{i['verdict']}, harm={i['harm']}] {i['kind']} ns={i['namespace']} replicas={i['replicas']} zone={i['to_zone']}: {i['text'][:140]}"
              for i in page]
    if out["has_more"]:
        lines.append(f"\nMore results: call again with offset={nxt}.")
    return _truncate("\n".join(lines))


@mcp.tool(name="network_orch_query_runbook", annotations={
    "title": "Search the runbooks", "readOnlyHint": True, "destructiveHint": False,
    "idempotentHint": True, "openWorldHint": False})
async def network_orch_query_runbook(params: RunbookInput) -> str:
    """Search the curated runbooks and topology notes. Does not modify anything.

    Args:
        params (RunbookInput): question (3-500 chars), top_k (1-10).

    Returns:
        str: JSON list of passages, or "Error: ..." if the runbook index has not been built.
    """
    try:
        from network_orch.rag.store import query as rag_query
        hits = await asyncio.to_thread(rag_query, params.question, top_k=params.top_k, persist_dir=SETTINGS.chroma_dir)
        return _truncate(json.dumps(hits, indent=2))
    except Exception as e:  # noqa: BLE001
        return f"Error: runbook search unavailable ({type(e).__name__}). Build the index first (see docs) or skip this tool."


@mcp.tool(name="network_orch_apply_action", annotations={
    "title": "Apply a checked action", "readOnlyHint": False, "destructiveHint": True,
    "idempotentHint": True, "openWorldHint": True})
async def network_orch_apply_action(params: ApplyInput) -> str:
    """Apply a scale or migrate action to the cluster, but only if network_orch_check_action returns ALLOW.

    Any other verdict is returned unchanged and nothing is applied. Dry-run is the default: it reports the
    change without making it. A real write also requires NETWORK_ORCH_ALLOW_APPLY=1 on the server. Every call
    is appended to an audit log. Scaling or migrating a workload changes live traffic; always check first.

    Args:
        params (ApplyInput): the same fields as network_orch_check_action, plus dry_run (default true).

    Returns:
        str: JSON {applied: bool, dry_run: bool, verdict, reason, next_step, result}. `applied` is true only
        after a real write. "Error: ..." if the cluster call fails.

    Examples:
        - kind='scale', replicas=3, dry_run=true -> shows the planned change
        - kind='scale', replicas=50 -> REPAIR, not applied; fix the replica count and retry
        - Don't use to explore; use network_orch_check_action.
    """
    try:
        r = _evaluate(params)
        out: dict[str, Any] = {"applied": False, "dry_run": params.dry_run, "verdict": r["verdict"],
                               "reason": r["reason"], "next_step": r["next_step"], "result": None}
        if r["verdict"] == ALLOW:
            a = r["action"]
            plan = (f"scale {a['deployment']} -> {a['replicas']} replicas in {a['namespace']}" if a["kind"] == "scale"
                    else f"migrate {a['deployment']} -> zone {a['to_zone']} in {a['namespace']}")
            allowed = os.getenv("NETWORK_ORCH_ALLOW_APPLY") == "1"
            if params.dry_run or not allowed:
                out["dry_run"] = True
                out["result"] = {"status": "dry-run", "action": plan}
                if not params.dry_run and not allowed:
                    out["next_step"] = "Writes are disabled on this server. Set NETWORK_ORCH_ALLOW_APPLY=1 to enable them."
            else:
                from network_orch.k8s.client import ClusterBackend

                def _go() -> dict:
                    be = ClusterBackend(namespace=a["namespace"], dry_run=False)
                    return (be.scale_workload(a["deployment"], a["replicas"]) if a["kind"] == "scale"
                            else be.migrate_workload(a["deployment"], a["to_zone"]))
                out["result"] = await asyncio.to_thread(_go)
                out["applied"] = out["result"].get("status") == "ok"
        _audit("apply_action", params.model_dump(mode="json", exclude={"response_format"}), out)
        return json.dumps(out, indent=2)
    except Exception as e:  # noqa: BLE001
        return _error(e)


def main() -> None:
    ap = argparse.ArgumentParser(prog=SERVER_NAME, description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1", help="bind address for streamable-http (keep it local)")
    ap.add_argument("--port", type=int, default=SETTINGS.mcp_port)
    a = ap.parse_args()
    if a.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host, mcp.settings.port = a.host, a.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
