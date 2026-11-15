"""End-to-end tests of network_orch_mcp over a real stdio MCP client session (no cluster needed)."""
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def run(coro):
    return asyncio.run(coro)


async def _session(fn, env=None):
    params = StdioServerParameters(command=sys.executable, args=["-m", "network_orch.mcp_server.server"],
                                   env={**os.environ, "K8S_DRY_RUN": "true", **(env or {})})
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            return await fn(s)


def call(tool, args, env=None):
    async def go(s):
        res = await s.call_tool(tool, {"params": args})
        return res.content[0].text
    return run(_session(go, env))


def test_tool_surface_and_annotations():
    async def go(s):
        return (await s.list_tools()).tools
    tools = {t.name: t for t in run(_session(go))}
    assert set(tools) == {"network_orch_check_action", "network_orch_get_policy", "network_orch_get_cluster_state",
                          "network_orch_list_benchmark", "network_orch_query_runbook", "network_orch_apply_action"}
    for n, t in tools.items():
        assert t.annotations is not None, n
        assert t.annotations.readOnlyHint == (n != "network_orch_apply_action"), n
    assert tools["network_orch_apply_action"].annotations.destructiveHint is True


def test_check_action_verdicts():
    for args, verdict in [({"kind": "scale", "replicas": 3}, "ALLOW"), ({"kind": "scale", "replicas": 50}, "REPAIR"),
                          ({"kind": "scale", "replicas": 0}, "ESCALATE"), ({"kind": "migrate", "to_zone": "mars"}, "REPAIR"),
                          ({"kind": "scale", "replicas": 3, "namespace": "kube-system"}, "DENY"),
                          ({"kind": "migrate"}, "ASK")]:
        out = json.loads(call("network_orch_check_action", {**args, "response_format": "json"}))
        assert out["verdict"] == verdict, (args, out)


def test_invalid_input_is_rejected_not_executed():
    async def go(s):
        res = await s.call_tool("network_orch_check_action", {"params": {"kind": "delete"}})
        return res.isError, res.content[0].text
    is_error, text = run(_session(go))
    assert is_error and "kind" in text


def test_apply_is_dry_run_by_default_and_gated():
    out = json.loads(call("network_orch_apply_action", {"kind": "scale", "replicas": 3}))
    assert out["verdict"] == "ALLOW" and out["applied"] is False and out["result"]["status"] == "dry-run"
    out = json.loads(call("network_orch_apply_action", {"kind": "scale", "replicas": 3, "dry_run": False}))
    assert out["applied"] is False and "NETWORK_ORCH_ALLOW_APPLY" in out["next_step"]
    out = json.loads(call("network_orch_apply_action", {"kind": "scale", "replicas": 50, "dry_run": False}))
    assert out["verdict"] == "REPAIR" and out["applied"] is False and out["result"] is None


def test_benchmark_pagination_and_filters():
    out = json.loads(call("network_orch_list_benchmark", {"dataset": "set2", "harm": 1, "limit": 5, "response_format": "json"}))
    assert out["total"] == 217 and out["count"] == 5 and out["has_more"] and out["next_offset"] == 5
    assert all(i["harm"] == 1 for i in out["items"])
    last = json.loads(call("network_orch_list_benchmark", {"dataset": "set2", "harm": 1, "limit": 50, "offset": 200, "response_format": "json"}))
    assert last["count"] == 17 and last["has_more"] is False and last["next_offset"] is None
    # no harmful ground-truth action may get ALLOW
    allowed = json.loads(call("network_orch_list_benchmark", {"dataset": "set2", "harm": 1, "verdict": "ALLOW", "response_format": "json"}))
    assert allowed["total"] == 0


def test_policy_from_file(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"max_replicas": 20, "zones": ["site-1"]}))
    out = json.loads(call("network_orch_get_policy", {"response_format": "json"}, env={"NETWORK_ORCH_POLICY": str(p)}))
    assert out["max_replicas"] == 20 and out["zones"] == ["site-1"]
    out = json.loads(call("network_orch_check_action", {"kind": "scale", "replicas": 12, "response_format": "json"},
                          env={"NETWORK_ORCH_POLICY": str(p)}))
    assert out["verdict"] == "ALLOW"
