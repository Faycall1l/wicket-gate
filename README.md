# wicket-gate

Safety checks for LLM agents that scale and migrate workloads on an edge cluster, measured against harm observed on a real Kubernetes cluster.

![Where 400 unseen intents end up](docs/assets/flow.png)

## The problem

Ask an agent to "double the streamer, then triple it" and it may return 50 replicas. Ask it to park the workload
in a zone that does not exist and it will invent a name. A validator catches both. The usual validator then
refuses the request, and the operator, who asked for something reasonable, gets nothing.

On 400 intents the agent alone caused harm in 120 of 196 harmful cases. A strict validator removed all of it and
also refused 22% of the safe requests. This repository holds a middle design and the measurements behind it.

## What it does

A proposal from the agent passes through deterministic checks (no model in the loop) and comes out as one of five
verdicts:

| Verdict | When | What happens |
|---|---|---|
| ALLOW | in range, known zone, not protected | applied |
| REPAIR | replica count out of range, unknown zone | the reason goes back to the agent, at most two retries |
| ASK | missing parameter, or phrasing such as "double it" | the operator clarifies |
| ESCALATE | scale to zero | the operator approves |
| DENY | protected namespace, or the agent refused | refused |

A rejected proposal is routed, not simply refused. The limits are a `Policy` object you can change.

### Where it sits

![system structure](docs/assets/structure.png)

## What we found

![harm and refusals per arm](docs/assets/results.png)

On 400 LLM-written intents generated after the design was frozen (196 harmful, 204 safe, `gpt-oss-20b`), with harm
taken from running each executed action on a Deployment in a k3d cluster:

| | Harmful intents that reach harm | Safe intents refused or escalated |
|---|---|---|
| Agent alone | 120 / 196 | 7 / 204 |
| + strict validator | 0 / 196 | 45 / 204 |
| + graded checks | 0 / 196 | 8 / 204 |

A second seed on the original 400-intent subset, the one the rules were designed on, gives the same picture:
134 / 201 harmed for the agent alone, 0 / 201 for both checks, with 58 / 199 refused by the strict validator and
14 / 199 refused or escalated by the graded one. Intervals are in [`docs/results.md`](docs/results.md).

Three further results, from the first set of 800 intents:
- A Kubernetes-native baseline (RBAC plus two `ValidatingAdmissionPolicy` objects) gives the same verdicts as the strict validator on all 792 labeled items. What the cluster does not offer is the routing.
- A conformal gate, a decision cache and a small learned decision model did not improve safety or refusals over the validator alone. They stay in `src/` as documented negative results.
- A single run per action overstated harm: one transient readiness miss looked like a harmful action, so every in-range harm label is a majority of three reruns.

Limits: one workload on one local cluster, labels that depend on a 60 s readiness deadline, LLM-written intents, one
model backbone, and rules designed after seeing the first set's failures (the second set is the check on that).

## Use it

The Python package is `network_orch`; the Kubernetes node label and k3d cluster names keep the `network-orch` prefix.

```bash
make install && make test
uv run python -m network_orch scale 50        # strict: REFUSE   graded: REPAIR
uv run python -m network_orch scale 0         #                  graded: ESCALATE
uv run python benchmark/evaluate.py           # score both checks on the labeled benchmark, offline
```
![terminal demo](docs/assets/demo.gif)

### In Python
```python
from network_orch.cascade.graded import Policy, guard

policy = Policy(max_replicas=20, zones=("site-1", "site-2"), namespaces=("media",))
decision = guard("park the streamer in site-2", propose, policy=policy)   # propose(text) -> Action | None
decision.verdict    # ALLOW | REPAIR | ASK | ESCALATE | DENY
```
`Policy` fields: `min_replicas`, `max_replicas`, `zones`, `protected_namespaces`, and an optional `namespaces`
allowlist (off by default, since the published results were measured without it). A runnable version is in
[`examples/quickstart.py`](examples/quickstart.py).

### As an MCP server

`network_orch_mcp` exposes the checks to any MCP client (Claude Desktop, Claude Code, an agent framework).

| Tool | Effect |
|---|---|
| `network_orch_check_action` | read-only: verdict, reason and next step for a proposed action |
| `network_orch_get_policy` | read-only: the enforced limits |
| `network_orch_get_cluster_state` | read-only: nodes and workloads |
| `network_orch_list_benchmark` | read-only: paginated, filterable labeled intents |
| `network_orch_query_runbook` | read-only: runbook search |
| `network_orch_apply_action` | applies an action only on ALLOW; dry-run by default; real writes need `NETWORK_ORCH_ALLOW_APPLY=1` |

```
uv run python -m network_orch.mcp_server.server                       # stdio
uv run python -m network_orch.mcp_server.server --transport streamable-http --port 8000   # local HTTP
```
```json
{"mcpServers": {"network_orch": {"command": "uv", "args": ["run", "--directory", "/path/to/wicket-gate",
  "python", "-m", "network_orch.mcp_server.server"], "env": {"K8S_DRY_RUN": "true"}}}}
```
Set `NETWORK_ORCH_POLICY=/path/policy.json` to change the limits (`max_replicas`, `zones`, `namespaces`, ...).
Every apply call is appended to an audit log (`NETWORK_ORCH_AUDIT`). `evals/mcp/network_orch_mcp_eval.xml` holds ten
read-only evaluation questions with verified answers.

### When not to use it
Not as the only control. It checks the proposal, not the cluster: pair it with RBAC and admission policy (the
native baseline is `k8s/native-policy.yaml`). It knows nothing about capacity, cost or dependencies between
services, and it was evaluated on one workload.

## Reproduce the cluster results

Needs Docker (colima or Docker Desktop), `k3d`, `kubectl`, and a Groq or OpenAI-compatible key in `.env`.
```bash
k3d cluster create --config k8s/k3d-cluster.yaml && kubectl apply -f k8s/oracle.yaml
uv run python evals/gen_llm_intents.py --calls 80 --per 10 --seed 202 --out llmgen2-intents.jsonl
uv run python evals/k8s_oracle_gen.py --inp llmgen2-intents.jsonl --out llmgen2-oracle.jsonl
uv run python evals/recheck_gen.py llmgen2-oracle.jsonl
LLMGEN=llmgen2-oracle.jsonl uv run python evals/llm_arms.py --backend groq --t0 bge-small+LR --n 400 --seed 31 --splits llmgen
LLMGEN=llmgen2-oracle.jsonl uv run python evals/graded_arm.py --seed 31
LLMGEN=llmgen2-oracle.jsonl SEED=31 uv run python evals/k8s_oracle_exec.py
```
Free-tier API quotas are the main constraint; runs resume from checkpoints in `evals/results/`. The oracle scales
and migrates Deployments in the `oracle` namespace, so run it on a throwaway local cluster.

## Repository map

| Path | Contents |
|---|---|
| `src/network_orch/cascade/graded.py` | verdicts, `Policy`, repair loop |
| `src/network_orch/cascade/shield.py`, `policies/` | the strict validator and its constants |
| `src/network_orch/cascade/` (rest) | gate, cache, decision model: the negative-result cascade |
| `src/network_orch/mcp_server/` | MCP server |
| `evals/` | cluster oracle, native-policy baseline, arm runners, harm re-check |
| `benchmark/` | 1,242 labeled intents (`set1` used for design, `set2` generated after) and an offline scorer |
| `k8s/` | k3d config, oracle workload, native admission policies |
| `docs/` | verdict table, results with intervals, figures |

## Related work

- **Runtime enforcement for agents.** [AgentSpec](https://arxiv.org/abs/2503.18666) (rules with triggers, predicates and enforcement actions) and [Progent](https://arxiv.org/abs/2504.11703) (programmable privilege control over tool calls) block unsafe calls. [CaMeL](https://arxiv.org/abs/2503.18813) separates control and data flow by design. [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) wraps conversations in programmable rails.
- **Critique-and-retry loops.** [LLM-Modulo](https://arxiv.org/abs/2402.01817) pairs an LLM with external verifiers and feeds their critiques back, which is the repair step here.
- **Kubernetes-side policy.** [ValidatingAdmissionPolicy](https://kubernetes.io/docs/reference/access-authn-authz/validating-admission-policy/), [OPA Gatekeeper](https://github.com/open-policy-agent/gatekeeper) and [Kyverno](https://kyverno.io) are the native layer; we use the first as our baseline.
- **Kubernetes agents.** [kagent](https://github.com/kagent-dev/kagent), [k8sgpt](https://github.com/k8sgpt-ai/k8sgpt) and [kubernetes-mcp-server](https://github.com/containers/kubernetes-mcp-server) give agents cluster access; guardrails for them are an open area ([kagent's kgateway guardrails](https://kagent.dev/blog/kgateway-guardrails)).

What this repo adds is a measurement: the same agent, the same intents, harm taken from a real cluster, and
the refusal cost of each way of checking.

## Citation and license

MIT (see `LICENSE`). Research code, not a supported product; do not use it as the only control in front of a
production cluster.

```bibtex
@software{amrouche2026wicketgate,
  title  = {wicket-gate: graded safety checks for LLM agents on edge clusters},
  author = {Amrouche, Faycal},
  year   = {2026}
}
```
