# Changelog

## Unreleased
- Graded verdicts (`cascade/graded.py`): ALLOW, REPAIR, ASK, ESCALATE, DENY with a bounded repair loop and a configurable `Policy`.
- Cluster oracle, native admission-policy baseline, and the arm runners used for the evaluation.
- `benchmark/`: 1,242 intents with cluster-measured harm labels and an offline scorer.
- MCP server `network_orch_mcp` (6 tools, stdio and streamable HTTP, dry-run-first apply) with tests and a 10-question evaluation.
- CLI (`python -m network_orch`), examples, lint and benchmark regression test in CI.
