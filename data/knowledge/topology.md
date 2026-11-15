# Topology + SLOs (ground truth for RAG)

## Zones and labels
The zones that exist in this deployment are `cloud`, `edge-a`, `edge-b`,
and `edge-c`. The cloud zone runs on the k3d server node; edge-a, edge-b and
edge-c run on k3d agent-0, agent-1 and agent-2. Select a zone for a workload
with the node label `topology.network-orch/zone`, e.g. via the `nodeSelector`
field `topology.network-orch/zone: edge-a`.
- Demo: Deployment `streamer` (nginx) ns `streaming-demo`, Service `streamer:80`.

## Latency SLOs
The p95 latency SLO is 50 milliseconds. The target success rate is greater
than 99.5% during migration windows and scaling windows. A latency above 75%
of the 50 ms SLO (37.5 ms) counts as degradation risk that may trigger a
proactive migration.

## Replica bounds and protected namespaces
Replica bounds are 1 to 8: never scale below 1 replica and never above 8.
The protected namespaces that agents must never touch are `kube-system`,
`kube-public`, and `kube-node-lease`. Scaling and migration tools validate
these bounds before executing anything.

## Cost model and abstention
Cost model (report §6-7): weigh migration cost against latency benefit and
only migrate when the benefit exceeds the cost. Abstain when confidence is
below threshold (0.55): prefer read-only state queries over low-confidence
writes, and let the validator reject anything unsafe.
