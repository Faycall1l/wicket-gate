"""Kubernetes client wrapper with first-class dry-run.

Dry-run mode (K8S_DRY_RUN=true or ClusterBackend(dry_run=True)) performs zero
cluster I/O and returns the manifest diff that WOULD be applied. This is what
makes `demo --dry-run` work on any machine and what the validator shows for
approval before live execution.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field


@dataclass
class ClusterState:
    nodes: list[dict] = field(default_factory=list)
    workloads: list[dict] = field(default_factory=list)


class ClusterBackend:
    def __init__(self, namespace: str = "streaming-demo", dry_run: bool = False) -> None:
        self.namespace = namespace
        self.dry_run = dry_run
        self._apps = None
        self._core = None
        if not dry_run:
            from kubernetes import client, config

            try:
                config.load_kube_config()
            except Exception:
                config.load_incluster_config()
            self._apps = client.AppsV1Api()
            self._core = client.CoreV1Api()

    # -- reads -----------------------------------------------------------
    def get_state(self) -> ClusterState:
        if self.dry_run:
            return ClusterState(
                nodes=[
                    {"name": "k3d-network-orch-server-0", "zone": "cloud", "ready": True},
                    {"name": "k3d-network-orch-agent-0", "zone": "edge-a", "ready": True},
                    {"name": "k3d-network-orch-agent-1", "zone": "edge-b", "ready": True},
                    {"name": "k3d-network-orch-agent-2", "zone": "edge-c", "ready": True},
                ],
                workloads=[
                    {"deployment": "streamer", "replicas": 2, "ready": 2, "zone": "edge-a"},
                ],
            )
        nodes = self._core.list_node().items
        workloads: list[dict] = []
        for d in self._apps.list_namespaced_deployment(self.namespace).items:
            zone = (d.spec.template.spec.node_selector or {}).get(
                "topology.network-orch/zone", "unknown"
            )
            workloads.append(
                {
                    "deployment": d.metadata.name,
                    "replicas": d.spec.replicas,
                    "ready": d.status.ready_replicas or 0,
                    "zone": zone,
                }
            )
        return ClusterState(
            nodes=[
                {
                    "name": n.metadata.name,
                    "zone": (n.metadata.labels or {}).get("topology.network-orch/zone", "?"),
                    "ready": any(
                        c.type == "Ready" and c.status == "True" for c in (n.status.conditions or [])
                    ),
                }
                for n in nodes
            ],
            workloads=workloads,
        )

    # -- writes (validated upstream; dry-run returns planned diff) -------
    def scale_workload(self, deployment: str, replicas: int) -> dict:
        if self.dry_run:
            return {
                "status": "dry-run",
                "action": f"scale {deployment} -> {replicas} replicas in {self.namespace}",
            }
        body = {"spec": {"replicas": replicas}}
        r = self._apps.patch_namespaced_deployment_scale(deployment, self.namespace, body)
        return {"status": "ok", "replicas": r.spec.replicas}

    def migrate_workload(self, deployment: str, to_zone: str) -> dict:
        """Migration = patch pod-template nodeSelector; ReplicaSet reschedules.

        The Service selector is unchanged so traffic follows the new pods —
        that is the service-continuity mechanism being demonstrated.
        """
        if self.dry_run:
            return {
                "status": "dry-run",
                "action": (
                    f"migrate {deployment} -> zone {to_zone} "
                    f"(patch nodeSelector topology.network-orch/zone)"
                ),
            }
        body = {
            "spec": {
                "template": {
                    "spec": {"nodeSelector": {"topology.network-orch/zone": to_zone}}
                }
            }
        }
        self._apps.patch_namespaced_deployment(deployment, self.namespace, body)
        return {"status": "ok", "zone": to_zone}

    # -- recovery primitives (rollback / canary) -------------------------
    def _annotate(self, deployment: str, annotations: dict) -> None:
        self._apps.patch_namespaced_deployment(
            deployment, self.namespace, {"metadata": {"annotations": annotations}}
        )

    def _current_zone(self, deployment: str) -> str | None:
        d = self._apps.read_namespaced_deployment(deployment, self.namespace)
        return (d.spec.template.spec.node_selector or {}).get(
            "topology.network-orch/zone"
        )

    def _wait_ready(
        self, deployment: str, min_ready: int, timeout: float = 240.0, poll: float = 2.0
    ) -> bool:
        """Poll ready_replicas until >= min_ready or timeout. Health gate used
        by canary steps and by rollback."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            d = self._apps.read_namespaced_deployment(deployment, self.namespace)
            if (d.status.ready_replicas or 0) >= min_ready:
                return True
            time.sleep(poll)
        return False

    def canary_migrate(self, deployment: str, to_zone: str, step_size: int = 1,
                       gate_timeout: float = 240.0) -> dict:
        """Make-before-break canary migration with a per-step health gate.

        Creates a shadow ``{deployment}-canary`` Deployment on ``to_zone``
        (same app labels and pod spec, so the Service routes to both), waits
        for it to become Ready, then steps the original Deployment down one
        ``step_size`` at a time. After all traffic is served from the canary
        the original is patched onto ``to_zone``, scaled back up, and the
        shadow is deleted — the Deployment name and Service selector never
        change. Before moving anything, the pre-migration zone and replica
        count are recorded in annotations for ``rollback()``.
        """
        if self.dry_run:
            return {
                "status": "dry-run",
                "action": (
                    f"canary_migrate {deployment} -> {to_zone} "
                    f"(shadow {deployment}-canary, step {step_size}, "
                    f"health-gate each step)"
                ),
            }
        src_zone = self._current_zone(deployment)
        if src_zone == to_zone:
            return {"status": "ok", "note": "no-op (already on target zone)"}
        orig = self._apps.read_namespaced_deployment(deployment, self.namespace)
        total = orig.spec.replicas or 1
        self._annotate(deployment, {
            "network-orch/last-known-zone": src_zone,
            "network-orch/last-replicas": str(total),
        })

        canary = f"{deployment}-canary"
        # shadow Deployment: clone the pod template, retarget the zone.
        shadow = copy.deepcopy(orig)
        shadow.metadata = copy.deepcopy(orig.metadata)
        shadow.metadata.name = canary
        shadow.metadata.resource_version = None
        shadow.metadata.uid = None
        shadow.metadata.creation_timestamp = None
        shadow.metadata.annotations = {}
        shadow.metadata.labels = dict(orig.metadata.labels or {})
        shadow.spec.replicas = 0
        shadow.spec.selector.match_labels = {"app": deployment}
        shadow.spec.template.metadata.labels = {
            "app": deployment, "canary": canary,
        }
        shadow.spec.template.spec.node_selector = {
            "topology.network-orch/zone": to_zone
        }
        try:
            self._apps.delete_namespaced_deployment(canary, self.namespace)
        except Exception:
            pass
        self._apps.create_namespaced_deployment(self.namespace, shadow)

        served = 0
        steps = []
        try:
            while served < total:
                step = min(step_size, total - served)
                # bring more canary capacity online (health-gated)...
                self._apps.patch_namespaced_deployment_scale(
                    canary, self.namespace, {"spec": {"replicas": served + step}}
                )
                if not self._wait_ready(canary, served + step, timeout=gate_timeout):
                    raise RuntimeError(
                        f"canary health gate failed at {served + step}/{total} "
                        f"(zone {to_zone})"
                    )
                # ...then peel the same amount off the original.
                self._apps.patch_namespaced_deployment_scale(
                    deployment, self.namespace,
                    {"spec": {"replicas": total - served - step}},
                )
                served += step
                steps.append(served)
            # all traffic is on the shadow: retarget the original and restore.
            self.migrate_workload(deployment, to_zone)
            self._apps.patch_namespaced_deployment_scale(
                deployment, self.namespace, {"spec": {"replicas": total}}
            )
            if not self._wait_ready(deployment, total):
                raise RuntimeError("original deployment did not come Ready on target")
            self._apps.delete_namespaced_deployment(canary, self.namespace)
            return {
                "status": "ok",
                "zone": to_zone,
                "replicas": total,
                "steps": steps,
                "note": f"last-known-good recorded: {src_zone}",
            }
        except Exception:
            # leave the canary intact so rollback() can clean it up.
            self._annotate(deployment, {"network-orch/canary-failed": "true"})
            raise

    def rollback(self, deployment: str, last_known_good: str | None = None) -> dict:
        """Rollback a migration to the last known good zone.

        ``last_known_good`` defaults to the zone recorded by
        ``canary_migrate()`` (annotation ``network-orch/last-known-zone``,
        replicas from ``network-orch/last-replicas``). Restores the original
        Deployment's nodeSelector and replica count, waits for it to become
        Ready (the recovery health gate), then removes any leftover canary
        shadow so the Service serves only from the recovered pods.
        """
        if self.dry_run:
            return {
                "status": "dry-run",
                "action": (
                    f"rollback {deployment} -> {last_known_good or '<last-known>'}"
                ),
            }
        d = self._apps.read_namespaced_deployment(deployment, self.namespace)
        ann = d.metadata.annotations or {}
        zone = last_known_good or ann.get("network-orch/last-known-zone")
        if not zone:
            raise RuntimeError("no last-known-good zone recorded for rollback")
        replicas = int(ann.get("network-orch/last-replicas", str(d.spec.replicas or 1)))
        # First take any live traffic off the canary (it may be serving).
        canary = f"{deployment}-canary"
        try:
            self._apps.read_namespaced_deployment(canary, self.namespace)
            self._apps.patch_namespaced_deployment_scale(
                canary, self.namespace, {"spec": {"replicas": 0}}
            )
        except Exception:
            pass
        # Restore the original controller to its pre-migration state.
        cur_zone = self._current_zone(deployment)
        if cur_zone != zone:
            self.migrate_workload(deployment, zone)
        d2 = self._apps.read_namespaced_deployment(deployment, self.namespace)
        cur = d2.spec.replicas or 1
        if cur != replicas:
            self._apps.patch_namespaced_deployment_scale(
                deployment, self.namespace, {"spec": {"replicas": replicas}}
            )
        if not self._wait_ready(deployment, replicas):
            raise RuntimeError(f"rollback did not reach Ready on {zone}")
        try:
            self._apps.delete_namespaced_deployment(canary, self.namespace)
        except Exception:
            pass
        return {"status": "ok", "zone": zone, "replicas": replicas}
