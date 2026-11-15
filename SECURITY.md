# Security

This is research code. Do not rely on it as the only control in front of a production cluster.

The cluster oracle scales and migrates Deployments and writes to the `oracle` namespace; run it only on a
throwaway local cluster with the restricted service account in `k8s/oracle.yaml`.

Report a bypass of the checks (a proposal that passes them and causes harm) by opening an issue with the
proposal and the cluster outcome.
