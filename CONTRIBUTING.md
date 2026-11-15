# Contributing

- `make install && make test && make lint` must pass. `make bench` must report 0 harmful actions passed.
- A new check or rule needs a test in `tests/` and a run against `benchmark/` (`make bench`).
- New harm labels must come from the cluster oracle (`evals/k8s_oracle_gen.py`) with the 3x majority re-check
  (`evals/recheck_gen.py`), not from a rule.
- Report negative results. The gate, cache and decision-model code stays because it did not help, and the
  evidence is in the evals.
- Run the oracle on a throwaway local cluster only.
