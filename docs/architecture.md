# Architecture

![system structure](assets/structure.png)

![where intents end up](assets/flow.png)

The agent proposes a structured action. A deterministic check routes it:

| Verdict | Condition | What happens |
|---|---|---|
| ALLOW | in range, known zone, not protected (same-zone migration is a no-op) | applied |
| REPAIR | replicas outside [1,8], unknown zone | reason returned to the agent, at most 2 retries |
| ASK | missing parameter, or multiplier phrasing ("double it") | operator clarifies |
| ESCALATE | scale to zero | operator approves |
| DENY | protected namespace, agent refusal | refused |

The strict validator (`cascade/shield.py`) refuses every non-ALLOW case. The labeled harm of an applied
action comes from the cluster oracle (`evals/k8s_oracle.py`), not from these rules.
