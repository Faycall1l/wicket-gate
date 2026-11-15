# Results with 95% Wilson intervals

Harm is measured on the cluster; counts are intents (harmful n, safe n). Refused = refused, asked or escalated.

## Fresh set, seed 31

| Arm | Harmful reaching harm | 95% CI (%) | Safe refused | 95% CI (%) |
|---|---|---|---|---|
| Agent alone | 120/196 | [54.2, 67.8] | 7/204 | [1.7, 6.9] |
| + strict validator | 0/196 | [0.0, 1.9] | 45/204 | [16.9, 28.2] |
| + graded checks | 0/196 | [0.0, 1.9] | 8/204 | [2.0, 7.5] |

## Original subset, seed 11

| Arm | Harmful reaching harm | 95% CI (%) | Safe refused | 95% CI (%) |
|---|---|---|---|---|
| Agent alone | 134/201 | [59.9, 72.8] | 11/199 | [3.1, 9.6] |
| + strict validator | 0/201 | [0.0, 1.9] | 58/199 | [23.3, 35.8] |
| + graded checks | 0/201 | [0.0, 1.9] | 14/199 | [4.2, 11.5] |

One backbone (`gpt-oss-20b`), one workload, one local cluster. Seed 11 is the subset the graded rules were designed on; seed 31 is the set generated after the design was frozen.
