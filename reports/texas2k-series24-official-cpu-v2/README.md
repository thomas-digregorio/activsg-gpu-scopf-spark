# Series24 mapped-official-contingency CPU v2 result

The requested replacement used the earlier TAMU/MATPOWER
`contab_ACTIVSg2000.m` branch-outage set mapped by physical identity into the
Series24 topology. The 2016 summer-peak gate failed as
`infeasible_restricted_master`, so the remaining five cold one-shot runs were
not consumed.

| Series24 scenario | Official run | End-to-end time (s) | Result |
|---|---|---:|---|
| 2016 summer peak | completed once | 9.365810 | `infeasible_restricted_master` |
| 2016 low load | not started | N/A | gated by Case 1 infeasibility |
| 2024 summer peak | not started | N/A | gated by Case 1 infeasibility |
| 2024 low load | not started | N/A | gated by Case 1 infeasibility |
| 2024 high renewables | not started | N/A | gated by Case 1 infeasibility |
| 2024 low load with GFM | not started | N/A | gated by Case 1 infeasibility |

## Case 1 evidence

- Frozen commit: `010b969f81cb221050bc892d4471bca6bcfa5c2a`
- Frozen tag: `experiment-texas2k-series24-official-cpu-v2`
- Configuration SHA-256:
  `9bbf871176aeba782773c09245ec12edff319001451541f144464047be4b5f0b`
- Raw result SHA-256:
  `71a7a35c191276b3082637a2dff0f43504b097d3aa4bcbc77be8bba4eb0cf0a6`
- Mapped contingency profile: 3,190 official unique branch entries, 2,740
  valid non-islanding outages, and 450 bridge exclusions.
- Target mapping: 3,082 exact parameter matches, 124 documented parameter
  updates, and 14 Series24 additions retained for modeling and monitoring but
  omitted as outage candidates.
- Round 1: HiGHS `Optimal`, objective `1104090.04437519`, bound
  `1103560.12734206`, gap `4.7995816629624e-4`, and solve time `7.426396` s.
- Exhaustive mapped screen: 510 violated pairs, maximum pair
  `c1525_m1544_lower`, maximum violation `2.20768265855073` p.u., and screen
  time `0.156706` s.
- Round 2: HiGHS `Infeasible` in `0.012922` solver seconds
  (`0.019004` adapter seconds), with approximately `1656.31` seconds of solver
  budget still available. Presolve found infeasibility before any B&B node or
  LP iteration.

The prior-round commitment was submitted as a MIP start. HiGHS first found
that commitment infeasible for the new rows and then independently presolved
the unrestricted round-2 model as infeasible. The separately preserved cold
continuous-relaxation preflight reached the same status. This was not a
30-minute timeout and not merely rejection of the MIP start.

No final exhaustive security verification can run because round 2 produced no
feasible incumbent. The result does not claim an independently replayed Farkas
certificate; it records the native HiGHS infeasibility status and the separate
cold-relaxation corroboration.
