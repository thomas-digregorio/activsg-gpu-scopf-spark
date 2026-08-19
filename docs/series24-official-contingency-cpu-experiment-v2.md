# Series24 mapped-official-contingency CPU experiment v2

## Replacement scope

This replacement preserves the failed Series24 all-eligible-branch Case 1 run
and assigns new benchmark identities. It runs the six Series24 MATPOWER cases
in source order with laptop CPU HiGHS, a `1e-3` relative MIP-gap target, and a
hard 1,800-second end-to-end limit per scenario.

Every scenario launches in a new Python worker and a new HiGHS instance. Round
1 is cold. Later constraint-generation rounds may reuse only the preceding
commitment from the same scenario. No result, basis, model, or MIP start crosses
scenario boundaries. Fixed-commitment pricing is outside this experiment.

## Contingency identity

The Series24 archive has no MATPOWER contingency table. This experiment uses
the branch-outage entries from the earlier TAMU/MATPOWER
`contab_ACTIVSg2000.m`, SHA-256
`198b39f0381925a4ddacbe2148973cb1d93ddfe220303829cf87b16d45190bba`.
The table was created for `case_ACTIVSg2000.m`, SHA-256
`8d00618de8fd10bf35a599f59d2deebfecd0d86e28fcff73219ad7c4ebab860b`.

Old branch row numbers are not applied directly. The versioned
`endpoint_parameter_assignment_v1` mapping:

1. groups reference and target circuits by their unordered endpoint bus IDs;
2. assigns each reference circuit one-to-one to the closest same-orientation
   target circuit using resistance, reactance, charging, `RATE_A`, tap, phase
   shift, and angle limits;
3. fails closed on a missing endpoint multiplicity, reversed assignment,
   non-equivalent minimum-cost tie, or normalized distance above `0.5`;
4. maps the official branch change rows while leaving generator outages
   deferred; and
5. keeps every target branch in base modeling and monitoring, even when it is
   not an outage candidate.

All 3,206 reference branches map one-to-one in every scenario: 3,082 are exact
parameter matches and 124 are documented parameter updates. The official table
contains 3,190 unique branch contingencies; 16 reference branches have no
branch-outage entry. Assignment ties occur only among physically equivalent
parallel circuits.

| Scenario | Target branches | Target additions not outage candidates | Valid mapped non-islanding outages |
|---|---:|---:|---:|
| 2016 summer peak | 3,220 | 14 | 2,740 |
| 2016 low load | 3,220 | 14 | 2,740 |
| 2024 summer peak | 3,911 | 705 | 2,785 |
| 2024 low load | 3,913 | 707 | 2,785 |
| 2024 high renewables | 3,663 | 457 | 2,795 |
| 2024 low load with GFM | 3,913 | 707 | 2,785 |

The increase above 2,740 in the 2024 cases occurs because new network circuits
make some mapped reference outages non-islanding. The new circuits themselves
are not outage candidates.

## Model and acceptance

The mathematical model otherwise remains unchanged: exact scenario `PMIN` and
`PMAX`, source-offline generators unavailable, one common preventive dispatch,
ten source-derived PWL cost segments, strict `RATE_A`, no corrective redispatch,
and no feasibility slack.

Success requires all of the following in the same run:

- a feasible incumbent;
- a HiGHS-certified relative MIP gap no larger than `1e-3`;
- a final exhaustive mapped-contingency screen with no violation above `1e-5`
  p.u.; and
- an independent verification pass after rereading and remapping the immutable
  raw sources.

Before the replacement benchmark was frozen, a component preflight applied the
mapped Case 1 catalog to the preserved base-only incumbent. It found 510
violated pairs, and the corresponding cold continuous round-2 relaxation was
declared infeasible by HiGHS. This is not an official benchmark result. Case 1
is nevertheless launched once as requested so its full registered result is
preserved; the remaining scenarios are not consumed if Case 1 fails the
required feasibility gate. The compact diagnostic evidence is preserved in
`reports/texas2k-series24-official-preflight-v2.json`.
