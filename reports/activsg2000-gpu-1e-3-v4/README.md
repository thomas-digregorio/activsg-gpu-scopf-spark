# ACTIVSg2000 DGX Spark scaled dynamic v4 result

The one authorized v4 run ended
`incomplete_restricted_master_gap_not_certified` after
1668.494 seconds. It is not a successful SCOPF
result, but it is also not infeasible and it did not reproduce the earlier
numerical failure.

The per-unit cuOpt formulation completed three add-resolve-screen rounds. Round
1 added 349 contingency pairs; round 2 added 17; round 3's exhaustive screen
found zero violations above `1e-5` p.u. The final incumbent independently
passed all 17,563,400 sides with maximum
security violation 5.912e-14 p.u.

The sole failed acceptance gate is the requested MIP gap. Round 3 returned
objective 1,133,240.316367, finite bound 1,130,745.878288, and gap
0.0022012, above the requested `0.001`. Therefore the result remains
incomplete and fixed-commitment prices are intentionally withheld.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows and the
gap-uncertified but independently secure round-3 commitment/dispatch, including
MW and p.u. values. It commits 330 generators. `bus-prices.csv`
contains all 2,000 bus identities with blank price fields and the reason prices
were withheld. `round-detail.csv`, `summary.csv`, and `evidence.json` preserve
the round timings, finite bounds/gaps, hashes, acceptance gates, and post-run
independent verification.

The v4 launcher enabled cuOpt console logging, but the one-shot parent captured
and then discarded successful-worker stdout; `launcher-console.log` therefore
contains only the final wrapper line. Native statuses, residuals, nodes, and
iterations remain serialized for every round, and `events.jsonl` preserves the
complete round timeline. The controller is corrected after this frozen run to
persist captured worker output for future launches. That correction does not
alter or rerun v4.

No v4 retry was performed.
