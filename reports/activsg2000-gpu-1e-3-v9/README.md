# ACTIVSg2000 DGX Spark corrected-start v9 result

The single authorized v9 run completed cleanly as
`deadline_budget_exhausted` after 1805.720
end-to-end seconds. It is not infeasible, but it is also not a successful
N-1 SCOPF result. The final solved master has objective
1,132,724.812824, lower bound 1,130,678.342896, and relative
gap 0.0018067, above the requested `0.001`.

V9 fixed the rejected-start defect. Before each later solve, a bounded HiGHS
LP fixed the prior GPU commitment and tested whether continuous redispatch
could satisfy the expanded master. Round 2's prior commitment was proven
infeasible under the 173 new rows, so round 2 solved cold. Round 3's precheck
did not produce an acceptable feasible completion, so round 3 also solved
cold. No start reached cuOpt in the official run, and the native audit records
zero MIP-start rejections and zero barrier warnings. There was no external CPU
initialization.

The separately approved exact translation diagnostic proves the usable-start
path itself: all 9,220 canonical values were translated into exactly 11,214
native values using 1,999 free-variable splits and five fixed/unused-column
eliminations. Presolve was disabled, native readback passed, and cuOpt printed
`Adding initial solution success!`. The unextendable prior-round diagnostic
separately proved the round-1 commitment infeasible for the round-2 master and
correctly solved that diagnostic master cold.

Round 1 solved in 5.673 adapter seconds and its
exhaustive screen added 173 pairs. Round 2 solved in
36.096 seconds, certified a
0.0009479 restricted-master gap, then exposed and added
13 more pairs. Round 3 used the remaining
1759.889 seconds, ending at gap
0.0018067. Its mandatory exhaustive screen checked
17,563,400 sides and exposed two more rows:
`c0675_m0681_lower` and `c0676_m0680_lower`. The maximum violation was
0.054596464 p.u. The rows were added
deterministically, but no solve budget remained for round 4.

The post-run independent raw-input checker repeated all 17,563,400 security
sides in 1.344 seconds. Base limits, exact
conditional source PMIN/PMAX, balance, DC equations, angles, integrality, and
the objective passed; maximum model residual was
3.648e-09 p.u. It independently found
the same maximum N-1 violation, so overall verification failed exactly at the
security gate.

The provisional incumbent commits 326
units and dispatches 67,109.210000 MW.
`generator-detail.csv` preserves all 544 source rows with exact PMIN/PMAX and
provisional commitment/dispatch in MW and p.u. Pricing is intentionally
withheld because neither the requested gap nor N-1 security was certified;
`bus-prices.csv` preserves all 2,000 bus identities with blank price fields.

Post-run inspection found one conservative telemetry defect in the HiGHS
precheck: an `Unknown` status could label a finite but grossly infeasible
vector as `has_incumbent`, although the independent residual gate still
rejected it and solved cold. Version 0.18.1 now also requires HiGHS' explicit
feasible-primal status before exposing or reusing a vector. This did not alter
the frozen v9 solve or its safe cold-start decision. No retry was performed.
