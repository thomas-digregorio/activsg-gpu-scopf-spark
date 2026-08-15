# ACTIVSg2000 MIP-gap sensitivity v1

The bounded campaign stopped at `1e-4`, exactly as registered. The `1e-3` run
completed `optimal_verified` with accepted fixed-commitment pricing in
1007.370 seconds. The `1e-4` run reached its
solver budget in round 2 after 1665.851
seconds with an incumbent gap of 1.461313e-04, above its requested
`1e-4`. Therefore `1e-5`, `1e-6`, and `1e-7` were not started.

| Gap | Status | Objective | Bound | Gap | Committed | Wall (s) | Pricing |
|---:|---|---:|---:|---:|---:|---:|---|
| 1e-3 | `optimal_verified` | 1,133,479.385501 | 1,132,599.352235 | 7.763999e-04 | 327 | 1007.370 | accepted |
| 1e-4 | `incomplete_restricted_master_not_optimal` | 1,131,340.383410 | 1,131,175.059173 | 1.461313e-04 | 325 provisional | 1665.851 | not reached |
| 1e-5 | not started | N/A | N/A | N/A | N/A | N/A | N/A |
| 1e-6 | not started | N/A | N/A | N/A | N/A | N/A | N/A |
| 1e-7 | not started | N/A | N/A | N/A | N/A | N/A | N/A |

## Accepted 1e-3 grid result

The accepted result commits 327 of the
432
source-online generators and dispatches 67,109.210000 MW
(671.092100 p.u.). Its objective is
1,133,479.385501, bound is 1,132,599.352235, and achieved
gap is 7.763999e-04. Three dynamic constraint-generation rounds
added 106 pairs. The final exhaustive violation
is 7.893e-10 p.u., the maximum independently
checked model residual is 2.421e-09 p.u., and
all 17,563,400 contingency sides
passed the independent checker.

The fixed-commitment pricing LP is independently N-1 secure. Its dispatch differs
from the MIP dispatch by at most
0.275199 MW and
0.751171 MW in aggregate. Prices range from
$-1,941.226809/MWh to $890.865626/MWh, with mean
$15.537987/MWh. On the 100 MVA base, the range is
$-194,122.680909 to $89,086.562615 per p.u.-hour.

## Why 1e-4 is not comparable as a final grid solution

Round 1 added 87 violated pairs. Round 2 used the prior commitment as a partial
MIP start, processed 76,511
nodes, and stopped with HiGHS `TimeLimit`. A provisional exhaustive screen of
that incumbent then found 18 more violated pairs, with maximum violation
0.203989 p.u. Those pairs were not
promoted into a new accepted master because round 2 had not met `1e-4`.
Independent verification and pricing were therefore not run.

For diagnostic context only, the provisional incumbent has
325 committed units, differs from the accepted `1e-3`
commitment at 32 source rows, and
has 6,143.939809 MW of
absolute dispatch difference. These are not accepted `1e-4` modeling results.

## Evidence files

- `generator-detail.csv`: all 544 exact source PMIN/PMAX rows, accepted `1e-3`
  commitment/MIP dispatch/pricing dispatch/price, and the clearly labeled
  provisional `1e-4` commitment/dispatch. The `1e-4` pricing fields are blank
  because no valid pricing solve occurred.
- `bus-prices.csv`: all 2,000 accepted `1e-3` nodal prices in $/MWh and
  $/p.u.-hour. The `1e-4` price fields are blank.
- `round-detail.csv`: solve, MIP-start, screening, and disposition evidence for
  every recorded internal round.
- `summary.csv` and `comparison.json`: campaign status, raw-result hashes,
  numerical checks, and the explicit stop gate.

The source-record audit found zero PMIN/PMAX substitutions, p.u. conversion
mismatches, conditional limit violations above `1e-6` MW, unavailable-generator
violations, or price conversion mismatches. ACTIVSg2000 is synthetic; this result
does not establish a suitable gap for a different case or formulation.
