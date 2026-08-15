# ACTIVSg2000 DGX Spark PDLP-policy v6 result

The one authorized v6 run ended
`incomplete_restricted_master_gap_not_certified` after
902.887 seconds. It is not a successful SCOPF
result, but it is not infeasible. The final incumbent is independently secure.
Round 1 started cold; the adapter submitted the prior GPU solution's
432 commitment columns in rounds 2
and 3. Retrospective native-log inspection found that cuOpt rejected both
starts after internal model expansion. No CPU commitment, dispatch, or lower
bound initialized the run, and the rejected starts are not counted as reuse.

Relative to v5, v6 changed only the registered cuOpt policy and frozen identity.
Every round selected method 1 (PDLP), Stable3 mode 4, FP64 precision 1, batched
PDLP strong branching, batched PDLP reliability branching, and reliability
factor 1. Requested and read-back values matched exactly. The native log
confirmed `Cooperative batch PDLP and Dual Simplex for strong branching` in all
three rounds. This evidence means PDLP assisted the MIP branching work; it does
not mean every branch-and-bound relaxation used PDLP alone.

The dynamic loop added 173 contingency pairs
after round 1 and 18 after round 2. Round 3's
exhaustive screen found zero violations above `1e-5` p.u. The independent raw-
input checker passed all 17,563,400 sides with
maximum security violation 2.827e-10
p.u. and maximum model residual
2.048e-09 p.u.

The sole failed acceptance gate is the requested MIP gap. Round 3 returned
objective 1,132,912.294626, finite bound 1,130,615.375711, and gap
0.0020274, above `0.001`. It reached its time limit after
817.447 native seconds. Total restricted-master
time was 899.652 seconds inside the
900-second cumulative allowance; end-to-end wall time stayed below 1,035
seconds. Fixed-commitment pricing is therefore intentionally withheld.

Compared with v5, v6 improved the final objective by
27.493744, improved the bound by
88.245250, and reduced the reported relative gap
from 0.0021296 to 0.0020274. That is
still insufficient for the requested certificate. Both solutions commit
328 units and serve
67,109.210000 MW, but
18 commitment decisions differ
(9 off-to-on and
9 on-to-off).

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows, both v5
and v6 secure-but-gap-uncertified commitments/dispatches, MW and p.u. values,
and blank price fields with the withholding reason. `bus-prices.csv` preserves
all 2,000 bus identities with blank prices. The other files retain parameter
readbacks, round timing, event/native logs, hashes, verification, and the v5
comparison. No v6 retry was performed.
