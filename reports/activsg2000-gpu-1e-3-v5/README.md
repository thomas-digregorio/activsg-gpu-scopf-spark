# ACTIVSg2000 DGX Spark 15-minute dynamic v5 result

The one authorized v5 run ended
`incomplete_restricted_master_gap_not_certified` after
902.938 seconds. It is not a successful SCOPF
result, but it is not infeasible. Round 1 started cold. Rounds 2 and 3 each
had 432 prior-GPU commitment columns
submitted by the adapter, but retrospective native-log inspection found that
cuOpt rejected both starts after internal model expansion. No CPU commitment,
dispatch, or lower bound initialized this run. This historical defect is
preserved rather than relabeled as successful start reuse.

The frozen controller reserved 900 seconds for raw loading and cumulative cuOpt
rounds after retaining 120 seconds for verification and 15 seconds for
serialization. Actual restricted-master solve time was
899.782 seconds and end-to-end wall time
was 902.938 seconds, below the 1,035-second hard
boundary.

The per-unit cuOpt formulation completed three add-resolve-screen rounds. Round
1 added 349 contingency pairs; round 2 added
14; round 3's exhaustive screen found zero
violations above `1e-5` p.u. The final incumbent independently passed all
17,563,400 sides with maximum security
violation 1.290e-13 p.u. and maximum
model residual 1.286e-10 p.u.

The sole failed acceptance gate is the requested MIP gap. Round 3 returned
objective 1,132,939.788370, finite bound 1,130,527.130461, and gap
0.0021296, above the requested `0.001`. cuOpt reached its time limit
after 851.770 native seconds and
852.377 adapter seconds. Therefore the result
remains incomplete and fixed-commitment prices are intentionally withheld.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows and the
gap-uncertified but independently secure round-3 commitment/dispatch in MW and
p.u. It commits 328 generators and dispatches
67,109.210000 MW. `bus-prices.csv` preserves all 2,000 bus identities
with blank price fields and the reason prices were withheld. `worker-console.log`
contains the complete native cuOpt stream for all three rounds. The remaining
files preserve round timing, hashes, the event timeline, and independent
verification.

No v5 retry was performed.
