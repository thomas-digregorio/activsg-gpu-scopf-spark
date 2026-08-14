# ACTIVSg2000 DGX Spark 30-minute PDLP v7 result

The one authorized v7 run ended
`incomplete_restricted_master_gap_not_certified` after
1805.749 seconds. It is not infeasible. Its final
incumbent is independently N-1 secure, but the requested `1e-3` MIP gap was
not certified, so the overall result remains incomplete and pricing is
intentionally withheld.

Round 1 started cold. Rounds 2 and 3 each received the prior GPU solution's
432 integer commitment columns; no
CPU commitment, CPU dispatch, or CPU bound initialized the run. V7 preserves
the v6 exact-PMIN model, inputs, tolerances, dynamic add-resolve-screen loop,
and cuOpt PDLP policy. Its substantive runtime change was increasing the
cumulative cuOpt allowance from 900 to 1,800 seconds.

Every round requested and read back method 1 (PDLP), Stable3 mode 4, FP64
precision 1, batched PDLP strong branching, batched PDLP reliability
branching, and reliability factor 1. The native log confirmed
`Cooperative batch PDLP and Dual Simplex for strong branching` in all three
rounds. This means PDLP assisted MIP branching; it does not mean every
branch-and-bound relaxation used PDLP alone.

Round 1 took 5.571 adapter seconds and added
173 contingency pairs. Round 2 took
39.215 seconds and added
14 more. Round 3 took
1757.532 seconds and stopped on cuOpt's time limit
after exploring 1,373,905 nodes. Its exhaustive screen checked
17,563,400 sides and found zero violations above
`1e-5` p.u. The post-run independent raw-input checker also passed all
17,563,400 sides, with maximum security
violation 5.982e-12 p.u. and maximum
model residual 1.614e-09 p.u.

The final objective is 1,133,078.528383, the finite lower bound is
1,130,841.046502, and the reported relative gap is 0.0019747.
That gap is above `0.001`. The recorded restricted-master total was
1802.319 seconds, which is
2.319 seconds above the nominal 1,800-second allowance because
cuOpt returned slightly after its final native time limit. The full controller
still completed inside the separate 1,935-second outer guard. This overrun is
retained as evidence rather than normalized away.

Compared with v6, v7's incumbent objective is
+166.233757, its lower bound is
+225.670790, and its reported gap changes by
-0.0000528. V7 commits
330 units and dispatches
67,109.210000 MW. The fresh parallel MIP path
has 32 commitment flips versus v6
(17 off-to-on and
15 on-to-off) and an L1 dispatch difference
of 5,744.037172 MW. This is
not a fixed-master timing comparison: the dynamic pair paths differ.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows, v6 and
v7 commitment/dispatch values in MW and p.u., and blank price fields with the
withholding reason. `bus-prices.csv` preserves all 2,000 bus identities with
blank prices. The other files retain round timing, requested/read-back PDLP
parameters, event and native logs, hashes, and independent verification. No
v7 retry was performed.
