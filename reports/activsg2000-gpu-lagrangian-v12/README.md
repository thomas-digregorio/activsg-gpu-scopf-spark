# ACTIVSg2000 GPU Lagrangian v12

This directory is the compact evidence index for the one-shot DGX Spark v12
run. The 22,403,487-byte raw result remains on the Spark and in the ignored
local results directory; it is intentionally excluded from Git.

## Frozen identity

- Commit: `3e74ee5684b7c9439263e82bd86079d9e2920d9e`
- Tag: `experiment-2000-gpu-lagrangian-v12`
- Config SHA-256: `fa482a28fe2d88da357317828aeeb3b29a6062badc8a65c185c63cb58ea189ac`
- Raw result SHA-256: `6b7b7254eb0f0d6b32081684e1b1fd33d3ae223e54257bb0dc88ec90e46cc24f`
- Spark raw result: `/home/dgxsparktd/activsg-gpu-scopf-spark/results/experiments/activsg2000-gpu-lagrangian-v12-dgx-spark.json`

## Outcome

The run stopped cleanly with
`incomplete_refinement_budget_reserve_reached`. End-to-end wall time was
`876.0529307219986` seconds. The independently verified secure objective was
`1142133.3767435683`, the independently replayed lower bound was
`1128529.212256944`, and the certified relative gap was
`0.011911187225271595`. The requested `0.001` gap was not certified.

The returned 331-unit commitment passed exact conditional PMIN/PMAX,
integrality, objective reconstruction, DC physics, base limits, and all
17,563,400 valid contingency sides. Maximum model residual was
`2.3442225938197226e-12` p.u.; maximum security violation was
`3.489135451673064e-9` p.u. All 13 frontier certificates replayed with zero
difference.

## Numerical fixes proved by v12

The earlier child failure did not recur. The corrected real-size probe and the
one-shot run showed:

- Phase I preserves the authoritative feasible primal after each exhaustive
  contingency screen;
- the cost-PDLP result is used only as a possible dual seed, never as the
  primal;
- both the complete Phase-I primal and row-identity-mapped parent dual were
  submitted and read back at the exact native dimensions;
- the Phase-I security loop screens its final solved round instead of solving
  and discarding it;
- all 24 child prechecks completed, 12 split transactions committed, and zero
  split transactions failed;
- cuOpt reported no MIP-start rejection for the complete 9,220-column
  incumbent path.

The one-shot run therefore failed only the requested gap gate, not a
numerical, feasibility, security, replay, or deadline-controller gate.

## Remaining lower-bound bottleneck

The 24 bounded cost-dual attempts consumed `314.14615181903355` seconds.
Eleven produced the selected child seed; 13 were weaker than the mapped parent
certificate. After 12 successful splits, two frontier leaves still retained
the exact root bound and the same Lagrangian-minimizing commitment. Their best
commitment-cut multipliers remained all zero. This identifies the next target:
robust GPU optimization of the disjunctive commitment-cut multipliers, rather
than more ordinary PDLP retries or numerical relaxation.

The accounted solver, GPU, verification, and replay stages total
`714.60763348396` seconds. The remaining `161.445297238039` seconds is
controller, model-copy, and repeated checkpoint overhead and is a secondary
throughput target.

## CPU boundary

The laptop comparison took `1007.3702709000063` seconds and certified a
`0.0007763998859633739` gap. V12 finished `131.3173401780077` seconds sooner,
but this is not a comparable speedup because the GPU run did not certify the
requested gap. CPU commitment, dispatch, objective, and bound were not used by
the GPU algorithm; the comparison was attached only after the GPU status was
final.

