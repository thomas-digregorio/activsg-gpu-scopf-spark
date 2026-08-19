# ACTIVSg2000 GPU Lagrangian v11

This is the compact evidence index for the one-shot DGX Spark v11 run. The
17,973,629-byte raw result remains on the Spark and in the ignored local
results directory; it is intentionally excluded from Git.

## Frozen identity

- Commit: `9eed34bbdac6a21ef6fee3713fb277635bbd17f1`
- Tag: `experiment-2000-gpu-lagrangian-v11`
- Config SHA-256: `257bf44aed3ab76f4d723cbc2b3d32d4e1fb487947cc8aa5bdef5d691c650e44`
- Raw result SHA-256: `209a11ddf6b423d6e1b2199aca3319340af7bf159ddfdbf07f5d6ddbb96735d9`
- Spark raw result: `/home/dgxsparktd/activsg-gpu-scopf-spark/results/experiments/activsg2000-gpu-lagrangian-v11-dgx-spark.json`

## Outcome

The run stopped cleanly with
`incomplete_refinement_budget_reserve_reached`. Total end-to-end wall time was
`756.1708413610031` seconds. The secure objective was
`1142133.3767435683`, the independently replayed lower bound was
`1128529.212256944`, and the certified relative gap was
`0.011911187225271595`. The requested `0.001` gap was **not** certified.

The returned commitment has 331 online generators. Independent raw-input
verification passed exact conditional PMIN/PMAX, integrality, objective
reconstruction, DC physics, base limits, and all 17,563,400 valid contingency
sides. Maximum model residual was `2.3442225938197226e-12` p.u. and maximum
security violation was `3.489135451673064e-9` p.u.

All 544 source generator rows, MW and p.u. dispatch, and all 2,000
fixed-commitment LP bus prices are retained in the raw result. Pricing is
certified for the returned fixed commitment; it is not a MILP dual price.

## Numerical attribution

The v10 defects addressed by v11 did not recur:

- the full 9,220-column canonical incumbent passed the feasibility precheck;
- cuOpt accepted the translated 9,215-column native MIP start with presolve
  disabled and returned a matching incumbent;
- there were zero MIP-start rejections, zero barrier warnings, zero
  free-variable warnings, and no missing-vector failure;
- the secure primal and the three-region lower-bound frontier both passed
  independent raw-input replay;
- a late child transaction could not consume the verification reserve.

The remaining bottleneck is child PDLP convergence. The native console emitted
27 coefficient-range warnings. Six unique long child cost-LP attempts reached
their short limits without meeting canonical primal acceptance. Two split
transactions rolled back safely. In one of those, the fallback Phase I found
an optimal zero-violation feasible point, but v11 did not feed that point back
into a restarted cost LP. The next revision should run Phase I before these
child cost LPs, use its feasible primal as the cost-LP start, and conservatively
remove numerically immaterial feasibility-cut coefficients.

## CPU boundary

The laptop run took `1007.3702709000063` seconds and certified a
`0.0007763998859633739` gap. V11 used 251.199 seconds less wall time, but this
is **not** a comparable speedup because the GPU run failed the requested gap
gate.
