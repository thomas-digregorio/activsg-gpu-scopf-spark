# ACTIVSg2000 GPU Lagrangian v24

This is the compact evidence index for the one-shot DGX Spark v24 run. The
46,409,615-byte raw result remains on the Spark and in the ignored local
results directory; it is intentionally excluded from Git.

## Frozen identity

- Commit: `ae5f6ee893a49f59bd49f0b4a6b11b4594c67b6f`
- Tag: `experiment-2000-gpu-lagrangian-v24`
- Config SHA-256: `ba9449c2f1a055b1a5e74755aafb94389cc5db2f48563952484e2f313325e632`
- Raw result SHA-256: `fd03b0156d0a7271cdabd57af8ffc459ed1aaae8830802b32e742bec8285b1ce`
- Spark raw result: `/home/dgxsparktd/activsg-gpu-scopf-spark/results/experiments/activsg2000-gpu-lagrangian-v24-dgx-spark.json`

The tagged v23 revision was stopped at its GPU component gate before a full
run because CuPy rejected NumPy-style tuple keys for `lexsort`. V24 uses one
stacked key array and passed exact GPU/host commitment and bound replay before
this one-shot run was launched.

## Outcome

The run stopped cleanly with
`incomplete_refinement_budget_reserve_reached`. End-to-end wall time was
`873.003703919996` seconds. The secure objective was `1142133.305099282`, the
independently replayed lower bound was `1130040.7470977695`, and the certified
relative gap was `0.010587694052456742`. The requested `0.001` gap was **not**
certified.

The returned commitment has 331 online generators. Independent raw-input
verification passed exact conditional source PMIN/PMAX, integrality,
objective reconstruction, DC physics, base limits, and all 17,563,400 sides
for 2,740 valid branch outages. Maximum model residual was
`2.460183168295771e-12` p.u.; maximum contingency violation was
`4.674901411760857e-6` p.u. All generator commitment/dispatch values and all
2,000 fixed-commitment LP bus prices remain in the raw result.

## Numerical correction

The numerical defects targeted by v24 did not recur. Across 120 native solver
invocations in the worker log, there were zero large-coefficient-range
advisories, barrier numerical warnings, free-variable warnings, MIP-start
rejections, infeasible status strings, or NaNs. The native matrix ranges
reported by cuOpt were no wider than `[2e-6, 1]`, compared with 21 coefficient
range advisories in v22. Every registered solve used the read-back primal
tolerance of either `1e-8` or the stricter Phase-I `1e-10`.

Both proof authorities passed independently. The secure primal passed the
raw-input exhaustive N-1 checker. The 27-leaf lower-bound frontier rebuilt
from raw inputs with 32 feasibility cuts, 34 cover cuts, and two analytic
capacity cuts; its recorded and replayed global bounds agreed exactly. The
direct commitment-cut replay cache needed four security-master builds and
served 23 hits.

## Remaining bottleneck

The child cost-PDLP bottleneck was removed: v24 spent zero seconds there.
Exact hard-cardinality evaluation consumed only `2.2765` seconds across the
final leaves, and 26 leaves obtained positive local bound lift. However, the
minimum leaf followed an all-at-least path through 26 disjoint cardinality
cuts and received zero lift at the inherited multipliers. That one leaf pinned
the global bound. More of the same branching is therefore unlikely to close
the gap efficiently; the next design needs a stronger GPU-resident commitment
master or multiplier update on the minimum path.

The largest measured blocks were `250.7295` seconds in relaxation PDLP,
`106.8475` seconds in child Phase I, `88.8507` seconds in fixed-commitment
PDLP, `67.2474` seconds in final/root raw-input replay, and `56.1287` seconds
in exact-minimizer separation. Controller/model-build/checkpoint overhead not
represented by those non-overlapping timers was about `290.7411` seconds.

## CPU boundary

The registered laptop run took `1007.3702709000063` seconds and certified a
`0.0007763998859633739` gap. V24 used `134.3665669800103` fewer seconds
(`13.34%` less wall time), but this is **not** a successful speedup: the laptop
met the requested gap and v24 did not. V24 was also 65.254 seconds slower and
had a worse gap than v22, despite eliminating v22's numerical warnings.
