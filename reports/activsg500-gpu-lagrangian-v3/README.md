# ACTIVSg500 GPU Lagrangian v3 result

The one authorized v3 DGX Spark run ended after 545.521630918 seconds with
status `deadline_budget_exhausted`. It did not certify the requested `1e-3`
gap, so this is not a successful solve. No integer solver, CPU
branch-and-bound, or custom CUDA kernel was used.

## What the correction fixed

V3 preserved the root certificate before attempting any integer commitment.
The root continuous relaxation reached a zero-new-violation exhaustive screen
after adding 167 security pairs. Its FP64 GPU Lagrangian certificate retained a
conservative lower bound of 77,393.120159841, and the independent DGX raw-input
replay passed with zero reported global-bound difference.

The bounded candidate queue also behaved as designed:

| Candidate | Units | Change from prior | Result | Wall seconds |
|---|---:|---|---|---:|
| Root PDLP rounded at 0.5 | 49 | first | rejected for dual divergence | 5.110312 |
| GPU Lagrangian minimizer | 49 | identical hash | skipped duplicate | 0 |
| Root PDLP thresholded at 0.25 | 50 | source row 17 turned on | internally secure | 0.918269 |

The third candidate has the same commitment SHA-256 as the registered laptop
HiGHS solution and the same objective to sub-nanodollar precision:
79,410.651432183. Its canonical residual was `9.21e-14` p.u.; its internal
exhaustive GPU screen evaluated 401,704 sides, found no new pair, and reported
maximum violation `2.84e-15` p.u. Thus the candidate policy reduced the prior
bad repair from roughly 540 seconds to 5.11 seconds and found the known secure
50-unit commitment one candidate later.

This evidence is still called *internally secure*, not independently verified.
The controller kept the solution only in memory while it refined the bound and
did not promote its dispatch and prices into the durable payload before the
later deadline. The commitment rows, hash, objective, residual, and screen are
preserved, but GPU dispatch and pricing vectors are absent. The previously
published CPU generator and price tables remain references; they are not
relabeled as GPU output.

## What failed next

The root gap was 2.540630553%, so disjunctive refinement split on generator
source row 16. Transactional persistence worked: while child `r0` was pending,
the complete parent remained the certified frontier. Child round 1 solved in
0.486 seconds, then found two additional violations:

- `c0225_m0230_upper`
- `c0226_m0229_upper`

A raw-case matrix rebuild shows that both become the exact same inequality,
`pg_g0017 <= 322`. The corresponding two base rows are also parallel,
`0.5 pg_g0017 <= 322`. Adding both pair rows therefore introduced an exact
linear dependency in an already degenerate leaf.

The warm-start dimensional contract was correct: cuOpt received 672 primal
coordinates and expanded the prior 1,978-row dual to 1,980 entries with exactly
two zeros. Nevertheless, five subsequent PDLP slices all hit their time limits.
The first four residuals stayed near 0.3627 p.u. while dual objectives stayed
near `2.4e20`; the final shortened slice deteriorated further. The child
adapter consumed 534.027672 seconds, or 97.893% of the full run. It never
produced a primal-feasible vector, so none of those iterates was screened and
the split was never committed.

This is consistent with exact-row degeneracy and possibly an infeasible child,
but it is not an infeasibility proof. A valid later design must either produce
a replayable GPU Phase-I/Farkas certificate for pruning or abandon the proposed
split while retaining its parent. Merely timing out or observing a divergent
dual cannot delete the leaf.

## CPU comparison

| System | Status | Objective | Bound | Gap | Wall seconds |
|---|---|---:|---:|---:|---:|
| Laptop HiGHS | optimal, independently verified | 79,410.651432 | 79,410.651432 | 0 | 2.702380 |
| DGX Spark v3 | deadline exhausted | 79,410.651432 secure candidate | 77,393.120160 replayed root | 2.540631% uncertified | 545.521631 |

The incomplete DGX path took 201.867 times the laptop wall time. This is a
system-to-system failed-run comparison, not a pure CPU-versus-GPU kernel
speedup. The useful GPU work was fast; one unresolved PDLP leaf consumed almost
all elapsed time.

## Remaining implementation issues before another run

1. Checkpoint and independently verify every secure incumbent immediately,
   including commitment, dispatch, per-unit dispatch, and bus prices.
2. Canonicalize mathematically identical security inequalities while retaining
   every source pair ID in an equivalence map for exhaustive verification.
3. Give lower-bound leaves bounded warm and cold PDLP attempts. On stagnation,
   roll back the transactional split and try another deterministic split; do
   not spend the global deadline on one leaf.
4. Add a replayable GPU infeasibility certificate before pruning any leaf.
5. Make cross-architecture LODF verification tolerant to harmless FP64
   recomputation differences. The DGX replay passed, but laptop recomputation
   differed bitwise for 162 of 167 pairs; the largest difference was only
   `7.99e-15`, with none above `1e-12`.

No automatic retry is authorized. Machine-readable measurements and immutable
artifact hashes are in [`evidence.json`](evidence.json); pre-run validation is
in [`preflight.json`](preflight.json).
