# ACTIVSg2000 GPU Lagrangian v2 contract

This is one new, nonofficial ACTIVSg2000 DGX Spark experiment with a hard
1,800-second end-to-end boundary. It preserves the immutable v1 result and uses
the new suite ID and tag `activsg2000-gpu-lagrangian-v2` and
`experiment-2000-gpu-lagrangian-v2`. No automatic retry is allowed.

## Corrected fixed-commitment primal path

For a fixed binary commitment, the local generator polytope is

```text
PMIN_g u_g <= Pg_g <= PMAX_g u_g
Pg_g = PMIN_g u_g + sum_s y_gs
0 <= y_gs <= width_gs u_g.
```

Its projection onto `Pg` is exactly `Pg = 0` when the source-online unit is off
and the source interval `[PMIN, PMAX]` when it is on. The v2 feasibility model
therefore retains only free on-unit dispatch columns and the balance, base-flow,
angle, and generated contingency coupling rows. It substitutes fixed dispatch
values and removes only constant-satisfied or box-redundant coupling rows.

cuOpt FP64 PDLP minimizes the shared Phase-I violation on this projected model.
After every feasible solve, CuPy screens the complete valid branch-N-1 set. All
new violating pairs are added deterministically and Phase I is solved again.
Success requires a final exhaustive screen with no violation above `1e-5` p.u.

Before any cost optimization, the dispatch is lifted to exact binary commitment,
the unmodified source PMIN, and the original ten PWL segments. The ordinary raw-
input checker must pass immediately. This verified solution is durable even if
the subsequent exact fixed-commitment production-cost LP fails. The cost LP gets
the lifted primal in its own native scaled column space; only a successful secure
cost LP can supply the reported fixed-commitment bus-price duals.

## Unchanged boundaries

- Single hour, preventive branch N-1, common commitment and dispatch.
- Exact TAMU source PMIN/PMAX and source-derived ten-segment production costs.
- No load shedding, spillage, overload slack, or automatic feasibility repair.
- cuOpt PDLP and CuPy on the Spark; no integer solver and no CPU branch-and-bound.
- No CPU commitment, dispatch, or lower-bound state seeds the GPU algorithm.
- The existing hashed laptop HiGHS result is comparison evidence only.
- One run, console logging enabled, 1,800-second hard limit, no retry.
