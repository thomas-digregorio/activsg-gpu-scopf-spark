# ACTIVSg500 GPU Lagrangian v2 result

The one authorized post-fix v2 run completed its controller boundary in
545.664419411 seconds with status `deadline_budget_exhausted`. It did not
produce an accepted objective, lower bound, relative gap, commitment, dispatch,
or prices, so it did not meet the requested `1e-3` global certificate.

This is not an SCOPF infeasibility result. It is a primal-candidate scheduling
failure: the corrected root relaxation solved quickly, but the controller gave
almost all remaining solver time to one rounded commitment whose fixed LP
showed strong numerical evidence of infeasibility.

## What v2 fixed

- The real root matrix had 37,062 nonzeros and minimum nonzero native
  coefficient approximately `1.4316e-6`, rather than v1's `4e-18` numerical
  dust.
- Root PDLP solved the 1,811-row base relaxation optimally in 0.667 native
  seconds, then solved the 1,978-row restricted master with 167 dynamically
  added security rows in 0.321 native seconds.
- The second root solution reached the exhaustive zero-new-violation screen;
  its rounded native-log objective was 77,393.1302.
- Every infeasible fixed-commitment iterate was rejected before contingency
  screening. Zero invalid security rows were added.
- Rounds 2 through 5 report that both native primal and native dual warm starts
  were submitted successfully.
- Failure evidence contains the true elapsed time and every completed repair
  round instead of v1's stale checkpoint.

The root solve demonstrates that the coefficient-cleanup fix addressed the v1
PDLP stagnation. However, v2 did not persist the completed root Lagrangian
certificate into the top-level checkpoint before starting primal repair.
Therefore 77,393.1302 is reported only as a rounded continuous-relaxation
diagnostic from the native log, not as the requested independently replayed
Lagrangian lower-bound certificate.

## Actual bottleneck

The first repair fixed the commitment derived from root-LP rounding. It was not
the known secure CPU commitment (their SHA-256 identities differ). Five PDLP
slices consumed 539.810513 native seconds:

| Slice | Native seconds | Canonical residual (p.u.) | Warm start | Screen |
|---:|---:|---:|---|---|
| 1 | 120.005759 | 0.296609398 | none | skipped |
| 2 | 120.006104 | 0.296608815 | primal + dual | skipped |
| 3 | 120.005353 | 0.406939649 | primal + dual | skipped |
| 4 | 120.002261 | 0.309633850 | primal + dual | skipped |
| 5 | 59.791035 | 0.296842555 | primal + dual | skipped |

The native dual objective diverged from `2.73e18` after slice 1 to `3.64e18`
after slice 5 while primal infeasibility remained material. That is strong
evidence that this fixed commitment is infeasible, but v2 did not obtain a
formal infeasibility certificate. The all-online fallback was selected next
but could not start because the controller correctly retained 45 seconds for
verification and 10 seconds for serialization.

## CPU comparison

| System | Status | Accepted objective | Accepted bound | Gap | Wall seconds |
|---|---|---:|---:|---:|---:|
| Laptop HiGHS | optimal and independently verified | 79,410.651432 | 79,410.651432 | 0 | 2.702380 |
| DGX Spark v2 | deadline budget exhausted | none | none | none | 545.664419 |

The failed DGX path used 201.920x the laptop wall time. This is a
system-to-system failed-run comparison, not a GPU speedup measurement. The GPU
root LPs themselves took only 0.988 native seconds; nearly all time was spent
on the single fixed-commitment repair.

## Required next controller change

A later experiment should preserve the root region/certificate immediately,
give each primal candidate its own small budget and stagnation/infeasibility
gate, then move deterministically through root rounding, the Lagrangian
minimizing commitment, and all-online (or another GPU-generated feasible)
candidate. This changes candidate scheduling, not PMIN, PWL costs, security
tolerances, the global gap definition, or the no-CPU-branch-and-bound policy.
It requires a separately registered run; v2 must not be overwritten or retried.

Machine-readable values and immutable artifact hashes are in
[`evidence.json`](evidence.json).
