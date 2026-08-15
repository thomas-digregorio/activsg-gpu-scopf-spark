# ACTIVSg2000 DGX Spark 1e-3 result

The single authorized run completed in 7.909
seconds but did **not** produce an accepted SCOPF solution. cuOpt returned native
status `FeasibleFound` for the base restricted master. Its reported gap was
1.255739e-04, below the requested `1e-3`, but the frozen fail-closed
adapter requires native `Optimal` before promoting contingency rows.

The controller still exhaustively screened the incumbent. It found
165 violated contingency pairs,
with a maximum violation of
2.041592 p.u., then stopped.
No security rows were added, independent verification was not reached inside the
run, and fixed-commitment pricing was not run.

| Platform | Status | Objective | Bound | Gap | Committed | Rounds | Wall (s) | Pricing |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Laptop CPU | `optimal_verified` | 1,133,479.385501 | 1,132,599.352235 | 7.763999e-04 | 327 | 3 | 1007.370 | accepted |
| DGX Spark | `incomplete_restricted_master_not_optimal` | 1,118,437.552574 | 1,118,297.105972 | 1.255739e-04 | 325 provisional | 1 | 7.909 | not reached |

## Independent post-hoc check

The raw-input checker was run once on the saved incumbent without another MIP
solve. Base-model residuals pass: the maximum is
1.893e-10 p.u., and the conditional
PMIN/PMAX residual is
2.827e-11 p.u. The
checker fails N-1 security across
17,563,400 sides with a maximum violation of
2.041592 p.u.

Thus the provisional commitment and dispatch are not comparable to the accepted
laptop grid solution. The lower GPU objective reflects a base restricted master,
not a better secure solution. GPU nodal prices do not exist for this run.

## Timing attribution

- Model and factor build: 0.283 s
- cuOpt round 1: 5.541 s
- CuPy exhaustive screen: 1.034 s
- End to end: 7.909 s
- Fresh suite-specific CUDA/CuPy cache; no full-case warmup

NVIDIA's [cuOpt MIP settings documentation][cuopt-mip-settings] describes
`mip_relative_gap` as a termination tolerance and notes that cuOpt MIP
optimality proofs remain under active development. Reclassifying
`FeasibleFound` when its reported gap passes would change the adapter acceptance
policy and requires a new frozen run identity; this run was not retried.

## Evidence files

- `generator-detail.csv`: all 544 exact source PMIN/PMAX rows, the accepted CPU
  solution, and the clearly labeled provisional GPU commitment/dispatch. GPU
  pricing fields are blank.
- `bus-prices.csv`: all 2,000 accepted CPU prices; GPU price fields are blank.
- `round-detail.csv`: CPU and GPU solver/screening attribution.
- `summary.csv` and `comparison.json`: frozen identities, raw hashes, post-hoc
  verification, record audits, and the explicit no-retry status.

The source audit found zero PMIN/PMAX substitutions, conditional limit
violations above `1e-6` MW, source-offline availability violations, or MW/p.u.
conversion mismatches.

[cuopt-mip-settings]: https://docs.nvidia.com/cuopt/user-guide/latest/mip-settings.html
