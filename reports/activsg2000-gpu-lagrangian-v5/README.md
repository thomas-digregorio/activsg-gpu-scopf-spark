# ACTIVSg2000 GPU Lagrangian v5 result

## Outcome

The single authorized DGX Spark v5 run ended cleanly with status
`deadline_budget_exhausted`. The v4 numerical acceptance failure is fixed: all
three root LP rounds passed the independent canonical FP64 residual gate, and
the exhaustive root contingency screen ended at `2.1282e-12` p.u. No unsafe
vector was screened or accepted.

The scientific success gate did not pass. The GPU search did not find a secure
integer commitment and dispatch, so objective, dispatch, commitment, pricing,
and a certified primal-versus-bound gap are unavailable. This run is not a
successful SCOPF solve and was not retried.

## Numerical correction result

The scale-safe `power_system_equilibrated_safe_v3` reformulation never reduces
the original per-unit row scale. The root residuals were `1.4828e-7`,
`1.8797e-11`, and `1.9145e-12` p.u., all below the registered `1e-6` p.u.
tolerance. The first root solve therefore did not need the one permitted
same-master refinement. Rounds two and three accepted the mapped prior primal
and row duals without a dimension mismatch.

The child acceptance gate also behaved correctly. One intermediate child had
a `1.7654e-6` p.u. residual, so it was not accepted; a bounded continuation
reduced that residual to `2.9104e-13` p.u. before the split committed. Failed
siblings remained uncertain and their transactions rolled back to the parent
certificate. No uncertain child was called infeasible or used to prune the
cover.

One numerical-performance warning remains. Every one of the 124 native PDLP
invocations warned that the input has a large coefficient range. In the root
model, constraint coefficients ranged from about `1e-8` to `6`, despite safe
row scaling and the certified small-coefficient cleanup. This warning did not
invalidate the canonical residual checks or certificate replay, but it is a
credible contributor to slow child convergence. It should not be described as
fully resolved numerical conditioning.

## Root relaxation

| Round | Native PDLP (s) | New pairs | Canonical residual (p.u.) | Exhaustive maximum violation (p.u.) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.041 | 84 | 1.4828e-7 | 1.5669193722 |
| 2 | 47.105 | 20 | 1.8797e-11 | 0.2348830163 |
| 3 | 37.246 | 0 | 1.9145e-12 | 2.1282e-12 |

Each exhaustive screen evaluated 17,563,400 contingency sides. The final root
LP objective was `$1,128,529.233432252`, with 53 fractional commitments and 104
logical security pairs. Its independently replayed conservative root bound was
`$1,128,529.223432194`.

## Primal and disjunctive search

The Phase-I-first initial-candidate path attempted its configured maximum of 12
GPU-generated commitments in 27.226 seconds. Three failed exact constant-row
checks, and nine network-repaired candidates had positive Phase-I optimum
violations from `0.05294` to `0.78799` p.u. This prevented those initial
candidates from consuming 60--90 seconds each in ordinary cost LPs, but none
was secure.

The disjunctive search completed 14 transactions: three committed, eleven
rolled back fail-closed, and a fifteenth remained transactional when the solve
budget closed. The complete frontier contains four regions. Thirty short
Phase-I child prechecks were numerically feasible at zero violation but did not
produce dual bounds strong enough to certify pruning, so ordinary cost LP work
was still required. Ten completed two-context parallel batches consumed
847.192 seconds of batch wall time. This child-LP convergence, rather than
contingency screening or the root residual bug, was the dominant bottleneck.

## Lower-bound certificate

The recorded four-region GPU lower bound is `$1,128,528.596742726`. A separate
laptop verifier reread the raw ACTIVSg2000 case and contingency files,
reconstructed the four leaves, verified the disjunctive cover, and replayed
`$1,128,528.596742735`, a difference of `9.08e-9` dollars.

CPU and GPU cleanup audits classified a small number of values differently at
the `1e-9` dust threshold (maximum count difference 222), while the maximum
replayed FP64 audit-value difference was `6.11e-10`. The certificate and cover
still passed the registered independent replay tests. This threshold-level
audit drift is recorded rather than hidden.

## Laptop comparison

The frozen laptop HiGHS run completed in 1,007.370 seconds (16.790 minutes),
with objective `$1,133,479.385501136`, bound `$1,132,599.352235491`, relative
gap `7.763999e-4`, 327 committed units, and independent secure verification.
The DGX worker used 1,656.017 seconds (27.600 minutes) and did not produce a
secure primal. Using the CPU incumbent only as an external reference, the GPU
bound implies a 0.436778% gap, above the requested 0.1%.

The laptop completed and the DGX experiment did not, so v5 provides no GPU
speedup. This is a system-to-system comparison, not a pure solver-kernel
comparison.

## Evidence identity

The run is frozen at commit
`8afe50f884d7e092dcdfbb11c6f389b447800499`, tag
`experiment-2000-gpu-lagrangian-v5`, config SHA-256
`085e4c919da395b71e2496d5ae6d4bb8ab8c84b63f95898ba2cc498cedb8f298`,
and image SHA-256
`19686764843633e1782c3441508865122bbbe59e19838ea74869aff313a9019b`.
The raw result SHA-256 is
`17962980825ae7920588f8658068d6014c3a44fc61510f28531923d4e32342d9`.
All exact artifact hashes and replay metrics are in
[evidence.json](evidence.json).
