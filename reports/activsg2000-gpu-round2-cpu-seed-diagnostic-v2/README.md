# ACTIVSg2000 round-2 CPU-seeded GPU diagnostic v2

## Outcome

The one-shot v2 diagnostic is closed as
`diagnostic_incumbent_returned_but_acceptance_gate_failed`. It is not an
infeasibility result. cuOpt accepted the independently verified full CPU state
and returned that state as a feasible incumbent, but its unscaled root LP
failed numerically. The native console printed `MIP Infeasible`, while the
Python API later exposed status `Optimal` with no finite bound and no finite
MIP gap. The fail-closed adapter therefore reported `OptimalGapMismatch` and
did not certify the requested `1e-3` gap.

## Frozen identity and hashes

- Commit: `3f070fafa020431744fcb376f9fbd890eb666de5`
- Tag: `diagnostic-2000-gpu-round2-cpu-seed-v2`
- Config SHA-256:
  `fe2a49f3b25af0fceca4567bde7b5d0e90b76d27bd06e02d4cd4a6e3157c6da2`
- Raw result SHA-256:
  `96f4068f4327b859d11cecfc8d3bf63c33a9a71779685d24a532a74379e60929`
- Console SHA-256:
  `7a0a638257988bf41715556344a231b7dcac7e721c721ce1aa56fc712b18cabe`
- Registry SHA-256:
  `9f8e21ded51d04d531bfdd56df0b531b32140f4b8a0df07f9520f8d4f4543dc6`

Raw result storage remains ignored under `results/`; this directory retains
the compact result facts and complete console stream.

## Pre-solve and returned-state proof

- Fixed model: 9,220 columns, 8,961 rows, 27,120 nonzeros, and the exact 173
  security rows from the prior round-1 screen.
- Full start: all 9,220 columns, including all 432 commitments.
- Bound correction: 79 sub-tolerance values projected to unchanged exact
  bounds; largest change `5.9940362e-8` MW; no commitment changed.
- Pre-solve maximum row violation: `2.4209862e-7` in canonical row units.
- Returned objective: `1,133,479.3855011382` dollars.
- Returned maximum canonical row violation: `2.4209862e-7`.
- Exhaustive GPU screen: zero violations above tolerance across 17,563,400
  sides; maximum raw screen excess `1.1368684e-15` p.u.
- Independent raw-input verification: passed; maximum model residual
  `2.4209726e-9` p.u. and maximum N-1 violation `1.2468718e-9` p.u.

Thus the incumbent is a valid secure grid state. It is not accepted as an
optimal result because no finite lower bound or certified gap exists.

## Native numerical failure

cuOpt warned about the coefficient range, then its root dual-simplex solve
reported repeated basis factorization failures and repairs. One repair found
three deficient columns. The objective/perturbation path diverged to values as
large as `4.1768e26` and `1.09e50`; perturbation removal failed, followed by a
barrier numerical error and the text `Root relaxation returned: INFEASIBLE` /
`MIP Infeasible`.

The same log also says:

```text
Adding initial solution success! feas 1 objective 1133479.385501
```

At termination, cuOpt retained that incumbent but supplied `-inf` as the
solution bound and `inf` as the relative gap. The Python statistics exposed
native status `Optimal`; the adapter rejected that internally inconsistent
combination. Native solve time was `1,657.746821` seconds and end-to-end wall
time was `1,661.545536` seconds.

The detailed trigger attribution and mathematically equivalent v3 correction
are documented in
[`../activsg2000-round2-numerical-attribution-v1/README.md`](../activsg2000-round2-numerical-attribution-v1/README.md).
