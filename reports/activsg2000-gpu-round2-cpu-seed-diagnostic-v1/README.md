# ACTIVSg2000 round-2 CPU-seeded GPU diagnostic v1

## Outcome

The separately authorized one-shot diagnostic did not enter cuOpt's solve
pipeline. The exact round-2 master and the complete CPU state passed every
pre-solve gate, but cuOpt 26.06 returned native `NoTermination` with zero native
solve time, zero nodes, and no incumbent or bound.

This is neither an infeasibility result nor a failure to discover a commitment.
The full CPU start was independently secure before translation. Native console
output stopped after the environment banner, before the normal model-size,
scaling, or presolve messages.

## Frozen identity and inputs

- Git commit: `70ac5b517e5bfee6a81685430f4ca718ad9f1900`
- Git tag: `diagnostic-2000-gpu-round2-cpu-seed-v1`
- cuOpt: `26.06.00` in the registered digest-pinned NVIDIA image
- Secure CPU result SHA-256:
  `5573425a8e625c0c964b2c61d33ca80666de74a350e9431a96e5ce9b90e02e3f`
- Failed GPU v3 result SHA-256:
  `9b9e7d3b0dfdad0faeb740f639307844920c18eb85c07ac9705f8b84e034366c`
- Raw diagnostic result SHA-256:
  `53f7bb8929eb27a930facbf2a0ede4a340ff01b6ddcc0418bfa7055a598945e7`
- Console log SHA-256:
  `0b29ebb627541dae7242b2b909c949563a3f93cee2bf3523fdcbee1bb9ea6247`

The raw case and contingency hashes remained the registered TAMU ACTIVSg2000
hashes. Raw results remain in ignored `results/` storage; the compact immutable
evidence is tracked here.

## Pre-solve proof

- Canonical model: 9,220 columns, 8,961 rows, 27,120 nonzeros.
- Security rows: the exact 173 stable pair IDs from GPU v3 round 1.
- MIP start: all 9,220 columns, including all 432 commitment variables.
- Largest canonical row residual: `2.420986220386112e-7`.
- Largest canonical variable-bound excess: `5.994036200718256e-8`.
- Largest integrality residual after normalization: `0.0`.
- Independent exhaustive security maximum: `7.875883056840393e-10` p.u.
- Independent verification: passed across all 2,740 valid outages and
  17,563,400 monitored sides.

All values passed the registered `1e-6` model tolerance and `1e-5` security
tolerance. No model limits, PMIN values, or acceptance tolerances were changed.

## Native result

- Native status: `NoTermination`.
- Adapter solve wall time: `0.3938759930024389` seconds.
- Native solve time: `0.0` seconds.
- End-to-end wall time: `1.9244666470040102` seconds.
- Nodes/simplex iterations: `0` / `0`.
- Incumbent/objective/bound: none / none / none.
- Translated dimensions: 9,220 columns, 8,961 canonical rows, 27,120 nonzeros,
  and 8,961 native constraints.

The serialized native MIP gap of `0.0` is not meaningful because no incumbent
or finite bound exists; the requested gap was not certified.

## Post-run attribution

A tiny Spark component test reproduced the immediate status without using any
ACTIVSg data:

- an exactly bound-feasible full MIP start reached the solver;
- adding only a `6e-8` variable-bound excess returned `NoTermination` at zero
  solve time; and
- an equality residual of `2.4e-7` did not cause the immediate return.

The ACTIVSg CPU state has one `5.994e-8` bound excess and additional smaller
floating-point bound excesses. This isolates the v1 error to the MIP-start
handoff: cuOpt requires the supplied start to respect variable bounds exactly,
even though the state is feasible under the registered model residual
tolerance. NVIDIA's 26.06 Python reference describes `setMIPStart` as an initial
primal solution hint, and the
[NVIDIA MIP documentation](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/mip/mip-api.html)
describes starts as feasible solutions.

The approved correction projects only sub-tolerance start-value excesses onto
the existing exact column bounds before calling cuOpt. It does not change the
canonical model, the CPU commitment, PMIN/PMAX, contingency rows, or tolerances.
