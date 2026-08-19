# ACTIVSg2000 GPU Lagrangian V28

V28 is preserved as a failed one-shot diagnostic run. It passed the V27
GPU/host alternate-argmin replay gate and completed the first proof-only child,
then failed while serializing that child's nested NumPy evidence arrays.

## Frozen identity

- Commit: `cfe04f2fda2d310d81b8c7b9fed39232d3cde4e0`
- Tag: `experiment-2000-gpu-lagrangian-v28`
- Config SHA-256: `70a3a589e97a0a7628d9a2a9b1f33d862c44cb2cd3b7f8252038b1af8d89be4b`
- CPU commitment, dispatch, objective, and lower-bound data used by the GPU
  algorithm: no

## Result

- Status: `failed_exception`
- Total wall time: 306.21677517000353 seconds
- Preserved secure incumbent objective: 1142133.305099282
- Preserved lower bound: 1130027.3066126746
- Relative gap at failure: 0.010599461930194742
- Committed generators: 331
- Failure: `Object of type ndarray is not JSON serializable`
- First failing nested field: `best_commitment_cut_dual`

The traceback occurred in `persist_region_evidence` after the first proof-only
child returned, demonstrating that V28 passed the alternate-argmin binary,
region, hard-cardinality, and exact host-objective replay introduced after V27.
The numerical certificate was not rejected; only its diagnostic evidence could
not be checkpointed.

The secure incumbent again passed independent exhaustive N-1 verification, the
exact full-start polish was accepted, and the complete native MIP-start
contract passed. V28 was not rerun. V29 recursively converts NumPy arrays and
scalars throughout the complete serialized region record without changing any
primal, lower-bound, PMIN/PMAX, tolerance, or branching value.
