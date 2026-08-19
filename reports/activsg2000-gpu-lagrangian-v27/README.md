# ACTIVSg2000 GPU Lagrangian V27

V27 is preserved as a failed one-shot diagnostic run. It fixed the V26 full
MIP-start numerical failure, but later stopped on an over-strict comparison of
two equal-value GPU and host Lagrangian minimizers.

## Frozen identity

- Commit: `48cf40ebfd6c8ab76a7c3ff2ebf7b9dfaab5d4d1`
- Tag: `experiment-2000-gpu-lagrangian-v27`
- Config SHA-256: `5de52585eae2e0acf3e50297de2f2c21ac4f621ac630ee7c40a859e35b06c4df`
- Platform: DGX Spark, cuOpt 26.6.0, FP64 PDLP/CuPy screening
- CPU commitment, dispatch, objective, and lower-bound data used by the GPU
  algorithm: no

## Result

- Status: `failed_exception`
- Total wall time: 288.82323547499254 seconds
- Preserved secure incumbent objective: 1142133.305099282
- Preserved lower bound: 1130027.3066126746
- Relative gap at failure: 0.010599461930194742
- Committed generators: 331
- Failure: `Proof-only child GPU minimizer failed exact host replay`

The incumbent independently passed all 2,740 valid outages / 17,563,400
monitored sides. Its maximum model residual was
`2.460183168295771e-12` p.u. and its maximum security violation was
`4.674901411760857e-06` p.u., below the registered `1e-05` p.u. security
tolerance.

## Numerical fix proven by V27

The exact GPU fixed-commitment start polish completed in 21.94326707799337
seconds. The reconstructed full start had a maximum native row residual of
`1.1406697808524768e-10`, the complete native MIP-start contract passed, and
cuOpt recorded zero MIP-start rejections and zero coefficient-range, free
variable, or barrier warnings. The sparse full heuristic returned
`FeasibleFound`; its commitment stayed unchanged for 25.61346308400971
seconds after the only observed commitment state.

V27 was not rerun. V28 replaces vector-identity comparison with a fail-closed
host objective, binary, region-mask, and hard-cardinality replay for alternate
GPU argmins. The exact host-replayed lower-bound value and host argmin remain
authoritative.
