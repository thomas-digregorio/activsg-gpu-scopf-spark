# ACTIVSg2000 GPU Lagrangian v32 evidence

V32 was launched exactly once on the DGX Spark and is preserved as a failed
experiment. It is not a successful lower-bound certificate run and it did not
reach final exhaustive raw-input verification.

## Frozen identity

- Commit: `e59e49988bd92712893ec801df4604c6bfd951ee`
- Tag: `experiment-2000-gpu-lagrangian-v32`
- Config SHA-256: `4f0b9d818563df64195607cf16ade084f185d3691d7b43ee5b548325e9227db8`
- Image ID: `sha256:f02b8c0118f26ccd52cd5cee45e13c4f4959e2ba23aa52dd8f1eb6f16f4655b1`

## Result

- Status: `failed_exception`
- Total wall time: `453.6286579290172` seconds
- Preserved secure objective: `$1,142,133.305099282`
- Preserved in-memory lower bound: `$1,130,026.5017574436`
- Uncertified relative gap at failure: `0.010600166624845857`
- Commitment count: `331`
- Unique additional primal attempts: `45`
- Primal-beam wall time: `241.86151393302134` seconds
- Primal-beam objective improvement: `$0`
- Analytic-cover bound lift: `$1,496.94815580873`
- Completed proof batches: `0`

The failure occurred while preparing the first block-diagonal proof pair:

```text
Hard-cardinality search retained a below-threshold finite_column_bound magnitude
```

The exact epigraph column normalization reduced the largest finite proposal
bound to one, but an asymmetric small bound was divided below the registered
`1e-4` proposal floor after the earlier cleanup pass. The fail-closed check
stopped the run before cuOpt received that proof model. This is a conditioning
pipeline defect in the non-authoritative proposal LP. It is not evidence of
SCOPF infeasibility and it did not alter exact PMIN, the original feasible set,
or the exact replay authority.

V32 must not be rerun unchanged. A replacement version must clean tiny bounds
after the exact diagonal coordinate transformation and must reproduce this
failure state in a Spark preflight before another full launch.

