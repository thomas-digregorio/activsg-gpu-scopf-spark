# ACTIVSg2000 GPU LP-certificate v12 failed attempt

The one frozen v12 attempt stopped after its first continuous restricted-master
solve. It did not time out and it did not report infeasibility. cuOpt explicitly
translated zero integer columns, selected PDLP Stable3 in FP64, and returned
`Optimal` after 22,800 iterations and 1.398 seconds:

- primal objective: `1,118,296.700341088`
- dual objective candidate: `1,118,296.699947574`
- absolute/relative duality gap in the native log: `3.94e-4 / 1.76e-10`
- native primal residual: `3.0493e-7`
- native dual residual: `2.9777e-5`
- branch-and-bound markers: none

The experiment correctly withheld the bound and stopped before contingency
screening because its new independent checker failed. The failure was in that
checker: cuOpt 26.06's low-level reduced-cost array contained zeros for
bound-active variables, while the checker treated that array as
`c - A^T y`. That produced an impossible reconstructed dual objective of
`1,601,502.546138` and a false stationarity error. A separate tiny analytical
LP demonstrated the same API behavior without running another ACTIVSg model.

Version 0.20.1 derives reduced costs directly from the returned row duals and
the immutable native matrix. It also checks PDLP residuals using cuOpt's
documented absolute-plus-relative scaling. V12 is preserved as a failed attempt
and cannot be overwritten; any approved replacement uses v13 paths.

## Frozen evidence hashes

- result JSON: `fe465489b2c551fcf677f55f1126e87a0f6aefd64624761549469692bbc80c2f`
- one-shot registry: `1d27265d86c23eccdb9c1d353607e9dd0689b94247ff13cc723d7617beea724f`
- worker/native console: `17bc2cd811f58ab1228c878d35de042d4c552373dad7852b95e7056d716142b1`
- launcher console: `e7d99b20481c4f842440cf98cf2158a88b561d5737112534a4d7c238075bd8d0`
- frozen commit/tag: `fcd9b59ce23d891b08b478a112fb4d71100702d4` /
  `experiment-2000-gpu-lp-certificate-v12`
