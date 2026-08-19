# ACTIVSg2000 GPU Lagrangian V31

V31 is preserved as one completed one-shot diagnostic run. It removed the
known zero-lift full-coupling proposal from V30, restored the GPU-only
best-first commitment beam, and reduced each proof-child proposal limit from
2.0 seconds to 0.5 seconds. The run retained a secure primal and an
independently replayed lower bound, but it did not certify the requested 0.1%
relative gap before the verification reserve.

## Frozen identity

- Commit: `fc9aaf09a06b29d847dcfb3b67474e6a86b6fc20`
- Tag: `experiment-2000-gpu-lagrangian-v31`
- Config SHA-256: `d7a0c5df2141506627d4208f3186d3f3b0a025be013649c6ab8a1c0e66fa1e6b`
- Container image: `sha256:ada1690b158e3e18176b70c39b3f95c44579d97fd88e8766a0d22f862e5e2601`
- CPU commitment, dispatch, objective, and lower-bound data used by the GPU
  algorithm: no

## Result

- Status: `incomplete_refinement_budget_reserve_reached`
- Total wall time: 876.4273981389997 seconds
- Secure incumbent objective: 1142133.305099282
- Independently replayed lower bound: 1130506.5379270415
- Certified relative gap: 0.010179868777427641
- Requested relative gap: 0.001
- Committed generators: 331
- Proof-only splits: 206
- Final frontier regions: 207
- Failed split transactions: 0
- Pruned regions: 0

The secure incumbent passed independent raw-input exhaustive verification over
2,740 valid outages and 17,563,400 monitored sides. Its maximum model residual
was 2.460183168295771e-12 p.u. and its maximum security violation was
4.674901411760857e-06 p.u., below the registered 1e-05 p.u. tolerance. The
frontier replay matched the recorded bound exactly.

## Measured diagnosis

The restored commitment beam used 180.4580240569776 seconds, attempted 34
additional unique commitments, and found no incumbent improvement. This is a
search-effectiveness result rather than a feasibility or numerical-failure
result.

The proof controller completed 412 child proposals in 322.1663592689729
seconds. Their median wall time was 0.7706451944977744 seconds and their median
local certificate lift was 69.15746648365166 dollars. All child transactions
were finite and replayable: 385 proposal vectors improved the exact replayed
certificate and 27 were rejected by the monotonicity gate. There were no failed
children.

The remaining numerical problem is isolated to the non-authoritative proposal
LPs. cuOpt reported 305 large-coefficient-range advisories; 408 of 412 child
proposal solves reached their registered 0.5-second limit. The proposal models
retained finite column bounds from approximately 1e-05 to 47 even after row
conditioning. Exact nonsmoothed replay prevented these warnings from changing
or invalidating a certificate, but the poorly scaled proposal coordinates
wasted solver iterations.

The next revision therefore normalizes every proposal column as well as every
row, preserves exact reconstruction and replay, and diversifies same-type unit
placement so harmless root-PDLP tie changes do not collapse the primal search
onto one commitment family. It will not retry V31 unchanged.
