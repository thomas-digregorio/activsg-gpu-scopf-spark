# ACTIVSg2000 GPU Lagrangian V29

V29 is preserved as a completed one-shot diagnostic run. It fixed V28's
nested NumPy evidence-serialization failure and completed 134 proof-only
disjunctive splits without a failed transaction. The run retained a secure
primal and an independently replayable lower bound, but it did not certify the
requested 0.1% relative gap before the verification reserve.

## Frozen identity

- Commit: `82d0238300647d0d90079d3c8ae77589342ec446`
- Tag: `experiment-2000-gpu-lagrangian-v29`
- Config SHA-256: `bae8ce4375b207199e97bfce39cb9f8593abba99fcad36dc71a6bcb4a9006653`
- CPU commitment, dispatch, objective, and lower-bound data used by the GPU
  algorithm: no

## Result

- Status: `incomplete_refinement_budget_reserve_reached`
- Total wall time: 828.0409738820163 seconds
- Secure incumbent objective: 1142133.305099282
- Independently replayed lower bound: 1130484.3060344174
- Certified relative gap: 0.010199334011936532
- Requested relative gap: 0.001
- Committed generators: 331
- Proof-only splits: 134
- Final frontier regions: 135
- Failed split transactions: 0
- Pruned regions: 0

The secure incumbent passed independent raw-input exhaustive verification over
2,740 valid outages and 17,563,400 monitored sides. Its maximum model residual
was 2.460183168295771e-12 p.u. and its maximum security violation was
4.674901411760857e-06 p.u., below the registered 1e-05 p.u. tolerance.

The V27 exact full-start polish and native MIP-start contract passed again.
V29 also passed the V28 GPU/host alternate-argmin replay gate and recursively
serialized every nested proof-child array and scalar. No CPU branch-and-bound
or integer lower-bound solver was used.

## Bottleneck attribution

The 268 proof-only child proposals consumed 503.97402626101393 seconds in
aggregate. Every proposal improved its exact replayed child certificate, but
the weakest live-region bound advanced too slowly: the frontier grew by one
region per split and no region could be pruned. V30 therefore targets bound
strength rather than relaxing any model, PMIN/PMAX, security, or numerical
acceptance tolerance.
