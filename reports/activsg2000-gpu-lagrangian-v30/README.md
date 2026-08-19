# ACTIVSg2000 GPU Lagrangian V30

V30 is preserved as a completed one-shot diagnostic run. It exercised the
all-coupling GPU PDLP multiplier proposal after the exact root-cut refresh and
then completed 110 proof-only disjunctive splits. Every child produced a finite,
monotonically accepted, exactly replayed certificate; no numerical child or
split transaction failed. The run retained a secure primal and an independently
replayed lower bound, but it did not certify the requested 0.1% relative gap
before the verification reserve.

## Frozen identity

- Commit: `c33786a2609e7bfa75dbc64938c067e315059be2`
- Tag: `experiment-2000-gpu-lagrangian-v30`
- Config SHA-256: `541ea60ceada573319c8791764414da52d2bac5a03af95a7a55a72579f9ecd46`
- Container image: `sha256:2d07ef4e315fd07aa547af0d0abf7b8a19bb5beb644d9d3bf744b04940611601`
- CPU commitment, dispatch, objective, and lower-bound data used by the GPU
  algorithm: no

## Result

- Status: `incomplete_refinement_budget_reserve_reached`
- Total wall time: 829.2535873809829 seconds
- Secure incumbent objective: 1142133.305099282
- Independently replayed lower bound: 1130468.1369321526
- Certified relative gap: 0.010213490942824206
- Requested relative gap: 0.001
- Committed generators: 331
- Proof-only splits: 110
- Final frontier regions: 111
- Failed split transactions: 0
- Pruned regions: 0

The secure incumbent passed independent raw-input exhaustive verification over
2,740 valid outages and 17,563,400 monitored sides. Its maximum model residual
was 2.460183168295771e-12 p.u. and its maximum security violation was
4.674901411760857e-06 p.u., below the registered 1e-05 p.u. tolerance.

The complete native MIP-start contract passed, with no native rejection,
barrier warning, or large-coefficient warning. All 220 proof-child proposals
were accepted after exact FP64 replay. Their aggregate wall time was
404.9012530562177 seconds and their median local certificate lift was
74.72601181245409 dollars.

## Full-coupling proposal and next correction

The all-coupling proposal model had 6,935 columns, 4,014 rows, and 18,917,181
retained nonzeros. cuOpt PDLP used its 90-second native limit and returned a
finite time-limit vector, but exact nonsmoothed replay found it 0.1565216768
dollars weaker than the inherited certificate. The monotone gate therefore
retained the inherited bound. The complete stage cost 102.02497748300084
seconds and lifted the certified bound by zero.

V30 therefore closes the numerical-failure question: the outstanding problem
is not infeasibility, MIP-start rejection, or invalid replay. The next revision
replaces this large generic proposal LP with a structure-exploiting GPU
finite-state smooth search whose every iterate is still scored by the exact
nonsmoothed FP64 certificate. It also restores the bounded GPU-only diversified
commitment search that V30 had configured to zero attempts.
