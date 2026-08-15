# ACTIVSg2000 GPU v8 failed-attempt record

V8 is a preserved failed attempt, not a completed gap experiment. The one-shot
registry was created at `2026-08-15T01:25:31.099112Z` for frozen commit
`53b734ca75ad4f07425b313b9f5c32d50c10b881` and tag
`experiment-2000-gpu-gap-v8`.

Round 1 completed in 5.502 seconds of adapter wall time. cuOpt returned an
incumbent objective of `1,118,442.384611587`, lower bound
`1,118,297.105971293`, and relative gap `0.00012989371852447946`. The
exhaustive 17,563,400-side screen took 1.030 seconds, found 173 new violated
pairs, and reported a maximum violation of `2.0459100279296125` p.u. Those 173
rows were appended in canonical pair-ID order.

Round 2 then submitted the prior 432 commitment values. The live native log
reported:

```text
Error cannot add the provided initial solution! Assignment size 11214 initial solution size 9220
```

The operator stopped the container after approximately 342 seconds rather than
spend the remaining registered budget on a solve that had already rejected its
start. The registry therefore remains `started`, and the checkpoint remains at
`active_stage=restricted_master_solve`, round 2. There is no final result,
independent verification, or pricing result.

The raw temporary cuOpt log was inside a `docker run --rm` container and was
lost when that container was stopped. The quoted line was observed live but
cannot be assigned a retained-file hash. This is an explicit evidence
limitation; the checkpoint, event stream, and registry hashes are retained in
[`evidence-summary.json`](evidence-summary.json).

Follow-up diagnostics isolated two independent facts:

1. cuOpt expands 1,994 non-reference free-angle columns before applying an
   expression-API MIP start and removes five fixed or unused columns. The
   adapter must make both transformations explicit for native start-vector
   identity.
2. After dimension identity was achieved (`11214 == 11214`), the same start
   was still rejected. A fixed-commitment HiGHS LP proved in 0.010 seconds that
   the saved round-1 commitment is infeasible for the exact 173-row round-2
   master. Its prior dispatch violated `c2440_m2453_lower` by 204.591 MW, and
   no redispatch with that commitment can repair the master.

The v9 policy therefore retains prior GPU commitments only when a bounded
fixed-commitment LP proves they are extendable. It submits the resulting
feasible full completion; otherwise it logs the infeasibility and solves the
same restricted master cold. V9 does not use an external CPU initialization.
