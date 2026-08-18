# ACTIVSg2000 GPU Lagrangian v2 result

The single frozen v2 DGX Spark run ended `failed_exception` after 110.193 seconds.
It did not time out, claim model infeasibility, produce a secure integer primal,
or report an optimality gap.

The root relaxation completed three internal constraint-generation rounds. Its
first solve took 10.551 seconds and added 84 logical contingency pairs; the
second took 50.965 seconds and added 20; the third took 37.932 seconds and its
exhaustive screen found no new violation. Each screen covered 17,563,400 valid
contingency sides. The final physical screening violation was
`1.41994860314298e-12` p.u.

The secure root LP objective was `1,128,529.23343229`. The GPU-generated
Lagrangian certificate independently replayed at the conservative lower bound
`1,128,529.22343225`; the root still had 53 fractional commitments.

The first binary candidate contained 333 committed units. During exact dispatch
projection, row `c0603_m0609_lower` had no coefficient on any free dispatch
column and was violated by its fixed dispatch contribution. This correctly
proves only that candidate infeasible: after substitution the row required
`0 <= -0.3799999999934869` MW, a 0.38 MW violation. V2 incorrectly propagated the condition
as `ScopfError`, which stopped the experiment before projected Phase I. V3
narrows it to a candidate rejection and continues the queue. No solver
infeasibility or numerical PDLP failure occurred at this gate.

The immutable raw result SHA-256 is
`5502462740192e66ae05c1ff1cc564c0b4427efeddb5f3a701a04815f09d2861`.
Additional artifact hashes and the exact frozen identity are in
[evidence.json](evidence.json).
