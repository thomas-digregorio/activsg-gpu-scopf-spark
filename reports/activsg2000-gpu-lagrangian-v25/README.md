# ACTIVSg2000 GPU Lagrangian v25

This is the compact evidence index for the one-shot DGX Spark v25 run. The
41,818,424-byte raw result remains on the Spark and in the ignored local
results directory; it is intentionally excluded from Git.

## Frozen identity

- Commit: `172bcab3de89f74c7b897f902bf5a771ba61c3a8`
- Tag: `experiment-2000-gpu-lagrangian-v25`
- Config SHA-256: `b3ddb4dd3bd8b05ab1ae4ad8a1186aeeba090f2a0e2951bdefd8c52bdc47d25c`
- Raw result SHA-256: `89e2ef0845afaf38473cd96834ca57c8be29dabbf168ccb1f374f879dce6f3b9`
- Spark raw result: `/home/dgxsparktd/activsg-gpu-scopf-spark/results/experiments/activsg2000-gpu-lagrangian-v25-dgx-spark.json`
- Local raw result: `C:\Users\thoma\Documents\activsg-gpu-scopf-spark\results\experiments\activsg2000-gpu-lagrangian-v25-dgx-spark.json`

## Corrections made before the run

V25 adds an eight-lane, FP64, GPU-resident nonsmooth Adam search for each
hard-cardinality child certificate. Every iteration solves the binary/PWL
generator subproblem and all disjoint hard-cardinality groups exactly on the
GPU. Stable row-wise `argsort` replaces the unsupported rank-three CuPy
`lexsort` call and gives generator position as the deterministic tie-break.
Only an exact evaluated multiplier can become certificate authority; invalid
or nonfinite lane updates recover to the last valid exact multiplier. Final
host FP64 replay remains mandatory.

Incomplete cuOpt PDLP restart states remain ineligible for resubmission; only
complete, audited states may be reused, with raw-dual fallback otherwise.
This fully GPU-resident lower-bound experiment used neither a MIP solver nor a
MIP start, so the earlier size-mismatched MIP-start path was not exercised.
No CPU commitment, dispatch, solution objective, or bound was supplied to the
GPU algorithm.

The controller timing preflight was also corrected. It now requires 50
seconds before launching a split: 20 seconds for the two bounded Phase-I
children, 12 seconds for their GPU dual searches, and a 15-second transaction
margin. This prevents a child transaction from starting without a declared
budget for the new optimizer.

Before freezing the revision, the Spark candidate image passed exact
GPU/host replay on the tiny hard-cardinality fixture. The direct bound and
Adam replay differences were exactly zero, the Polyak replay difference was
`1.1368683772161603e-13`, and there were no nonfinite evaluations or updates.

## Outcome

The one authorized run stopped cleanly with
`incomplete_refinement_budget_reserve_reached`. End-to-end wall time was
`851.3616478240001` seconds. The secure objective was `1142133.305099282`,
the independently replayed lower bound was `1130040.747115001`, and the
certified relative gap was `0.01058769403736957` (1.0587694%). The requested
`0.001` (0.1%) gap was **not** certified. At the returned incumbent, the bound
needed to reach `1140991.1717941826`; the proof was short by
`10950.424679181539` dollars.

The returned commitment has 331 online generators. The raw result contains
all 544 source generator records with source status, exact PMIN/PMAX,
commitment, MW and p.u. dispatch, and PWL segment dispatch. It also contains
2,000 fixed-commitment bus prices from an optimal GPU PDLP pricing solve.

Independent raw-input verification passed exact conditional source
PMIN/PMAX, integrality, objective reconstruction, DC physics, base limits,
and all 17,563,400 monitored sides for 2,740 valid branch outages. Maximum
model residual was `2.460183168295771e-12` p.u.; maximum contingency
violation was `4.674901411760857e-6` p.u., below the registered `1e-5` p.u.
security tolerance.

The 23-leaf final lower-bound frontier was independently rebuilt from the raw
inputs. It replayed 32 feasibility cuts, 34 cover cuts, and two analytic
capacity cuts. Recorded and replayed global bounds agreed exactly, as did
every region certificate.

## Numerical audit

The numerical failure modes targeted by this iteration did not recur. Across
112 native cuOpt invocations, the worker log contains 103 `Optimal` exits and
nine controlled `Time` exits. It contains zero numerical, NaN, infeasible,
error, warning, factorization, large-coefficient-range, free-variable,
MIP-start-rejection, or initial-solution-rejection messages.

Every native matrix reported coefficients within `[2e-6, 1]`. Sixty-four
invocations used an absolute primal tolerance of `1e-8`; 48 Phase-I
invocations used `1e-10`. All eight Adam lanes on every final frontier leaf
recorded zero nonfinite evaluations and zero nonfinite projected updates.

## Remaining bottleneck

The new optimizer was valid and fast enough per child, but it did not change
the minimum proof path. It improved 11 of the 23 final frontier leaves, with a
maximum local lift of `2131.659953114111` dollars. The minimum leaf followed a
depth-22 cardinality path and improved by only `2.3283064365386963e-10`
dollars, so it continued to pin the global lower bound.

The controller completed 22 split transactions, 45 region solves, and 44
Phase-I child prechecks with no rollback or numerical prune. Initial primal
construction was also expensive: it recorded 52 candidate entries and 46
repairs before refinement. The largest measured blocks were `233.1825`
seconds in relaxation PDLP, `89.2013` seconds in child Phase I, `88.6952`
seconds in fixed-commitment PDLP, `67.1293` seconds in independent frontier
replay, `55.8605` seconds in exact-minimizer separation, and `39.0563`
seconds in GPU Lagrangian evaluation. Controller/model-build/checkpoint
overhead not represented by those non-overlapping timers was about
`271.0748` seconds.

This means the outstanding problem is not numerical feasibility. The current
exact-type/laminar cardinality disjunction does not remove or strengthen the
weak all-at-least path quickly enough. A next iteration should change the
commitment-level relaxation or disjunction strength, not loosen PMIN,
tolerances, or security verification.

## Prior boundaries

V25 was `21.6420560959959` seconds (2.48%) faster than v24, but its global
bound improved by only `0.0000172315631061792` dollars and its gap was
effectively unchanged. The registered laptop run took
`1007.3702709000063` seconds and certified a `0.0007763998859633739` gap.
V25 used `156.008623076006` fewer wall seconds, but this is **not** a
successful GPU speedup comparison: the laptop passed the requested gap and
v25 did not.
