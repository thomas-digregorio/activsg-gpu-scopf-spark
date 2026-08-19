# ACTIVSg GPU DC-SCOPF on DGX Spark

Fresh, auditable prototype for a one-hour preventive branch-N-1 DC
security-constrained commitment and dispatch MILP on the synthetic TAMU
ACTIVSg500, ACTIVSg2000, and ACTIVSg10k systems. The bounded MIP-gap studies
apply the same mathematical contract to ACTIVSg500 and ACTIVSg2000; the earlier
ACTIVSg10k benchmark and experiment evidence remains preserved.

Each registered case comparison is exactly one end-to-end laptop CPU run versus
exactly one end-to-end DGX Spark run. The laptop uses HiGHS with NumPy/SciPy;
the Spark uses NVIDIA cuOpt with CuPy. No custom CUDA kernels are present.

## Scope guardrails

- Only ACTIVSg500, ACTIVSg2000, and ACTIVSg10k are registered. Every other
  ACTIVSg size is rejected.
- Exact source-case `PMIN` and `PMAX` are conditional on commitment. Source-offline
  generators are unavailable.
- The interval is exactly one hour. Ramping, minimum up/down times, startup
  trajectories, reserves, and every inter-period constraint are absent.
- Source polynomial production-cost curves become ten equal-MW chord segments
  over exact `[PMIN, PMAX]`. They are not submitted market offers.
- Version 1 includes source-listed, in-service, non-islanding branch outages.
  Generator outages and corrective redispatch are deferred.
- There is no load shedding, generation spillage, overload slack, or feasibility
  repair.
- Raw cases, factor caches, solver artifacts, environments, and detailed results
  are ignored by Git.
- The repository and every input, output, cache, temporary, and environment path
  used by the application must be outside OneDrive.

The complete equations and conventions are in
[`docs/model-contract.md`](docs/model-contract.md).

## Immutable inputs

Obtain the selected pair from the TAMU distribution and place it under
`data/raw/matpower-8.1/`:

| File | SHA-256 |
|---|---|
| `case_ACTIVSg500.m` | `8ca6d54ea5179eeb03fe29d7b645618e7a86338c172247e81687476660f6dcbe` |
| `contab_ACTIVSg500.m` | `f6b2e7e38fd1cf5e09e877cf04233b4d0487d6d0e99903070d519eade12b76a9` |
| `case_ACTIVSg2000.m` | `8d00618de8fd10bf35a599f59d2deebfecd0d86e28fcff73219ad7c4ebab860b` |
| `contab_ACTIVSg2000.m` | `198b39f0381925a4ddacbe2148973cb1d93ddfe220303829cf87b16d45190bba` |
| `case_ACTIVSg10k.m` | `ead10b25fecc4dcc02f88bacdfb3526fe8b8985b81f7e539c95abddb32575590` |
| `contab_ACTIVSg10k.m` | `7e1681a960b0a2a99d824766e0e94cc291fa36a7ec33b6dee24bf12ac67ddef7` |

The parser reads MATPOWER text without executing MATLAB code and refuses a hash
mismatch. Stable identities such as `gen-row-0001` and `branch-row-0001` refer
to immutable one-based source rows. The tracked
[`ACTIVSg2000 source manifest`](data/source-manifests/activsg2000.json) records
the original source-online PMIN/PMAX totals and case dimensions. The tracked
[`ACTIVSg10k source manifest`](data/source-manifests/activsg10k.json) records
1,937 source-online generators and their aggregate exact PMIN of 85,764.93 MW;
the detailed ingest/result manifest retains every generator row and PMIN value.

## Laptop setup

Use the approved local path
`C:\Users\thoma\Documents\activsg-gpu-scopf-spark`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-cpu.lock -r requirements-dev.lock
.\.venv\Scripts\python -m pip install --no-deps -e .
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
```

Tests use only tiny fixtures. They do not solve any full ACTIVSg case.

## Versioned CLI

All commands consume JSON configuration and produce JSON evidence:

```powershell
activsg-scopf ingest --config configs\activsg500.json --output work\ingest.json
activsg-scopf solve --config configs\activsg500.json --platform laptop_cpu --output work\nonofficial-solve.json
activsg-scopf verify --config configs\activsg500.json --solution work\nonofficial-solve.json --output work\verification.json
activsg-scopf benchmark --config configs\activsg500.json --platform laptop_cpu --output results\laptop-cpu-official.json
activsg-scopf ingest --config configs\activsg10k.json --output work\activsg10k-ingest.json
activsg-scopf benchmark --config configs\activsg10k.json --platform laptop_cpu --output results\activsg10k-laptop-cpu-official.json
activsg-scopf benchmark --config configs\activsg10k-v2.json --platform laptop_cpu --output results\activsg10k-v2-laptop-cpu-official.json
activsg-scopf benchmark --config configs\activsg10k-v3.json --platform laptop_cpu --output results\activsg10k-v3-laptop-cpu-official.json
activsg-scopf gpu-lagrangian-experiment --config configs\activsg500-gpu-lagrangian-v5.json --output results\experiments\activsg500-gpu-lagrangian-v5-dgx-spark.json
```

The authorized v6 speed experiment moves the replayable GPU Phase-I solve in
front of each ordinary disjunctive child cost LP. A positive certified
Phase-I bound prunes immediately; a zero or uncertain result proceeds to the
cost LP and can supply a primal-only warm start. See the
[v6 controller contract](docs/gpu-lagrangian-500-v6.md). Its one-shot commands
are:

```bash
bash scripts/spark-build-500-gpu-lagrangian-v6.sh
bash scripts/spark-run-500-gpu-lagrangian-v6.sh
```

The frozen v6 run failed before its root LP because a validator confused
generator source-row keys with canonical commitment-column indices. That run
is preserved in the [v6 result report](reports/activsg500-gpu-lagrangian-v6/README.md).
The corrected, not-yet-run v7 identity is documented in the
[v7 bugfix contract](docs/gpu-lagrangian-500-v7.md).

`solve` is a bounded nonofficial end-to-end run. `benchmark` is the registered
one-shot run. Do not invoke `benchmark` casually: before starting work it writes
an ignored, durable registry entry, and it refuses an automatic retry or
replacement even after failure.

## ACTIVSg500 GPU Lagrangian/disjunctive experiment

The registered `activsg500-gpu-lagrangian-v5` experiment uses no integer solver and
no CPU branch-and-bound. cuOpt PDLP generates continuous relaxation and
fixed-commitment dispatch solutions; CuPy performs exhaustive contingency
screening and a persistent FP64 projected-supergradient Lagrangian loop over
the exact binary generator subproblems. Any missing contingency rows weaken,
but cannot invalidate, its lower bound. If the root certificate is short of
`1e-3`, disjoint on/off regions refine it and the minimum leaf bound certifies
their exhaustive union. Exact duplicate post-cleanup contingency rows share one
cuOpt representative while retaining every pair identity. A stalled child gets
one cold restart and then a GPU Phase-I attempt; it can be pruned only with a
serialized box-dual infeasibility certificate that independently replays.

The authorized v5 full-model execution is one registered DGX Spark run
with a 600-second end-to-end deadline. Development uses tiny fixtures only. Its
laptop comparison is the already-published, hashed HiGHS `1e-3` result; no new
laptop full-model solve is part of this experiment. See the complete
[v5 correction and acceptance contract](docs/gpu-lagrangian-500-v5.md). The
[v4 result report](reports/activsg500-gpu-lagrangian-v4/README.md) preserves the
verified incumbent, valid two-leaf bound evidence, and row-identity verifier
failure without relabeling that run as successful. The
[v3 result report](reports/activsg500-gpu-lagrangian-v3/README.md) preserves the
bounded-candidate run and its durable root/incumbent evidence. The
[v2 failure report](reports/activsg500-gpu-lagrangian-v2/README.md) preserves
the exhausted candidate-repair attempt and its valid root relaxation evidence;
the [v1 failure report](reports/activsg500-gpu-lagrangian-v1/README.md) preserves
the first attempt without calling its primal-infeasible PDLP iterate a model
infeasibility.

## ACTIVSg2000 GPU Lagrangian v2 experiment

The ACTIVSg2000 v2 bugfix keeps the v1 mathematical model, exact source PMIN,
runtime limits, GPU lower-bound method, and no-branch-and-bound policy. For a
binary candidate, its feasibility-first PDLP now uses the exact projection onto
free committed-generator dispatch: fixed commitment columns, off-unit dispatch,
and local PWL segment bookkeeping are omitted. Every retained dispatch keeps its
source PMIN/PMAX interval, and every active network or security coupling row is
copied exactly after fixed dispatch is substituted. A zero-violation projected
dispatch is lifted into the unchanged ten-segment model and independently
verified before it warm-starts the ordinary cost LP. If cost polishing fails,
the verified secure dispatch remains the incumbent, with pricing explicitly
unavailable unless the cost LP succeeds. See the
[v2 contract](docs/gpu-lagrangian-2000-v2.md).

```bash
bash scripts/spark-build-2000-gpu-lagrangian-v2.sh
bash scripts/spark-run-2000-gpu-lagrangian-v2.sh
```

The frozen v2 run preserved a valid, independently replayed root lower bound
but stopped before its first projected Phase-I solve: a contingency row became
a violated constant after fixed dispatch substitution, and that candidate-level
infeasibility was incorrectly raised as an experiment-level exception. The
[v2 result report](reports/activsg2000-gpu-lagrangian-v2/README.md) preserves
the failure. The v3 correction rejects only that binary candidate and continues
the deterministic candidate queue; it does not call the full model infeasible.
See the [v3 correction contract](docs/gpu-lagrangian-2000-v3.md).

The frozen `activsg2000-gpu-lagrangian-v19` run showed that an outward RHS
relaxation alone does not bound the raw violation hidden by an already-added
security row. Its root PDLP solves were numerically clean, but the independent
screen correctly rejected a 0.000145078 p.u. residual. The v20 correction keeps
the v19 model, exact source PMIN/PMAX, 0.1% target, and 900-second boundary. For
security rows only, coefficient cleanup is now limited by the full dispatch-box
error envelope: outward RHS relaxation plus the maximum raw-activity increase.
The registered 5e-6 p.u. envelope plus the 1e-6 p.u. canonical residual limit
stays strictly below the 1e-5 p.u. exhaustive-screen tolerance. The one-shot
commands are:

```bash
bash scripts/spark-build-2000-gpu-lagrangian-v20.sh
bash scripts/spark-run-2000-gpu-lagrangian-v20.sh
```

The immutable v20 run confirmed that correction: its three-round root solve
ended with an exhaustive maximum N-1 violation of `3.85572234620213e-6` p.u.,
below the registered `1e-5` p.u. tolerance. It did not produce an incumbent,
however, because two older hand-written version gates stopped at v19. That made
the v20 candidate policy `null` and sent the first commitment directly into
four long ordinary cost-PDLP attempts instead of the bounded projected Phase-I
pipeline.

The v21 controller correction leaves the v20 numerical row fix, mathematical
model, exact source PMIN/PMAX, tolerances, and 900-second deadline unchanged.
ACTIVSg2000 versions now have one ordered registration sequence and named
monotone capability sets. The candidate controller fails closed at startup if
v21 is missing any required feasibility-first, cost-projection, child-Phase-I,
diversification, best-first, or GPU-heuristic route. The one-shot commands are:

```bash
bash scripts/spark-build-2000-gpu-lagrangian-v21.sh
bash scripts/spark-run-2000-gpu-lagrangian-v21.sh
```

The immutable v21 run removed the candidate-controller bypass and all observed
numerical solver failures. It produced a secure, independently verified primal
and replayable lower bound, but stopped at a `0.007230802237839392` relative
gap. Three intermediate raw-input certificate rebuilds consumed about 166
seconds even though the same frontier was checked again at final acceptance.

The v22 lower-bound throughput revision keeps the v21 mathematical model,
source PMIN/PMAX values, tolerances, security-row envelope, and 900-second
deadline. It performs raw-input certificate replay only at the mandatory root
and final gates, runs one bounded cost-PDLP dual search after all root cuts are
installed, and uses Phase I followed by a 30-second dual-only cost solve for
each disjunctive child. The secure Phase-I primal remains authoritative; every
proposed multiplier is rescored by the exact FP64 Lagrangian evaluator. The
one-shot commands are:

```bash
bash scripts/spark-build-2000-gpu-lagrangian-v22.sh
bash scripts/spark-run-2000-gpu-lagrangian-v22.sh
```

The v23 numerical/refinement revision preserves the same SCOPF model,
source-derived PWL costs, exact source PMIN/PMAX values, tolerances, and
900-second deadline. Its native transformation equilibrates coefficients
without shrinking a row merely because its finite limit is large, and maps
cuOpt's absolute primal tolerance back to the canonical per-unit residual
gate. For disjoint commitment-cardinality branches, the GPU now minimizes the
binary generator subproblem exactly by ordered on-values at the inherited
replayable multipliers. This removes the v22 child cost-PDLP searches and the
zero-lift strengthened-root cost search. Phase I still supplies and screens
each child's feasible continuous primal; the final raw-input replay and
exhaustive N-1 verifier remain mandatory. The one-shot commands are:

```bash
bash scripts/spark-build-2000-gpu-lagrangian-v23.sh
bash scripts/spark-run-2000-gpu-lagrangian-v23.sh
```

## MIP-gap sensitivity experiments

The separately registered `activsg10k-gap-sensitivity-v2` experiment runs the
unchanged model once at each requested HiGHS relative MIP gap: `1e-3`, `1e-4`,
`1e-5`, `1e-6`, and `1e-7`. These five runs have no wall-clock deadline. Each
starts independently from the base restricted master; gap levels do not seed
one another. The persistent HiGHS session still passes the previous round's
commitment as a partial MIP start after new contingency rows are added.

The separately frozen `activsg500-gap-sensitivity-v1` suite uses the same five
gap levels on ACTIVSg500. Every level has a hard 1,800-second end-to-end limit,
including raw loading, all dynamic constraint-generation rounds, independent
verification, fixed-commitment pricing, and worker result serialization. The
controller refuses to start a later gap unless every earlier gap completed as
`optimal_verified` with accepted pricing. It reserves 120 seconds for post-MIP
work and 15 seconds for serialization; the parent watchdog remains the hard
30-minute boundary.

The separately frozen `activsg500-gpu-gap-sensitivity-v1` suite repeats those
five ACTIVSg500 gap levels on the DGX Spark, once each and in the same strict
order. Its MIP uses cuOpt, exhaustive contingency screening uses CuPy, and each
rebuilt restricted master receives the prior round's integer commitment as a
partial MIP start. Fixed-commitment nodal pricing uses HiGHS 1.15.1 inside the
same Spark container because the cuOpt adapter does not expose the required
nodal row duals. Every level retains the same hard 1,800-second end-to-end
boundary, and the GPU suite has its own registry and result namespace.

That Spark suite is complete. All five one-shot runs finished
`optimal_verified` with accepted fixed-commitment pricing in 3.671 to 4.917
seconds. Every requested gap returned the same 50-unit commitment, objective,
dispatch, and nodal prices, with a zero reported gap. The tracked
[Spark gap report](reports/activsg500-gpu-gap-sensitivity-v1/README.md) contains
full source-row generator and all-bus price tables plus stage and round timing.
The paired [laptop-versus-Spark report](reports/activsg500-cpu-vs-spark-gap-v1/README.md)
shows identical grid decisions within numerical precision; on this small case,
the laptop remained faster end-to-end.

The `activsg2000-gap-sensitivity-v1` suite applies that same bounded, ordered,
one-shot design to ACTIVSg2000. Each gap is limited to 1,800 seconds and a later
gap cannot start after any timeout, failure, verification failure, or missing
fixed-commitment pricing. It uses the exact source-case PMIN values and does not
reuse a solution or contingency-pair list from another gap.

The separately registered `activsg2000-gpu-gap-sensitivity-v1` suite authorizes
exactly one DGX Spark run at `1e-3`. It uses the same immutable source hashes,
exact PMIN/PMAX, model, tolerances, and 1,800-second boundary as the accepted
laptop `1e-3` run. No GPU `1e-4` or later-gap configuration is registered.

That single Spark run is now closed without a retry. cuOpt returned native
`FeasibleFound` for the base restricted master in 7.909 seconds with a reported
gap of `1.255739e-4`, but the frozen adapter required native `Optimal` before
adding contingency rows. The exhaustive screen found 165 violated pairs with a
maximum violation of 2.041592 p.u.; therefore no security rows, accepted
verification, or GPU pricing followed. A post-hoc raw-input check confirmed the
base-model and exact conditional PMIN/PMAX residuals but failed exhaustive N-1
security. The tracked [ACTIVSg2000 Spark report](reports/activsg2000-gpu-1e-3-v1/README.md)
preserves the provisional 544-generator record, blank GPU price fields, timing,
and failure evidence.

The explicitly authorized `activsg2000-gpu-gap-sensitivity-v2` replacement
changes only the restricted-master acceptance policy and frozen identity. A
cuOpt `FeasibleFound`, `Optimal`, or `TimeLimit` incumbent can proceed only when
the objective and finite dual bound independently reproduce a relative gap at
or below `1e-3`, cuOpt's reported gap also passes, and all recorded native
constraint, integrality, and variable-bound residuals are at most `1e-6`.
Every accepted master is exhaustively screened; all newly violated pairs are
added before a re-solve, and the new dispatch is screened again. Success still
requires both a certified requested gap and a final exhaustive screen with zero
violations above `1e-5` p.u., followed by the independent raw-input checker.

V2 ran once and is closed as `failed_exception`: round 1 solved below the
requested gap, but NumPy boolean values in the new certificate could not be
serialized into the next checkpoint. The failure occurred before screening,
so it produced no accepted commitment, dispatch, or prices. The tracked
[v2 failure report](reports/activsg2000-gpu-1e-3-v2-failure/README.md) preserves
the diagnostic solve evidence and raw-artifact hashes. The user then authorized
v3 with only the scalar-serialization correction and a regression test; the
model and iterative security acceptance rule are unchanged.

V3 is also closed without a retry. Round 1 correctly certified its `1e-3` gap,
screened all 17,563,400 sides, and added all 173 violated pairs. The rebuilt
round-2 master received 432 partial integer start values, but cuOpt returned
native `Infeasible` after 1,659.012 seconds with no incumbent or finite bound.
No second screen, final verification, or pricing was possible, so the run ended
`incomplete_no_incumbent` after 1,667.116 seconds. This is not evidence that the
mathematical case is infeasible: the accepted laptop solution uses the same
model and passes the complete contingency set, providing a feasible witness for
the 173-row subset. The tracked [v3 report](reports/activsg2000-gpu-1e-3-v3/README.md)
contains every exact-PMIN generator row, the clearly labeled provisional
round-1 dispatch, blank GPU prices, and full timing/status evidence.

The separately authorized follow-up is a one-shot fixed-master diagnostic. It
rebuilds those exact 173 round-2 security rows, independently proves the known
secure CPU commitment and full dispatch state feasible, then gives all 9,220
canonical values to cuOpt with native console logging enabled. It performs no
constraint-generation rounds. See the
[seeded round-2 diagnostic contract](docs/activsg2000-seeded-round2-diagnostic.md).
V1 returned native `NoTermination` before presolve because the accepted CPU
floating-point state contained a `5.994e-8` variable-bound excess. The approved
v2 correction projects only numerical excesses onto the unchanged exact bounds.
The v2 native console then accepted that full point as feasible, but its
unscaled root LP suffered repeated basis-factorization repairs, an unremovable
`1.09e+50` perturbation, and a barrier numerical error before printing
`MIP Infeasible`. The [numerical attribution](reports/activsg2000-round2-numerical-attribution-v1/README.md)
shows why that label is contradicted by the known feasible witness. The
run retained that witness and passed exhaustive verification, but returned no
finite bound or gap, so it correctly failed the requested gap certificate. The
authorized v3 correction is an invertible per-unit diagonal reformulation only
inside the cuOpt adapter; it removes no row and changes no PMIN, limit,
objective, security tolerance, or canonical solution meaning.
The scaled fixed-master diagnostic proved the correction: it certified a
`0.0008423` gap in 20.04 native solve seconds. Its improved dispatch then
exposed 14 new contingency pairs, so it correctly failed the exhaustive gate.
The v4 experiment restores dynamic add-resolve-screen rounds with the same
scaling and prior-commitment partial MIP starts.
That one v4 run is now closed. It added 349 pairs in round 1 and 17 in round 2;
round 3's exhaustive screen found zero violations, and a post-run independent
checker passed the saved incumbent. Round 3 nevertheless ended at gap
`0.0022012`, above the requested `0.001`, so the run remains incomplete and
pricing is withheld. See the
[v4 report](reports/activsg2000-gpu-1e-3-v4/README.md). No retry was performed.

The subsequently authorized v5 run retained the exact v4 mathematical model,
per-unit scaling, dynamic contingency generation, and later-round partial
integer starts, but used no CPU initialization. It reserved 900 seconds for raw
loading and cumulative cuOpt rounds inside a 1,035-second end-to-end boundary.
Round 1 started cold, round 2 received the prior GPU commitment and added 14
new pairs, and round 3 again received the prior GPU commitment. Round 3 reached
its time limit at objective `1,132,939.788370`, bound `1,130,527.130461`, and
gap `0.0021296`. Its exhaustive screen found zero violations, and a post-run
independent checker passed all 17,563,400 sides. The result is therefore secure
but gap-uncertified, remains incomplete, and has no accepted pricing. See the
[v5 report](reports/activsg2000-gpu-1e-3-v5/README.md). No v5 retry was
performed.

The separately authorized v6 experiment changes only the cuOpt solver policy
and frozen identity relative to v5. It explicitly selects the PDLP method in
Stable3 mode with FP64 arithmetic and enables cuOpt's batched PDLP strong- and
reliability-branching controls with reliability factor 1. The adapter reads
every parameter back before solving and records both the requested and observed
native values in each round. The exact-PMIN model, per-unit reformulation,
`1e-3` target, cold first round, prior-GPU integer starts in later rounds,
dynamic add-resolve-screen loop, and cumulative 900-second solve allowance are
unchanged. The native log confirmed cooperative batch PDLP and dual-simplex
strong branching in all three rounds. V6 added 173 then 18 pairs; round 3's
exhaustive screen found zero violations and the independent checker passed all
17,563,400 sides. It nevertheless ended at objective `1,132,912.294626`, bound
`1,130,615.375711`, and gap `0.0020274`, above the requested `0.001`. The run
is secure but gap-uncertified, remains incomplete, and has no accepted pricing.
See the [v6 report](reports/activsg2000-gpu-1e-3-v6/README.md). No v6 retry was
performed.

The separately authorized v7 experiment preserved the complete v6 model,
PDLP profile, initialization, screening loop, tolerances, and raw inputs while
doubling the cumulative cuOpt allowance from 900 to 1,800 seconds. It added 173
then 14 contingency pairs. The adapter submitted the prior GPU integer
commitment before rounds 2 and 3, but retrospective native-log inspection found
that cuOpt rejected both starts after internal model expansion (`10911` versus
`9220` columns). Round 3 stopped on cuOpt's time limit and found no new
violations in an exhaustive
17,563,400-side screen. The post-run independent checker passed every side,
with maximum security violation `5.982e-12` p.u. The final objective was
`1,133,078.528383`, the lower bound was `1,130,841.046502`, and the gap was
`0.0019747`, above the requested `0.001`. The incumbent is therefore secure
but gap-uncertified, the run remains incomplete, and pricing is withheld.
Recorded restricted-master time was `1,802.319` seconds, including a
`2.319`-second cuOpt return overrun, while end-to-end wall time remained inside
the separate 1,935-second guard at `1,805.749` seconds. See the
[v7 report](reports/activsg2000-gpu-1e-3-v7/README.md). No v7 retry was
performed.

The separately authorized v8 replacement attempted to correct that start
defect without using a CPU initialization. Its single run is now closed as a
failed attempt. Round 1 completed, added 173 security rows, and round 2 then
reported a native start-vector mismatch (`11214` assignment values versus
`9220` supplied values). The operator stopped the run after approximately 342
seconds. There is no v8 final result, independent verification, or pricing.
See the [v8 failed-attempt record](reports/activsg2000-gpu-1e-3-v8-failure/README.md).

The authorized v9 replacement fixes both native start translation and start
feasibility handling. For a later round, it first fixes the prior GPU
commitment in a bounded HiGHS continuous LP. If that commitment is extendable,
the resulting feasible full solution is submitted to cuOpt after exact native
free-variable splitting and fixed/unused-column elimination. If it is not
extendable, the reason is logged and that same restricted master is solved
cold. V9 has no external CPU initialization and keeps the v8 raw inputs,
exact-PMIN model, PDLP profile, tolerances, dynamic screening, 1,800-second
solver allowance, and 1,935-second outer guard. Exactly one v9 optimization
run is authorized.

That single v9 run is now closed without a retry. Both later-round prior GPU
commitments failed the bounded fixed-commitment feasibility gate, so rounds 2
and 3 correctly solved cold; no start reached cuOpt and native log audits show
zero start rejections or barrier warnings. Round 1 added 173 contingency pairs,
round 2 added 13, and round 3's mandatory exhaustive screen exposed two more
after the solve allowance was exhausted. The final solved-master objective was
`1,132,724.812824`, the bound was `1,130,678.342896`, and the gap was
`0.0018067`, above the requested `0.001`. Independent raw-input verification
passed exact PMIN/PMAX and all base-model checks but confirmed a maximum N-1
violation of `0.054596464` p.u. for the two unresolved pairs. Pricing is
withheld. The [v9 report](reports/activsg2000-gpu-1e-3-v9/README.md) preserves
the frozen run, all generator rows, blank price identities, native evidence,
and the separate diagnostic proving that a genuinely feasible 9,220-column
canonical start translates to 11,214 native columns and is accepted by cuOpt.

Post-run inspection also tightened HiGHS incumbent classification: version
0.18.1 requires explicit feasible-primal status before a finite vector can be
exposed or reused. The frozen v9 controller had already rejected the affected
`Unknown` precheck vector by direct residual checks, so this telemetry fix does
not alter the recorded solve or its cold-start decision.

The separately authorized commitment-trace diagnostic keeps the v9 mathematical model,
source hashes, exact PMIN/PMAX, gap and security tolerances, PDLP policy,
start-feasibility gate, and no-CPU-initialization rule. It changes the
cumulative cuOpt allowance to 600 seconds and enables cuOpt's supported
incumbent callback. Every distinct within-solve commitment state is recorded
with elapsed time, objective/bound, on-unit count, fingerprint, and exact
on/off deltas; every completed restricted-master solve also records a complete
544-source-row commitment and round-to-round changes. This is a diagnostic run,
not a timing comparison, because callback instrumentation adds host work.

The frozen v10 launch failed before case loading because its suite was omitted
from the one-shot experiment registry; no optimization ran. Version 0.19.1
registers both the preserved v10 identity and the replacement v11 identity,
and tests every gap-experiment JSON through the same identity check used by the
CLI. The one authorized computational replacement therefore uses v11 paths and
an immutable v11 tag rather than overwriting v10's failed-launch evidence.

The single v11 computational run used 600.041 seconds across three cuOpt
rounds and finished in 601.954 seconds end to end. Its final incumbent is
independently exhaustive-screened N-1 secure, but its 0.002211857 MIP gap did
not certify the requested 0.001 threshold. Unit commitment did not stabilize:
round endpoints changed by 46 and then 39 source rows, and round 3 recorded 30
distinct callback commitments with its final change only 7.287 seconds before
return. The [v11 commitment report](reports/activsg2000-gpu-commitment-trace-v11/README.md)
preserves all 544 round endpoints, 64 distinct incumbent states, and 2,123
exact within-solve unit flips. Its component probe also confirms that the
corrected cuOpt adapter translates a five-column canonical MIP start into the
proper four-column native vector with no assignment-size rejection.

The separately authorized v12 experiment asks a different question: can the
DGX Spark certify the `1e-3` gap using the continuous relaxation rather than a
CPU branch-and-bound proof? It relaxes all 432 source-online commitment columns
to `[0, 1]`, explicitly selects FP64 PDLP, and exhaustively screens each
fractional restricted-master solution before adding every newly violated N-1
row. The adapter translates zero integer variables and independently checks the
returned row duals, reduced costs, primal/dual objectives, stationarity, and
residuals. The resulting bound is a numerical LP lower-bound certificate, not
an exact rational certificate and not an integer solution. It is compared with
the independently verified v11 feasible objective only after the final
fractional screen and a fresh raw-input verification pass. The one-shot command
has a hard 600-second worker boundary:

```console
activsg-scopf lp-certificate \
  --config configs/activsg2000-gpu-lp-certificate-v12.json \
  --output results/experiments/activsg2000-gpu-lp-certificate-v12-dgx-spark.json
```

The frozen v12 launch stopped after its first continuous solve because the new
independent checker incorrectly trusted cuOpt 26.06's low-level reduced-cost
array. PDLP itself returned `Optimal` in 1.398 seconds with zero integer
columns, primal objective `1,118,296.700341`, and dual objective
`1,118,296.699948`; no contingency screen was reached and no bound was
accepted. Version 0.20.1 preserves that failed attempt and corrects the checker
by deriving the bound multipliers from `c - A^T y`. It also evaluates the
reported primal, dual, and gap residuals against the same absolute-plus-scaled
thresholds used by cuOpt. Any replacement run must use the distinct v13
configuration, tag, registry, and output paths.

The single approved v13 replacement also stopped after its first `Optimal`
PDLP solve, in `3.024` seconds end to end. It translated zero integer columns
and used no branch-and-bound, but the checker found one syntactically unbounded
angle column and also attempted to reconstruct a post-presolve cuOpt residual
threshold from the pre-presolve RHS. No contingency screen ran and no bound was
accepted. Version 0.20.2 preserves that failure and prepares the distinct v14
identity. V14 derives finite redundant bounds for every angle from the existing
DC flow equations, finite `RATE_A` limits, and fixed reference angle; uses the
model's `1e-6` primal tolerance; and enables cuOpt's per-constraint PDLP
residual mode. The derivation changes no physical feasible point, PMIN value,
cost, outage, or security tolerance.

The single v14 run is closed as incomplete. Rounds 1 and 2 passed the corrected
certificate checker, exhaustively screened all 17,563,400 sides, and added 84
then 20 security pairs. Round 3 reached its 120-second per-round PDLP cap and
returned `TimeLimit`, so it had no accepted vector to screen. The run stopped
after 208.669 seconds despite the separate 600-second outer boundary, performed
no final verification, and accepted no exhaustive LP bound. The tracked
[v14 report](reports/activsg2000-gpu-lp-certificate-v14/README.md) preserves
the timing, conservative dual reconstruction, and raw-artifact hashes.

The single v15 replacement completed in `129.385` seconds and resolved the
deadline, warm-start, time-limit-vector, and dual-coordinate defects. Its third
round solved the exhaustive LP to optimality and the final screen found zero
violations, but the verified lower bound of `$1,128,529.120570` implies a
`0.003988135` gap to the known secure integer incumbent. That is above the
requested `1e-3`, so plain continuous PDLP cannot provide the requested proof
for this formulation. The tracked
[v15 report](reports/activsg2000-gpu-lp-certificate-v15/README.md) contains the
full round trace, independent post-run verification, and artifact hashes.

That ACTIVSg2000 campaign is now closed. The `1e-3` run completed
`optimal_verified` with accepted fixed-commitment pricing; the `1e-4` run hit
its restricted-master solver budget above the requested gap and its provisional
screen still found 18 violated pairs. Accordingly, `1e-5`, `1e-6`, and `1e-7`
were not started. The tracked [ACTIVSg2000 gap report](reports/activsg2000-gap-sensitivity-v1/README.md)
contains all accepted generator/pricing records and the labeled incomplete
incumbent evidence.

Use `gap-experiment`, not `solve` or `benchmark`. The command writes a durable
one-shot registry before worker launch and refuses a second run for that gap:

```powershell
activsg-scopf gap-experiment --config configs\activsg10k-gap-1e-3.json --output results\experiments\activsg10k-gap-v2-1e-3-laptop.json
activsg-scopf gap-experiment --config configs\activsg500-gap-1e-3.json --output results\experiments\activsg500-gap-v1-1e-3-laptop.json
activsg-scopf gap-experiment --config configs\activsg2000-gap-1e-3.json --output results\experiments\activsg2000-gap-v1-1e-3-laptop.json
```

On the Spark checkout, use the guarded scripts after checking out the frozen
GPU experiment tag:

```bash
bash scripts/spark-build-500-gpu-gap.sh
bash scripts/spark-gap-500-gpu.sh 1e-3
# Single authorized ACTIVSg2000 GPU run
bash scripts/spark-build-2000-gpu-1e-3.sh
bash scripts/spark-gap-2000-gpu-1e-3.sh
# Explicitly authorized replacement after the v1 status-gate failure
bash scripts/spark-build-2000-gpu-1e-3-v2.sh
bash scripts/spark-gap-2000-gpu-1e-3-v2.sh
# Explicitly authorized replacement after the v2 serialization failure
bash scripts/spark-build-2000-gpu-1e-3-v3.sh
bash scripts/spark-gap-2000-gpu-1e-3-v3.sh
# Separately authorized exact round-2 solve with the secure CPU full start
bash scripts/spark-build-2000-round2-cpu-seed-diagnostic-v1.sh
bash scripts/spark-run-2000-round2-cpu-seed-diagnostic-v1.sh
# Approved correction for v1's sub-tolerance MIP-start bound excess
bash scripts/spark-build-2000-round2-cpu-seed-diagnostic-v2.sh
bash scripts/spark-run-2000-round2-cpu-seed-diagnostic-v2.sh
# Approved numerical correction after v2's root-LP factorization failure
bash scripts/spark-build-2000-round2-cpu-seed-diagnostic-v3.sh
bash scripts/spark-run-2000-round2-cpu-seed-diagnostic-v3.sh
# One fresh scaled dynamic rerun after the numerical fix
bash scripts/spark-build-2000-gpu-1e-3-v4.sh
bash scripts/spark-gap-2000-gpu-1e-3-v4.sh

bash scripts/spark-build-2000-gpu-1e-3-v5.sh
bash scripts/spark-gap-2000-gpu-1e-3-v5.sh
# One authorized PDLP-policy comparison against v5
bash scripts/spark-build-2000-gpu-1e-3-v6.sh
bash scripts/spark-gap-2000-gpu-1e-3-v6.sh
# One authorized 30-minute PDLP run; otherwise identical to v6
bash scripts/spark-build-2000-gpu-1e-3-v7.sh
bash scripts/spark-gap-2000-gpu-1e-3-v7.sh
# One authorized corrected-start 30-minute run
bash scripts/spark-build-2000-gpu-1e-3-v8.sh
bash scripts/spark-gap-2000-gpu-1e-3-v8.sh
# One authorized feasibility-checked-start replacement
bash scripts/spark-build-2000-gpu-1e-3-v9.sh
bash scripts/spark-gap-2000-gpu-1e-3-v9.sh
# Frozen v10 pre-solve failure (preserved; do not rerun)
# One authorized 10-minute computational replacement
bash scripts/spark-build-2000-gpu-commitment-trace-v11.sh
bash scripts/spark-gap-2000-gpu-commitment-trace-v11.sh
# Preserved v12 and v13 LP-certificate failures; do not rerun either identity
# V14 is closed as incomplete; do not rerun this identity
bash scripts/spark-build-2000-gpu-lp-certificate-v14.sh
bash scripts/spark-run-2000-gpu-lp-certificate-v14.sh
bash scripts/spark-build-2000-gpu-lp-certificate-v15.sh
bash scripts/spark-run-2000-gpu-lp-certificate-v15.sh
```

Continue with `1e-4` through `1e-7` only after the prior level finishes
`optimal_verified` with accepted pricing. Each label can be launched only once.

The v1 `1e-3` attempt is preserved as failed evidence. HiGHS returned an
internal error while checking its round-2 partial MIP start. The explicitly
authorized v2 replacement still attempts every prior commitment; only that
specific internal error causes the same restricted master to be rebuilt and
solved cold. It is not a general removal of MIP starts.

After a MIP solution is independently verified, pricing fixes its commitment,
relaxes integrality, reoptimizes dispatch as an N-1-secure continuous LP, and
uses the nodal-balance duals as prices. Dispatch and source PMIN/PMAX are stored
in MW and p.u. Prices are stored in $/MWh and in $/p.u.-hour, equal to the
$/MWh value times the case base MVA. This price solve is not another MIP run and
its dispatch is retained separately from the MIP incumbent. See
[`docs/gap-sensitivity-experiment.md`](docs/gap-sensitivity-experiment.md).

## Five-minute official protocol

The benchmark controller requires a clean tracked worktree and requires `HEAD`
to equal the tag registered by the selected configuration (`benchmark-v1` for
the preserved 500 run and `benchmark-10k-v1` for the first 10k run).
For the explicitly authorized second 10k laptop run, `activsg10k-v2` uses
`benchmark-10k-v2`, a persistent incremental HiGHS session, and a 45-second
verification reserve. It does not replace the failed v1 evidence or change the
model, source hashes, tolerances, exact PMIN, or ten-segment costs.

That v2 run is now closed: it reached the 300-second parent watchdog during
round 2 and did not reach independent verification. Do not invoke the v2
benchmark command again. See
[`reports/activsg10k-v2-status.md`](reports/activsg10k-v2-status.md) for the
preserved evidence and post-run deadline correction.

The explicitly authorized `activsg10k-v3` CPU run keeps the same unseeded
dynamic constraint-generation model. It adds one-second HiGHS MIP progress,
native solver messages, and durable phase events under `results/diagnostics/`
so a watchdog termination still leaves internal attribution evidence. It does
not use a preloaded or learned security-pair set.

That v3 run is now closed. It finished inside 300 seconds, but round 2 reached
HiGHS `TimeLimit` at a `5.57e-4` gap and its provisional exhaustive screen found
8 additional pairs. See
[`reports/activsg10k-v3-profile.md`](reports/activsg10k-v3-profile.md) for the
internal timing attribution.

The subsequently authorized DGX Spark v3 diagnostic is also closed. It ran
once as a nonofficial `solve` because the laptop result had not passed the
official gate. It stopped after 253.750 seconds when round 2 exposed 6 more
pairs and no solver budget remained for round 3 before the verification
reserve. See
[`reports/activsg10k-v3-spark-diagnostic.md`](reports/activsg10k-v3-spark-diagnostic.md)
for the evidence and bounded system-to-system comparison. The laptop partial
commitment MIP start remains implemented. That frozen v3 Spark evidence used the
older adapter that rebuilt each round without a start; the new ACTIVSg500 GPU
gap suite rebuilds each round while carrying the prior integer commitment.

The controller measures worker launch through
the first complete result serialization, including raw input loading, factor and
model construction, every solve/screen round, and independent exhaustive
verification. The 10k solver receives only the time left after a 75-second
verification reserve in v1 or a 45-second reserve in v2, plus a 5-second
serialization reserve. A parent watchdog
terminates the worker at 300 seconds and serializes its last checkpoint.

Run the laptop first. A Spark run is rejected unless its `--laptop-result` is an
official, under-300-second, `optimal_verified` result from the exact same commit,
tag, and configuration hash. See
[`docs/benchmark-protocol.md`](docs/benchmark-protocol.md).

## DGX Spark setup

The approved path is `/home/dgxsparktd/activsg-gpu-scopf-spark`. The Spark image
is pinned by digest and contains cuOpt 26.06.00. Environment build and image pull
are setup, outside the measured interval:

```bash
cd /home/dgxsparktd/activsg-gpu-scopf-spark
# Preserved ACTIVSg500 workflow
bash scripts/spark-build.sh
bash scripts/spark-benchmark.sh
# ACTIVSg10k workflow
bash scripts/spark-build-10k.sh
bash scripts/spark-benchmark-10k.sh
# ACTIVSg10k v2 workflow (only after a passing matching laptop result)
bash scripts/spark-build-10k-v2.sh
bash scripts/spark-benchmark-10k-v2.sh
# Explicitly authorized nonofficial v3 diagnostic after the laptop gate failed
bash scripts/spark-build-10k-v3-diagnostic.sh
bash scripts/spark-solve-10k-v3-diagnostic.sh
# Frozen ACTIVSg500 GPU gap experiment
bash scripts/spark-build-500-gpu-gap.sh
bash scripts/spark-gap-500-gpu.sh 1e-3
```

The repository is mounted read-only in the container, with only ignored
`results/` mounted read-write. The matching laptop result must first be copied
to the result filename expected by the selected Spark script. Details are in
[`docs/environment-contract.md`](docs/environment-contract.md).

## Acceptance

Success requires both official runs to:

1. complete in at most 300 seconds;
2. report an optimal restricted master in the final constraint-generation round;
3. have no remaining contingency violation above `1e-5` p.u.; and
4. pass an independent raw-input checker at `1e-6` p.u. model tolerance.

The comparison reports objective, bound/gap, committed count, rounds, added
pairs, residuals, exhaustive violation, peak memory, and end-to-end wall time.
It is a system-to-system comparison, not a pure GPU speedup claim.

## Licensing

No license is granted for the original code at this time. See
[`NOTICE.md`](NOTICE.md) for source-data and dependency attribution.
