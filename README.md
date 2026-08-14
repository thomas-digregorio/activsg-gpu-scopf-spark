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
```

`solve` is a bounded nonofficial end-to-end run. `benchmark` is the registered
one-shot run. Do not invoke `benchmark` casually: before starting work it writes
an ignored, durable registry entry, and it refuses an automatic retry or
replacement even after failure.

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
unchanged. Only one v6 optimization run is authorized; there is no automatic
retry.

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
