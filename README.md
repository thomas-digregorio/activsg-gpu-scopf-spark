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

The `activsg2000-gap-sensitivity-v1` suite applies that same bounded, ordered,
one-shot design to ACTIVSg2000. Each gap is limited to 1,800 seconds and a later
gap cannot start after any timeout, failure, verification failure, or missing
fixed-commitment pricing. It uses the exact source-case PMIN values and does not
reuse a solution or contingency-pair list from another gap.

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
