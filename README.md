# ACTIVSg500 GPU DC-SCOPF on DGX Spark

Fresh, auditable prototype for a one-hour preventive branch-N-1 DC
security-constrained commitment and dispatch MILP on the synthetic TAMU
ACTIVSg500 system.

The registered comparison is exactly one end-to-end laptop CPU run versus
exactly one end-to-end DGX Spark run. The laptop uses HiGHS with NumPy/SciPy;
the Spark uses NVIDIA cuOpt with CuPy. No custom CUDA kernels are present.

## Scope guardrails

- ACTIVSg500 only. Every explicit reference to another ACTIVSg size is rejected.
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

Obtain these two files from the TAMU ACTIVSg500 distribution and place them under
`data/raw/matpower-8.1/`:

| File | SHA-256 |
|---|---|
| `case_ACTIVSg500.m` | `8ca6d54ea5179eeb03fe29d7b645618e7a86338c172247e81687476660f6dcbe` |
| `contab_ACTIVSg500.m` | `f6b2e7e38fd1cf5e09e877cf04233b4d0487d6d0e99903070d519eade12b76a9` |

The parser reads MATPOWER text without executing MATLAB code and refuses a hash
mismatch. Stable identities such as `gen-row-0001` and `branch-row-0001` refer
to immutable one-based source rows.

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

Tests use only tiny fixtures. They do not solve ACTIVSg500.

## Versioned CLI

All commands consume JSON configuration and produce JSON evidence:

```powershell
activsg-scopf ingest --config configs\activsg500.json --output work\ingest.json
activsg-scopf solve --config configs\activsg500.json --platform laptop_cpu --output work\nonofficial-solve.json
activsg-scopf verify --config configs\activsg500.json --solution work\nonofficial-solve.json --output work\verification.json
activsg-scopf benchmark --config configs\activsg500.json --platform laptop_cpu --output results\laptop-cpu-official.json
```

`solve` is a bounded nonofficial end-to-end run. `benchmark` is the registered
one-shot run. Do not invoke `benchmark` casually: before starting work it writes
an ignored, durable registry entry, and it refuses an automatic retry or
replacement even after failure.

## Five-minute official protocol

The benchmark controller requires a clean tracked worktree and requires `HEAD`
to equal the configured `benchmark-v1` tag. It measures worker launch through
the first complete result serialization, including raw input loading, factor and
model construction, every solve/screen round, and independent exhaustive
verification. The solver receives only the time left after a 45-second
verification reserve and a 5-second serialization reserve. A parent watchdog
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
bash scripts/spark-build.sh
bash scripts/spark-benchmark.sh
```

The repository is mounted read-only in the container, with only ignored
`results/` mounted read-write. The laptop result must first be copied to
`results/laptop-cpu-official.json`. Details are in
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

