# One-shot benchmark protocol

## Frozen identity

An official run requires all of the following to agree:

- clean Git `HEAD` and the tag registered by the selected case configuration;
- complete Git commit hash;
- SHA-256 of the selected case configuration;
- both immutable raw-input hashes; and
- the registered platform profile.

The ACTIVSg500 tag and evidence remain immutable. ACTIVSg10k uses its own tag,
output names, checkpoints, and benchmark-ID-specific registry. The detailed
result and durable run registry live in ignored `results/`. A
sanitized comparison may be committed only after both one-shot runs finish.

The failed `activsg10k-v1` laptop evidence also remains immutable. The
explicitly authorized second laptop run is registered as `activsg10k-v2` with
tag `benchmark-10k-v2`; it has a separate registry, checkpoint, and output.

## Boundary

The measured boundary starts immediately before launching the isolated worker
that loads the raw case. It ends when that worker has serialized a complete
result. It includes Python startup, environment validation, raw parsing and
hashing, sparse factorization, LODF validation, model construction, every MILP
solve, every exhaustive screen, checkpoint writes, independent verification,
and result serialization.

Git checkout, dependency installation, Docker image pull/build, and copying the
two registered raw files are setup and are excluded.

## Deadline behavior

The global limit is 300 seconds. Solver calls receive the current remaining
budget minus the configured verification reserve (45 seconds for ACTIVSg500,
75 seconds for ACTIVSg10k) and 5 seconds for serialization. Each
adapter also sets its native time limit. The parent process independently kills
the worker at the global boundary if it has not returned.

ACTIVSg10k v2 restores the verification reserve to 45 seconds using the v1
measurements (8.84 seconds for model/factor construction and 2.03 seconds per
complete screen). Its HiGHS adapter loads the canonical model once, appends only
new security rows, and passes the prior integer assignment as a partial MIP
start. HiGHS warnings are inspected through model status rather than promoted
to exceptions. These are runtime changes only; all mathematical inputs and
acceptance tolerances match v1.

The ignored registry is written as `started` before worker launch. A timeout,
exception, infeasibility, nonoptimal solver return, verification failure, or
successful completion all finalize the same record. The code refuses a second
official run for that platform. Replacing a run therefore requires an explicit
user decision and a deliberately new registered configuration; it never occurs
as an automatic retry or tuning loop.

## Comparison boundary

The only timing comparison is:

- laptop: complete Windows CPU/HiGHS/NumPy system;
- Spark: complete ARM64 DGX Spark/cuOpt/CuPy system inside the pinned image.

cuOpt's MIP solver includes CPU and GPU work. Hardware, operating systems,
Python versions, and solvers differ, so any ratio is labeled system-to-system.
No isolated screening, same-host CPU, fixed-commitment LP, warmup, or repeated
timing result belongs to the registered comparison.
