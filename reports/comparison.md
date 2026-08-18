# ACTIVSg500 one-shot benchmark result

Both registered runs passed every acceptance gate. On this small one-hour
ACTIVSg500 MILP, the laptop CPU system completed in **3.0052 s** and the DGX
Spark system completed in **4.5717 s**. The Spark took 1.521 times the laptop
time, so this result does not show an end-to-end speedup from the Spark.

This is a one-shot system-to-system comparison. It is not a pure GPU comparison:
the hardware, operating systems, Python versions, solvers, and screening stacks
all differ, and no warmup or repetition was allowed.

## Acceptance and agreement

| Metric | Laptop CPU | DGX Spark |
|---|---:|---:|
| Status | `optimal_verified` | `optimal_verified` |
| Total boundary time | 3.0052048 s | 4.5717223 s |
| Objective | 79,410.6514321820 | 79,410.6514321823 |
| Best bound | 79,410.6514321820 | 79,410.6514321823 |
| MIP gap | 0 | 0 |
| Committed generators | 50 | 50 |
| Constraint-generation rounds | 3 | 3 |
| New pairs by round | 167, 2, 0 | 167, 2, 0 |
| Total added pairs | 169 | 169 |
| Maximum model residual | 1.212e-12 p.u. | 3.710e-13 p.u. |
| Final exhaustive violation | 0 p.u. | 4.206e-14 p.u. |
| Exhaustively checked sides | 401,704 | 401,704 |
| Peak process RSS | 93.59 MiB | 1,078.32 MiB |
| Peak CUDA device-memory delta | n/a | 5.882 GiB |

The systems used the same commit, configuration hash, case hash, contingency
hash, 169 deterministic pair IDs, and final commitment count. The absolute
objective difference was `2.474e-10`; maximum generator-dispatch difference was
`6.594e-12 MW`; maximum base-flow difference was `3.339e-11 MW`.

## What was solved

- 56 source-online generators were commitment-eligible; 34 source-offline rows
  remained unavailable.
- Every on generator used its exact TAMU `PMIN`; the eligible `PMIN` sum was
  2,659.06 MW and the independent conditional-bound violation was at numerical
  noise (`2.785e-14` p.u. on the laptop result).
- Each source polynomial production-cost curve used ten equal-MW chord segments
  over exact `[PMIN, PMAX]`. These are source-derived production costs, not
  submitted offers.
- The base model had 1,769 columns, 1,713 rows, and 4,833 nonzeros. Dynamic
  generation ended at 1,882 rows and 5,171 nonzeros.
- 337 source-listed branch outages were valid. All 254 branch exclusions were
  islanding bridges. Ninety source generator outages were recorded as deferred.
- The independent checker reread both raw files and explicitly solved all 337
  post-outage DC networks; no slack, shedding, spillage, or repair was available.

## Boundary breakdown

| Included stage | Laptop CPU | DGX Spark |
|---|---:|---:|
| Raw loading and hashing | 0.00574 s | 0.00341 s |
| Network/model/factor build | 0.14827 s | 0.09132 s |
| Three solver rounds | 1.99356 s | 2.69516 s |
| Three exhaustive screens | 0.01186 s | 0.79637 s |
| Independent verification | 0.35975 s | 0.20629 s |
| Result serialization | 0.01333 s | 0.00136 s |

The boundary also includes worker startup, platform validation, checkpointing,
and controller overhead, so the listed stages do not sum exactly to total time.
With no warmup permitted, CUDA context/startup and host-device setup remain part
of the Spark screening cost. That is appropriate for the requested end-to-end
boundary but means this small case does not amortize GPU startup overhead.

## Frozen evidence

- Commit: `08b06fc2d76b1df4472da690a2d01dedefc14815`
- Tag: `benchmark-v1`
- Configuration SHA-256:
  `61a147a74c73970860bf3b9f00f29100e520c97946f1f0e4b05cd249c6a5fdbf`
- Laptop detailed result SHA-256:
  `ec673b6dcf0ecc80133a3e3b858c5d74f1dc5b8702dd9810e71926a9cba78c21`
- Spark detailed result SHA-256:
  `a4bd65e935efe8d32806b94e44be1a04e6fa551d27270bf2ec2824b0a78c2ffe`

Detailed result files and run registries remain ignored locally. The tracked
machine-readable summary is [`official-results.json`](official-results.json).

## Spark setup disclosure

Two container preflight launches failed before registry creation, raw-case
loading, or the measured worker boundary: first because the non-root container
lacked a writable home for CuPy's cache, then because the cuOpt base image lacked
Git for frozen-commit attestation. The registered launch used fresh ephemeral
cache paths and an attestation-only derived image adding Ubuntu Git 2.34.1. It
retained the exact frozen application image and cuOpt base. Neither preflight
performed a model solve, screen, or benchmark round.

