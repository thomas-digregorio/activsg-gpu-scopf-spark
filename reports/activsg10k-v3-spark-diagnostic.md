# ACTIVSg10k v3 DGX Spark diagnostic

## Outcome

The explicitly authorized DGX Spark diagnostic ran exactly once, with no retry,
and stopped cleanly at `deadline_budget_exhausted` after 253.750 seconds. It was
a nonofficial `solve` run because the matching laptop result had not passed the
official laptop-first gate. It used the unchanged `activsg10k-v3` mathematical
model and configuration hash, no seeded security-pair list, no custom kernels,
cuOpt for the MILP, and CuPy for exhaustive screening.

Round 1 solved the 33,913-row base restricted master and found 322 violated
contingency pairs. Round 2 solved the resulting 34,235-row master, but its
exhaustive screen found six more violated pairs, led by
`c8726_m8049_lower` at 0.157519 p.u. A third solve was therefore required. The
controller stopped before that solve because only the protected verification
and serialization reserve remained. Independent verification was not reached.

The last cuOpt result carried an `Optimal` status label, objective
2,224,250.176636, bound 2,224,050.063925, and gap `8.99686e-5`. That gap is
about 90 times the configured `1e-6` requirement, so this report does not accept
the label as satisfying the numerical gate. The six new security pairs would
independently prevent acceptance even if the gap requirement had been met.

## End-to-end attribution

| Phase | Wall time (s) | Share of 253.750 s |
|---|---:|---:|
| Raw input loading | 0.067 | 0.03% |
| Model and FP64 factor construction | 2.053 | 0.81% |
| CuPy screen-workspace construction | 0.339 | 0.13% |
| Solver-session construction | <0.001 | <0.01% |
| Round 1 cuOpt adapter call | 48.230 | 19.01% |
| Round 1 exhaustive screen | 1.492 | 0.59% |
| Round 2 cuOpt adapter call | 199.457 | 78.60% |
| Round 2 exhaustive screen | 0.480 | 0.19% |

The two cuOpt calls consumed 247.687 seconds, or 97.61% of the measured run.
Both exhaustive 171,488,782-side screens together consumed 1.972 seconds, or
0.78%. As on the laptop, repeated MILP optimization—not contingency
screening—was the dominant cost.

## Round details

| Metric | Round 1 | Round 2 |
|---|---:|---:|
| Restricted-master rows | 33,913 | 34,235 |
| Adapter wall time (s) | 48.230 | 199.457 |
| Native solve time (s) | 47.682 | 199.144 |
| Objective | 2,201,957.040454 | 2,224,250.176636 |
| Bound | 2,201,955.868701 | 2,224,050.063925 |
| Reported gap | `5.321e-7` | `8.997e-5` |
| MIP nodes | 13,184 | 50,076 |
| Simplex iterations | 63,402 | 1,152,011 |
| Screen time (s) | 1.492 | 0.480 |
| New pairs | 322 | 6 |
| Maximum security violation (p.u.) | 8.287000 | 0.157519 |

Peak process RSS was 5,039,640,576 bytes. The maximum recorded CuPy pool use
was 850,997,760 bytes, and the recorded CUDA device-memory delta was
12,205,293,568 bytes.

## Laptop comparison

The v3 laptop run took 283.648 seconds; this Spark diagnostic took 253.750
seconds, 29.898 seconds or 10.54% less wall time (a 1.118 laptop/Spark ratio).
Round 1 was 60.239 seconds on the laptop and 48.230 seconds on Spark. Round 2
was 209.237 seconds on the laptop and 199.457 seconds on Spark.

This is only a system-to-system diagnostic comparison, not a pure GPU speedup.
The laptop used HiGHS, a persistent incremental solver session, and the prior
commitment as a partial MIP start. Spark used cuOpt, rebuilt the model each
round, and did not pass a prior commitment start. The systems also ended with
different incumbents, gaps, and provisional violation sets. Neither run passed
the acceptance gates.

The laptop partial-MIP-start implementation has been preserved. The Spark
container and launcher changes do not alter `src/` or `configs/` relative to
the v3 CPU benchmark commit; they only make the authorized diagnostic runnable
in the pinned container.

## Identity and evidence boundary

- Execution commit: `533c8d5c66057fbe4a5067f06b758f874099ef7c`
- Execution tag: `benchmark-10k-v3-spark-diagnostic`
- Configuration SHA-256:
  `3f3bcbfec96a7c018d062c385d8867c907dfe69a14dd4b2d3603175e985a6fb8`
- Case SHA-256:
  `ead10b25fecc4dcc02f88bacdfb3526fe8b8985b81f7e539c95abddb32575590`
- Contingency SHA-256:
  `7e1681a960b0a2a99d824766e0e94cc291fa36a7ec33b6dee24bf12ac67ddef7`
- Raw result SHA-256:
  `5730ba04f0a8ac3e36f9af6fa7f8770d63fb4ee8f71e5b7fc8d852104db8966a`
- Checkpoint SHA-256:
  `b4da631bf40066c52dd13581d1f3343b43ae1c59f46bbe6861bc5463159592ec`

The ignored raw result and checkpoint remain local evidence. This run is
closed to automatic reruns and is not an optimal, verified N-1 solution.
