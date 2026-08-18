# ACTIVSg500 GPU Lagrangian v1 result

The one-shot v1 run failed. It did **not** establish that the SCOPF was
infeasible, and it produced no accepted primal objective, lower bound, gap,
commitment, dispatch, or prices.

The frozen identity was commit `951b6eccfd4ad106e005944f5ebae9527875674f`
and tag `experiment-500-gpu-lagrangian-v1`. The configuration SHA-256 was
`9852370a7c1020be594fb1edac9c99ea21374b7286a7a14ef0822839a3950a3b`.

## Observed failure

| Round | Rows | Columns | Nonzeros | cuOpt status | Iterations | Native solve seconds | Final primal infeasibility | Objective |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 1,811 | 672 | 44,558 | Time Limit | 3,945,800 | 120.005 | 13.1 | 0 |
| 2 | 1,814 | 672 | 44,721 | Time Limit | 3,983,200 | 120.004 | 12.6 | 0 |

Both solves remained at a primal-infeasible zero-dispatch iterate. The v1
controller nevertheless screened that iterate after round 1 and added three
apparent contingency rows. After round 2, those rows were excluded from the
"new violation" set even though the same infeasible iterate still violated
them, leading to the misleading terminal message `Region r final screen
residual exceeds tolerance`.

The root numerical trigger was a reduced matrix coefficient range extending
down to approximately `4e-18`; cuOpt reported a large-range warning. The v2
fix retains the exact physical injection operator, drops only solver-row
dispatch coefficients with absolute value at most `1e-14`, and relaxes each
affected upper-row RHS outward by the exact box-domain worst-case error. It
also refuses to screen any PDLP iterate that fails both the native numerical
certificate and a canonical residual check.

## Timing and comparison

The controller boundary was 242.249907491 seconds, versus 2.702380100 seconds
for the registered laptop HiGHS result, so the failed v1 path consumed 89.643x
the laptop time. This is only a system-to-system failure comparison; v1 did
not solve the requested GPU problem and therefore provides no GPU speedup or
solution-quality result.

Peak measurements were 1,121,177,600 bytes process RSS, 480,210,944 bytes CUDA
device-memory delta, and 1,614,848 bytes in the CuPy pool.

## Preserved evidence hashes

| Evidence | SHA-256 |
|---|---|
| Final result JSON | `5e6c82073e1d6bfcbc99fb296b03ba58bf4d568aeff7fe9dfb72f7fdf1213d60` |
| Run registry JSON | `128a3c59a5ffea268c91457feec355e7029b8cc77fcf7e5a1aee2c299876b691` |
| Last checkpoint JSON | `dae62e868f8d7a00c43b6bcf600ec2a4f2d708fec215072aa53663800f0f58b4` |
| Worker console log | `f174193905d4040d74534b7e91bbce41e3bf2293ff977441fd5906808e0047d9` |

The large raw evidence remains ignored under `results/`; this report records
its immutable identity without committing generated solver artifacts.
