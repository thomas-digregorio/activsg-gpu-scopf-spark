# ACTIVSg2000 GPU LP-certificate v14 result

The single approved v14 run is closed as
`incomplete_continuous_solve_not_optimal`. It did not report infeasibility and
did not use branch-and-bound. All 432 source-online commitment columns were
relaxed to `[0,1]`, and cuOpt used FP64 PDLP with per-constraint residuals.

V14 fixed the prior checker defects. Both completed restricted-master solves
passed the independent numerical dual reconstruction and exhaustive CuPy
screening:

| Round | Rows before solve | cuOpt solve (s) | Primal objective | cuOpt dual | Conservative reconstructed bound | New N-1 pairs | Maximum violation (p.u.) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,788 | 1.665332 | 1,118,296.700187 | 1,118,296.700187 | 869,168.654605 | 84 | 1.566919341 |
| 2 | 8,872 | 83.995705 | 1,127,661.401771 | 1,127,661.401737 | 900,321.062254 | 20 | 0.234883007 |
| 3 | 8,892 | 120.005333 | not returned | not returned | not returned | not screened | not screened |

Round 3 reached the configuration's 120-second per-round cap. Its final native
log line showed a provisional objective near `1,128,529.23`, primal residual
`2.87e-10`, dual residual `5.56e-3`, and absolute primal-dual gap near `0.113`,
but cuOpt returned `TimeLimit` rather than `Optimal`. The frozen adapter
therefore exposed no primal or dual vector, did not perform another security
screen, and stopped cleanly. This was the immediate blocker: only 208.669
seconds of the 600-second outer boundary had elapsed, leaving approximately
391 seconds that the per-round cap did not allocate.

The run added 104 unique contingency pairs and completed two exhaustive screens
of all 17,563,400 outage/monitored-line sides. It did not reach a zero-violation
final screen, independent final verification, or the comparison against the
required `1e-3` lower bound of `1,131,914.820566`. No final LP certificate was
accepted.

The large difference between cuOpt's reported dual objective and the
independently reconstructed box-dual bound is preserved rather than hidden.
The reconstructed value is deliberately used as the conservative candidate;
the reported value is telemetry only. Resolving that dual-vector/postsolve
disagreement and deciding how to allocate the unused global time are separate
changes that require a new frozen identity and run approval.

## Timing and memory

- end-to-end wall time: `208.669164070` seconds
- cumulative cuOpt solve time: `205.666369534` seconds
- adapter/controller solver time: `206.198515763` seconds
- cumulative exhaustive screening: `1.141945804` seconds
- peak process RSS: `1,323,315,200` bytes
- peak CUDA device-memory delta: `859,934,720` bytes
- peak CuPy pool use: `70,301,696` bytes

## Frozen evidence hashes

- result JSON: `73d913937a95ac95f72d14fc6b3d847e0a6ceac30978d14e66faa60386a2c8a6`
- one-shot registry: `c628017d1824ef6db63c25d83705ea1d458842955a344c6bf0db3b5a22db7dea`
- worker/native console: `f0ab2d22b090256a14612adfe5a5b44276ae09562bf6ac774dff028e6ca04f6d`
- launcher console: `bc3aec2a254beb4a3df01b4fe3aa602aba9a9f01498257759ae416afdf1dfb6e`
- frozen commit/tag: `20878c68418685007ca365a9b33b0235404b4998` /
  `experiment-2000-gpu-lp-certificate-v14`
