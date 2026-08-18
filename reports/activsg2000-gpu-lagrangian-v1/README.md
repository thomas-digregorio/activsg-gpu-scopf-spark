# ACTIVSg2000 GPU Lagrangian v1 result

The one authorized DGX Spark run stopped cleanly with status
`deadline_budget_exhausted`. It remained inside the 1,800-second outer limit,
finishing in 1,652.818277 seconds, but it did **not** produce a secure integer
incumbent. Therefore it has no GPU objective, dispatch, prices, internally
certified MIP gap, or final incumbent N-1 verification. This is an incomplete
run, not an infeasibility result.

The run did produce a valid GPU-resident lower-bound result. Its three-region
frontier bound is 1,128,529.223414 dollars. The DGX raw-input replay reproduced
that bound exactly, and a separate laptop replay reproduced it within
`7.68e-9` dollars. No integer solver, CPU branch-and-bound, or custom CUDA
kernel was used. Exact TAMU source PMIN/PMAX values were unchanged.

The frozen run used commit `b22660acf1a57517b892e2004b01fcd8fce4dc26`,
tag `experiment-2000-gpu-lagrangian-v1`, and container image
`sha256:1de3457655afa395cb1a93457134f79f3661bbea7b627847a4edebe2cdc6b19a`.
No retry or duplicate full run was performed.

## CPU comparison

| System/run | Status | Objective | Lower bound | Gap | End-to-end seconds |
|---|---|---:|---:|---:|---:|
| Laptop HiGHS | optimal, independently verified | 1,133,479.385501 | 1,132,599.352235 | 0.077640% | 1,007.370271 |
| DGX Spark GPU Lagrangian | deadline exhausted, no secure primal | - | 1,128,529.223414 | - | 1,652.818277 |

The DGX run used 1.641 times the laptop wall time and took 645.448006 seconds
longer, while failing the primal and gap gates that the laptop passed. This is
a system-to-system comparison between different algorithms and solver stacks,
not a pure GPU speedup comparison.

Using the already-known CPU incumbent only as an external diagnostic, the GPU
bound is 4,950.162087 dollars below that incumbent, a 0.436723% gap. The bound
would have needed to be 3,816.682701 dollars higher to establish a 0.1% gap
against that particular incumbent. These are not internal GPU gap claims,
because this run never found its own verified incumbent.

## What completed

The root continuous relaxation completed in three constraint-generation
rounds. It added 104 logical contingency pairs (81 distinct solver rows), then
exhaustively screened all 17,563,400 valid outage/monitored-line sides with a
maximum final violation of `1.42e-12` p.u. The root LP objective was
1,128,529.233432 dollars and contained 53 fractional commitments.

The device-resident 512-iteration Lagrangian evaluation took 2.071624 seconds
and produced the conservative root bound 1,128,529.223432. Two generator
disjunctions, on source rows 16 and 492, were completed successfully, producing
three active frontier regions. Six other split attempts were rolled back when
one child could not be numerically certified. A ninth split was in progress at
the deadline; its completed child was discarded transactionally and the
parent certificate remained in the valid frontier.

The final lower bound, 1,128,529.223414 dollars, is effectively unchanged from
the root bound. Thus disjunctive refinement did not materially strengthen the
certificate on ACTIVSg2000 during this run.

## Commitment-candidate behavior

| Candidate origin | Committed units | Change from prior | Last residual (p.u.) | Seconds | Result |
|---|---:|---:|---:|---:|---|
| Root PDLP rounding | 333 | - | 0.141543 | 62.980 | rejected |
| Root Lagrangian minimizer | 323 | 24 bits | 0.074322 | 62.953 | rejected |
| Root PDLP threshold 0.25 | 338 | 25 bits | 0.028698 | 90.828 | rejected |
| All-online fallback | 432 | 94 bits | 0.001267 | 90.866 | rejected |
| Region `r_s004_0` rounding | 335 | 97 bits | 0.123248 | 63.231 | rejected |
| Region `r_s004_1` rounding | 334 | 3 bits | 0.132891 | 63.349 | rejected |
| Region `r_s004_1_s008_0` rounding | 333 | 1 bit | 0.510250 | 63.564 | rejected |

The last three commitment counts stabilized at 335, 334, and 333, and the last
two transitions changed only three bits and then one bit. That did not imply
dispatch feasibility: their residuals worsened. The run consequently has no
unit-commitment vector that can be reported as feasible, and it would be
misleading to serialize dispatch or prices from any rejected candidate.

## Bottleneck

PDLP adapter calls consumed 1,571.390041 seconds, or 95.07% of total wall time,
across 73 calls. The main buckets were:

| PDLP work | Seconds |
|---|---:|
| Hard children in rolled-back split attempts | 541.707586 |
| Seven fixed-commitment candidates | 486.050058 |
| Successful relaxation regions | 278.330262 |
| Solved siblings later rolled back transactionally | 122.877434 |
| In-progress final split children | 110.293722 |
| Phase-I prechecks and fallbacks | 32.130979 |

By contrast, recorded exhaustive CuPy screening for completed and retained
region solves totaled about 1.45 seconds, and recorded GPU Lagrangian evaluation
totaled about 5.69 seconds. The bottleneck is therefore cuOpt PDLP convergence
and numerical certification on the 10,076-row, 4,014-column fixed-region LPs,
not contingency screening or the device-resident Lagrangian kernel.

All 16 short Phase-I prechecks found zero violation, so none could prune a
child. They did provide eligible primal warm starts. Six fallback Phase-I
solves also returned zero primal violation but had `dual_bound_not_strong_enough`
certificates, so the controller correctly retained the parents rather than
claiming infeasibility.

## Verification and artifacts

The final three-region cover was independently replayed from raw inputs on the
DGX with zero bound difference. The laptop cross-architecture replay passed
with a `7.68e-9` dollar global-bound difference and a `3.31e-14` maximum LODF
difference, below the registered `1e-12` tolerance. Cross-architecture cleanup
dust counts differed by at most 222 coefficients, while the maximum certificate
effect was only `6.11e-10`; all outward-relaxation invariants passed.

Peak recorded process RSS was 4.535 GB and peak CUDA device-memory growth was
4.506 GB. The 27 MB raw result, checkpoint, and console log remain in ignored
local and DGX result storage. Their hashes are recorded in
[evidence.json](evidence.json); frozen pre-run evidence is in
[preflight.json](preflight.json).

The overall acceptance result is **FAIL/incomplete**: the lower-bound
certificate passed, but secure-primal, exhaustive incumbent verification, and
requested-gap certification did not.
