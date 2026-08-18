# ACTIVSg2000 GPU Lagrangian v3 replacement result

## Outcome

The single authorized DGX Spark replacement run completed cleanly with status
`deadline_budget_exhausted`. The v2 exception-scope bug is fixed: an infeasible
fixed commitment now prunes only that candidate and the search continues. The
run did not crash, did not call an integer solver, did not perform branch and
bound, did not use a CPU commitment or dispatch seed, and was not retried.

The scientific success gate did not pass. No secure integer primal was found,
so objective, commitment, dispatch, pricing, and MIP gap are unavailable. The
run must not be described as a successful SCOPF solve.

## Root relaxation

The secure continuous root relaxation completed three internal
constraint-generation rounds:

| Round | GPU solve (s) | New logical pairs | Exhaustive maximum violation (p.u.) |
| ---: | ---: | ---: | ---: |
| 1 | 10.485 | 84 | 1.5669193893 |
| 2 | 48.962 | 20 | 0.2348830163 |
| 3 | 36.293 | 0 | 1.4199486031e-12 |

Every screen evaluated 17,563,400 contingency sides. The final root LP
objective was `$1,128,529.233432291`, with 53 fractional commitments. Its
conservative GPU Lagrangian certificate was `$1,128,529.223432247` and replayed
exactly on the DGX.

## Fixed-commitment correction and primal result

Eight deterministic GPU-generated commitments were attempted: 333, 323, 338,
432, 335, 334, 334, and 334 committed units. Each was rejected in 1.66--1.81
seconds because exact PMIN/PMAX substitution made contingency row
`c0603_m0609_lower` constant and violated by 0.38 MW. Each record explicitly
states `fixed_commitment_candidate_only`; no full-model infeasibility was
claimed. This is the intended v3 behavior, whereas v2 aborted after candidate
one.

The common rejection is not a projection false positive. A read-only replay of
the known secure 327-unit CPU commitment against the same 104 root security
pairs successfully built the exact fixed-commitment projection (215 free
dispatch columns and 431 active coupling rows). The GPU candidates were
therefore genuinely infeasible under the exact TAMU PMIN values; the full model
was not.

## Disjunctive lower-bound result

The solver completed 12 disjunctive split transactions before its solve budget
closed. Four splits committed (generator source rows 16, 17, 152, and 320),
eight rolled back fail-closed, and five frontier regions remained. A thirteenth
split on source row 493 was still transactional when the deadline arrived, so
its parent certificate remained in the cover. No region was pruned by an
uncertain numerical result. Many rolled-back children had a zero-violation
Phase-I warm start but their ordinary PDLP cost solve did not cross the strict
`1e-6` p.u. model-residual gate within its 30--90 second child budget.

The final recorded global lower bound was `$1,128,528.259328175`. A separate
laptop replay reread the raw case and contingencies, reconstructed all five
frontier certificates, verified the disjunctive cover, and returned
`$1,128,528.259328180`: a difference of `5.12e-9` dollars. The bound is valid,
but it is `$0.964104` weaker than the root certificate because the conservative
certificate for one child controls the minimum over the exhaustive cover.

## Timing and CPU context

The measured DGX worker boundary was 1,654.359 seconds (27.573 minutes), with
1,653.389 seconds inside the experiment controller. The NVIDIA GB10 run used a
peak process RSS of 4.918 GiB and a 4.841 GiB CUDA-device-memory delta.

The frozen laptop HiGHS comparison completed in 1,007.370 seconds (16.790
minutes), with objective `$1,133,479.385501136`, bound
`$1,132,599.352235491`, relative gap `7.7639989e-4`, 327 committed units, and
independent secure verification. The laptop completed and the DGX experiment
did not, so this result provides no GPU speedup. It is a system-to-system
comparison, not a solver-kernel comparison. Even using the CPU incumbent only
as an external reference, the final GPU bound implies a 0.436808% gap, above
the requested 0.1%.

## Evidence identity

The experiment is frozen at commit
`6d4c141238c7d67d45fef4bbbbe61d021c072676`, tag
`experiment-2000-gpu-lagrangian-v3`, config SHA-256
`fb867a2c19b131001ddb752cc9f2c1a85a8fcd358f1b25457ed35fe30e827d01`,
and image
`sha256:abe2edbceff89e6d8c74044e680018f473b7a21b35c6d3f0a223a632848e7aee`.
The raw result SHA-256 is
`5d7e8d0ddb9b92b281cb485482b891d479d0720cff2c272f42cbae210ab248d4`.
All artifact hashes and replay values are in [evidence.json](evidence.json).

The frozen preflight contains a manually entered approval timestamp of
`17:47:00Z`, which is 10.1 seconds later than the registry start time. That
timestamp was entered inaccurately. The approval-bearing commit and tag were
created at `17:45:51Z` and `17:45:52Z`, respectively, before the run started at
`17:46:49.887984Z`; the ordering gate was satisfied.
