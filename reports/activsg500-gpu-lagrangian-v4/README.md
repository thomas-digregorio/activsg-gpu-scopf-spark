# ACTIVSg500 GPU Lagrangian v4 result

The one authorized v4 DGX Spark run stopped with `failed_exception` after
39.087285 end-to-end seconds. It did **not** certify the requested `1e-3`
relative gap, so it is not a successful solve. The failure was in independent
certificate replay after the first committed split, not in the primal SCOPF
solution or cuOpt's Phase-I solve.

The run used the frozen commit `dee8d9b614cd57111a7d7f2f7972db546fb47a7f`
and tag `experiment-500-gpu-lagrangian-v4`. It did not use an integer solver,
CPU branch-and-bound, or a custom CUDA kernel. The source model, exact source
PMIN values, PWL costs, contingency set, and registered tolerances were not
changed.

## Preserved result

The GPU found and immediately serialized the same 50-unit commitment as the
registered laptop HiGHS result. The fixed-commitment PDLP objective was
79,410.651432180, only 2.36e-9 dollars below the laptop value because of FP64
solver arithmetic. The raw result contains the complete commitment, MW and
p.u. dispatch, and all bus prices. These prices are fixed-commitment PDLP
demand-dual prices, not MILP duals.

The independent raw-input checker passed before bound refinement:

- 337 valid branch outages and 401,704 monitored sides checked;
- maximum contingency violation: `9.67e-12` p.u. on DGX;
- maximum model residual: `5.46e-13` p.u.;
- conditional exact-PMIN/PMAX violation: `5.68e-16` p.u.; and
- zero fractional commitments and zero objective-recalculation difference.

Thus the incumbent is a valid preventive N-1 solution. It is not yet a
globally gap-certified solution.

## Bound refinement reached before the exception

The root GPU Lagrangian lower bound was 77,393.120159792. V4 split generator
source row 16. The off child stalled in bounded warm and cold PDLP attempts,
then the GPU Phase-I LP solved in 0.102842 native seconds and produced a
conservative infeasibility bound of `0.054864230309151` p.u., well above the
registered `1e-6` p.u. pruning threshold. The on child solved normally and
produced a conservative Lagrangian lower bound of 77,996.459656986.

The committed cover therefore consists of one active child and one
Phase-I-pruned child. The corrected independent verifier replays that complete
cover and the 77,996.459656986 bound. On DGX all replay differences are exactly
zero. On the laptop, the largest bound difference is `5.82e-11` dollars, the
largest Phase-I difference is `6.52e-16` p.u., and the largest LODF difference
is `7.99e-15`, below the registered `1e-12` tolerance.

The current bound implies a relative gap of **1.780859%**, not the
`2.540631%` stored in the failed payload. The stored value is the root gap and
was stale because the exception occurred between frontier replay and the next
loop-level gap update. A `1e-3` certificate would require a bound of at least
79,331.240780747 for this incumbent, leaving a 1,334.781124-dollar bound
shortfall. More disjunctive refinement is therefore still required.

## Failure and corrections

The frozen verifier encoded a Phase-I row identity as
`phase1__<insertion-index>__<side>__<source-row-name>`. The GPU master appended
a newly discovered security pair at the end, while raw-input replay rebuilt
the same pair set in deterministic pair-ID order. The inequalities and dual
multipliers matched, but the diagnostic insertion indices did not, causing
`Phase-I certificate row identity mismatch`.

Post-run commit `d35059ad38fb770731453ec475c8b7d4cd361e4f` fixes this by:

1. keying Phase-I duals by source-row identity and side, independent of row
   insertion order, while retaining ordered and semantic hashes;
2. remaining backward-compatible with the frozen v4 certificate;
3. validating cleanup proof invariants while treating cross-architecture
   counts of sub-`1e-14` FP64 dust as telemetry; and
4. recomputing the displayed relative gap whenever the certified frontier
   changes.

Commit `44adaa6ea10af338609a0a673f9498991fb11c3a` adds a read-only replay utility.
The untouched v4 artifact now passes both primal and lower-bound replay on the
DGX container and on the laptop. The full local suite passes with 185 tests and
15 GPU-only skips; Ruff also passes.

## Timing attribution

| Work | Seconds |
|---|---:|
| Raw load and reduced-network build | 0.074584 |
| Root PDLP adapters | 1.475463 |
| Root exhaustive GPU screening | 0.984040 |
| Root GPU Lagrangian evaluation | 1.983096 |
| Three bounded commitment candidates | 21.328333 |
| Off-child restricted attempts | 10.602302 |
| Off-child GPU Phase-I adapter | 0.126734 |
| On-child PDLP, screen, and Lagrangian evaluation | 0.882796 |
| Accounted work subtotal | 37.457347 |
| Full launcher boundary | 39.087285 |

The registered laptop HiGHS run completed optimally and independently verified
in 2.702380 seconds. The incomplete v4 GPU path consumed 14.464 times that
wall time before its verifier exception. This is a failed system-to-system
comparison, not a pure CPU-versus-GPU speedup measurement.

No automatic replacement run was launched. Machine-readable measurements and
artifact hashes are in [evidence.json](evidence.json); frozen pre-run evidence
is in [preflight.json](preflight.json).
