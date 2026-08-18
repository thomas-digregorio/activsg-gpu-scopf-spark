# ACTIVSg500 GPU Lagrangian experiment v5

V5 is the separately approved replacement for the failed v4 verifier run. It
preserves the v4 raw inputs, exact source `PMIN` and `PMAX`, source-derived
ten-segment PWL costs, mathematical model, GPU solver and screening stack,
constraint-generation logic, numerical tolerances, `1e-3` gap target, and
600-second end-to-end deadline.

The only changes are evidence-handling corrections discovered by v4:

- Phase-I dual rows use source-row identity and inequality side rather than a
  diagnostic insertion position. Ordered and semantic hashes are both checked.
- Frozen v4-style certificates remain replayable after their original ordered
  hash is validated, then their duals are aligned by semantic row identity.
- Cross-architecture cleanup replay requires exact policy and proof invariants,
  while counts and magnitudes of coefficients below the registered `1e-14`
  cleanup threshold remain reported FP64 telemetry.
- The incumbent-relative gap is recomputed whenever the frontier bound changes,
  including before any checkpoint that might become a failure result.

All v4 controller gates remain active: bounded warm/cold PDLP attempts, exact
security-row deduplication with every logical pair retained, transactional
split rollback, replayable GPU Phase-I pruning, immediate durable incumbent
verification, and independent replay of the complete active/pruned cover.

Success requires an independently verified secure dispatch, a complete
independently replayed disjunctive cover, and a relative gap no greater than
`1e-3`. No integer solver or CPU branch-and-bound is permitted. The one-shot
launcher refuses pre-existing v5 paths and runs only from tag
`experiment-500-gpu-lagrangian-v5`.
