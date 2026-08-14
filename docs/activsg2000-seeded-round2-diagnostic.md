# ACTIVSg2000 seeded round-2 cuOpt diagnostic

This is a separately authorized, one-shot diagnostic. It is not a replacement
benchmark and it does not change the preventive DC-SCOPF model.

The diagnostic rebuilds the exact 8,961-row, 9,220-column, 27,120-nonzero
round-2 canonical master from the frozen GPU v3 evidence. Its last 173 rows are
the security-pair IDs added after the v3 round-1 exhaustive screen, in the same
stable order.

It reconstructs all 9,220 values from the accepted CPU MIP solution:

- 432 source-online commitment variables;
- generator dispatch and all nonzero-width PWL segment variables;
- 2,000 bus angles; and
- 3,206 base branch flows.

Integer values are normalized to exact zero or one. Before cuOpt can start, the
controller checks every canonical row, variable bound, integer value, objective,
and an independent raw-input exhaustive N-1 verification. A failed pre-solve
gate aborts without consuming the authorized solver launch.

The fixed master is then translated once and solved once with cuOpt 26.06.00.
All 9,220 columns receive a MIP start. Native console logging is enabled and the
Spark launch script retains the full console stream. No contingency rows are
generated or added during this diagnostic.

Interpretation is deliberately narrow:

- If cuOpt returns native `Infeasible` after the full start passed every
  pre-solve gate, failure to discover a commitment is excluded. The remaining
  fault domain is model translation or native solver behavior.
- If cuOpt accepts or improves the start, the returned state is screened across
  every valid branch outage and independently verified before it is described
  as accepted.
- A time limit or another native status is reported as observed; it is not
  converted into an infeasibility claim.

The frozen configuration is
`configs/activsg2000-gpu-round2-cpu-seed-diagnostic-v1.json`. V1 preserved the
accepted CPU floating-point values verbatim; cuOpt returned `NoTermination`
before presolve because one dispatch value exceeded its exact column bound by
`5.994e-8`. The approved v2 correction in
`configs/activsg2000-gpu-round2-cpu-seed-diagnostic-v2.json` projects only such
sub-tolerance excesses onto the existing exact variable bounds. The model,
commitment, PMIN/PMAX values, security rows, and tolerances are unchanged.

The two immutable
input results live only under ignored `results/diagnostic-inputs/` paths and are
accepted only at their registered SHA-256 hashes. The run has a hard 1,800-second
controller boundary, while the one cuOpt call receives the exact
1,657.606618394988-second budget recorded for GPU v3 round 2.
