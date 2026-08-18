# ACTIVSg500 GPU Lagrangian experiment v7

V7 is the prepared replacement for the failed v6 run. V6 remains immutable:
it stopped before its root LP because a new validator treated generator
source-row dictionary keys as canonical commitment-column indices.

V7 changes that lookup to:

```text
generator source row -> commitment_by_generator[source row] -> canonical column
```

The regression test uses an eligible generator whose source row is nonzero and
whose commitment column is zero, the exact identity pattern the original tiny
fixture did not cover. A read-only ACTIVSg500 model-build check also validated
all 56 eligible source-row-to-column mappings without running an optimization.

Everything else is identical to v6: raw inputs, exact source `PMIN`/`PMAX`,
source-derived ten-segment PWL costs, mathematical model, contingency logic,
Phase-I-first controller, two-second precheck cap, numerical tolerances,
`1e-3` target, DGX solver profile, and 600-second end-to-end deadline.

V7 is not authorized to run merely by being prepared. Its one-shot launcher
must remain unused until an explicit replacement-run approval is received.
