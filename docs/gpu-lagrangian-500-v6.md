# ACTIVSg500 GPU Lagrangian experiment v6

V6 is the separately authorized Phase-I-first speed experiment. It preserves
the v5 raw inputs, exact source `PMIN` and `PMAX`, source-derived ten-segment
PWL costs, mathematical model, contingency set and generation, numerical
tolerances, `1e-3` gap target, GPU stack, and 600-second end-to-end deadline.

For every disjunctive child, v6 performs this registered sequence:

1. Build the exact child model and record a simple aggregate feasibility
   interval from the source `PMIN`/`PMAX` values. This check is telemetry and
   cannot prune a child.
2. Run a cold GPU Phase-I PDLP solve with a two-second cap inside the global
   deadline.
3. Prune only when the projected Phase-I dual gives a positive independently
   replayable infeasibility certificate above the registered threshold.
4. Otherwise solve the ordinary cost LP. If Phase I returned a zero-violation
   source primal within the model tolerance, pass its native source-column
   vector as the cost-LP primal warm start. Phase-I row duals are not passed
   because Phase I splits ranged rows into upper inequalities.
5. If the ordinary LP later rejects after contingency rows were added, run the
   original full-budget Phase I on that expanded model. An inconclusive short
   precheck therefore does not weaken the v5 fallback behavior.

The source-column scaling used by Phase I and the cost LP is checked for exact
equality before a warm start is retained. The controller also validates the
prepared child's commitment bounds and security-pair identity before solving.

Success still requires an independently verified secure dispatch, an
independently replayed exhaustive disjunctive cover, and a relative gap no
greater than `1e-3`. No integer solver, CPU branch-and-bound, or custom CUDA
kernel is used. The one-shot launcher refuses pre-existing v6 paths and runs
only from tag `experiment-500-gpu-lagrangian-v6`.
