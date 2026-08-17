# ACTIVSg500 GPU Lagrangian experiment v2

V2 preserves the v1 mathematical question and all source-data/model gates. It
is a separately registered post-fix run because the v1 one-shot evidence is
immutable.

## Bound-preserving numerical cleanup

The exact FP64 injection, flow, and angle operators are unchanged and remain
the operators used for contingency screening and full-solution reconstruction.
Only a solver copy of an upper coupling row is cleaned. For original row

`a p <= b`

and cleaned coefficients `a'`, the v2 solver row is

`a' p <= b + max_(l <= p <= u) ((a' - a) p)`.

The maximum is evaluated exactly over the finite dispatch box. Therefore every
point feasible for the unmodified row remains feasible for the cleaned row:
the row is a relaxation and any lower bound from it remains valid for the
original SCOPF. The count and largest dropped coefficient, and every outward
RHS adjustment, are recorded. The frozen threshold is `1e-14`.

On the real ACTIVSg500 base reduced model, the static preflight changed the
smallest nonzero canonical matrix coefficient from approximately `4e-18` to
`1.4316216157616265e-6`. It dropped 7,904 generator-row coefficient instances;
the largest was `3.4528731607818616e-15`. The maximum RHS adjustment on one row
was `6.197676560931975e-12`. Exact physical-flow reconstruction against an
explicit DC solve still differed by at most `1.1439738045737613e-11` MW.

## Primal-feasibility gate and continuation

After each cuOpt PDLP call, v2 checks both cuOpt's independently recomputed
numeric primal-feasibility result and the canonical model residual. A failed
iterate is logged but is never passed to contingency screening, never adds a
security row, and never becomes a primal candidate. If the termination is a
time limit with finite vectors, v2 submits the same native primal and dual
vectors back to the unchanged restricted master and continues within the one
global deadline. When security rows are later appended, the dual warm start is
deterministically zero-extended for the new rows.

All v1 acceptance gates remain: a secure fixed-commitment primal, independent
raw-input verification, CPU replay of every Lagrangian leaf certificate, an
exhaustive disjunctive cover, relative gap at most `1e-3`, and completion within
600 seconds. No CPU branch-and-bound or integer solver is used. The comparison
continues to use the previously hashed 2.702380100-second laptop HiGHS result;
no new laptop run is part of v2.
