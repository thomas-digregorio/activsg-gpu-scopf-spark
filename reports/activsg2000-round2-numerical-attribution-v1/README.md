# ACTIVSg2000 round-2 numerical attribution

This component analysis rebuilds the exact fixed round-2 master used by the
seeded cuOpt diagnostic: 9,220 columns, 8,961 rows, 27,120 nonzeros, 432 binary
commitment variables, and the same 173 canonically ordered contingency rows.
It performs no optimization and changes no model data.

The native cuOpt `MIP Infeasible` text is not evidence that this SCOPF is
mathematically infeasible. The independently accepted CPU state satisfies the
exact conditional PMIN/PMAX bounds, has maximum canonical row violation
`2.4209521e-7`, and passes exhaustive N-1 verification. In the v2 console, cuOpt
also says `Adding initial solution success! feas 1` before its root-relaxation
failure.

## Potential triggers checked

- Poor scaling is confirmed. Absolute nonzero matrix coefficients span
  `0.0015042163` through `142857.142857`, a ratio of `9.4971145e7`.
- Numerical dependence is present among the 173 security rows. Their numerical
  rank is 162; the smallest singular value is `3.7543e-18`, and the maximum
  absolute row cosine is `0.99999249`. No pair exceeds `0.999999`, however, and
  no row was proven redundant against all variable bounds. Removing any of
  these physical constraints would therefore be an unjustified model change.
- Degeneracy indicators are present: 5,766 columns have zero objective
  coefficient, 1,999 columns are free, and there are 5,638 equality rows. The
  equality block has full structural rank 5,638, so this is not a structural
  equality-rank defect.
- A native factorization/barrier failure is directly observed. The v2 console
  records repeated basis factorization failures and repairs, three deficient
  columns during one repair, failure to remove a `1.09e+50` perturbation, a
  primal infeasibility of `4.110060e+09`, and `Barrier Solve status A numerical
  error was encountered.` These numbers arise after the root LP calculation
  diverges; they do not describe the independently verified start.

## Numerical correction

V3 uses an invertible diagonal reformulation only in the cuOpt adapter. The
canonical relation is `x = D z`, where MW power variables use the source case's
100 MVA base and commitment/angle variables retain scale one. Each row is then
multiplied by a positive diagonal row scale. Thus `A x <= b` and
`(R A D) z <= R b` define exactly the same feasible set and objective.

The resulting native matrix coefficient ratio is `7.890e3`, and the objective
coefficient ratio falls from `1.349e3` to `26.30`. On the known secure state,
the maximum native row-activity identity error is `1.7373e-14`, the maximum
canonicalized row-violation identity error is `1.0247e-11`, the variable
round-trip error is `2.2737e-13`, and the objective identity error is
`2.3283e-10` dollars. All are far inside the unchanged acceptance tolerances.

Reproduce the attribution without launching a solver:

```powershell
.\.venv\Scripts\python.exe scripts\analyze-activsg2000-round2-conditioning.py
```

Machine-readable values are in `evidence.json`.
