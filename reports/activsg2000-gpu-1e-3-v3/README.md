# ACTIVSg2000 DGX Spark 1e-3 v3 result

The one authorized v3 run ended `incomplete_no_incumbent` in
1667.116 seconds. It is not a successful SCOPF result.

Round 1 returned cuOpt `FeasibleFound` with objective
1,118,442.384611, bound 1,118,297.105972, and
gap 1.298937e-04. The finite-bound certificate passed. The
mandatory exhaustive screen then checked
17,563,400 sides, found
173 violations, and added all 173 rows.
The provisional round-1 solution committed 325 generators.

Round 2 received the prior values for all
432 integer
commitment columns. After 1659.012 seconds,
cuOpt returned native `Infeasible` with no incumbent or finite bound. Therefore
there was no round-2 screen, independent verification, final commitment or
dispatch, or fixed-commitment pricing. The native `mip_gap=0` attached to that
no-incumbent status is not an accepted gap and is omitted from the summary.

The native status does **not** establish that the mathematical SCOPF is
infeasible. The accepted laptop result uses the same immutable inputs and model
contract and passes the complete N-1 set; it is therefore a feasible witness
for any subset of the 173 rows added after GPU round 1. This result instead
isolates a cuOpt/adapter numerical or solver-behavior issue in the rebuilt
secured master.

## Provisional round-1 audit

The raw-input checker was run without another MIP solve. It passed the basic
model with maximum residual
7.957e-11 p.u. and exact conditional
PMIN/PMAX residual
2.855e-12 p.u. It
failed N-1 security, as expected, with maximum violation
2.045910 p.u. across
17,563,400 sides.

`generator-detail.csv` preserves all 544 exact source PMIN/PMAX rows and labels
the GPU round-1 commitment/dispatch as provisional. GPU pricing columns are
blank. `bus-prices.csv` contains the accepted laptop prices solely as a
reference; every GPU price field is blank. `round-detail.csv`, `summary.csv`,
and `comparison.json` preserve timing, status, hashes, and acceptance gates.

## Timing and peak memory

- Model and factor build: 0.283 s
- Round 1 cuOpt solve: 5.573 s
- Round 1 CuPy exhaustive screen: 1.079 s
- Round 2 cuOpt solve: 1659.012 s
- End to end: 1667.116 s
- Peak process RSS: 6,451,380,224 bytes
- Peak CUDA device-memory delta: 12,171,591,680 bytes
- Peak CuPy pool use: 70,301,696 bytes

No v3 retry was performed.
