# ACTIVSg10k v3 CPU profile

## Outcome

The separately authorized `activsg10k-v3` laptop benchmark ran exactly once and
finished inside the global boundary at 283.648 seconds. It returned
`incomplete_restricted_master_not_optimal`: round 2 reached HiGHS `TimeLimit`
at objective 2,224,833.477308, bound 2,223,593.923442, and relative gap
`5.57145e-4`. The required gap is `1e-6`. No retry or Spark run occurred, and
independent verification was not reached.

V3 used commit `3687dea52997f9551fa4f1b1ba4e15c88070b37f`, tag
`benchmark-10k-v3`, and configuration SHA-256
`3f3bcbfec96a7c018d062c385d8867c907dfe69a14dd4b2d3603175e985a6fb8`.
Its raw inputs, mathematical model, exact source PMIN values, tolerances,
deadline, reserves, and unseeded dynamic constraint-generation loop are
identical to v2. Logging is the only intended benchmark change.

## End-to-end attribution

| Phase | Wall time (s) | Share of 283.648 s |
|---|---:|---:|
| Raw input loading | 0.101 | 0.04% |
| Model and FP64 factor construction | 8.844 | 3.12% |
| Solver-session construction | 0.035 | 0.01% |
| Round 1 HiGHS solve | 60.239 | 21.24% |
| Round 1 exhaustive screen | 1.948 | 0.69% |
| Round 2 HiGHS call | 209.237 | 73.77% |
| Round 2 exhaustive screen | 2.058 | 0.73% |

The two solver calls consumed 269.497 seconds, or about 95.0% of the complete
run. Both exhaustive 171,488,782-side screens together consumed 4.007 seconds,
or about 1.41%. Constraint screening is not the bottleneck.

## Round 1 internals

Round 1 solved the 33,913-row restricted master to `Optimal` in 60.239 seconds
with 35 nodes and 52,891 LP iterations. HiGHS attributed 0.88 seconds to
presolve and 59.34 seconds to solve. Within solve, 18.01 seconds went to the main
MIP and 41.33 seconds to 10 sub-MIP calls. The iteration report attributed 227
iterations to strong branching, 8 to separation, 19,243 to heuristics, and
10,701 to two repair LPs.

The resulting objective was 2,201,957.040411 with bound 2,201,955.195613 and
gap `8.38e-7`. Its exhaustive screen took 1.948 seconds and found 322 new
security pairs, with maximum violation 8.287 p.u. All 322 were appended.

## Round 2 internals

Appending the 322 rows took 0.0169 seconds. Installing the previous integer
commitment as a partial MIP start took 0.00045 seconds. Neither operation is a
material bottleneck.

The material surprise occurred inside `Highs.run()`. HiGHS first attempted to
find a feasible solution by fixing the user-supplied discrete values and solving
an LP. That phase lasted about 30.73 wall seconds, ended with an unknown/highly
infeasible LP state, and HiGHS declared the MIP start infeasible. The prior
round's commitment therefore did not provide a useful warm start after the 322
security rows were imposed.

Only after that attempt did the main security-constrained MIP begin. HiGHS
reported 178.49 seconds of native MIP timing: 0.74 seconds in presolve and
177.75 seconds in solve. The solve portion consisted of 104.17 seconds in the
main MIP and 73.58 seconds across 8 sub-MIP calls. It performed 87,567 LP
iterations: 9,922 attributed to strong branching, 30 to separation, and 42,854
to heuristics. The native log used 12 of 24 logical threads but one MIP search
worker with parallel search off.

The root, cut, and heuristic work was substantial. HiGHS found its first logged
incumbent after 15.99 native MIP seconds. The final incumbent appeared at 90.27
seconds while the logged node count was still zero. Positive processed-node
counts first appeared around 96 seconds. At the time limit, HiGHS had reached
473 nodes, 87,567 LP iterations, and only 0.44% reported tree exploration. The
incumbent had not improved after 90.27 seconds; subsequent work primarily
improved the dual bound.

The configured native limit was 178.474 seconds, but the full `Highs.run()` call
took 209.237 wall seconds. The approximately 30.73-second fixed-integer LP
attempt was outside the native MIP timing. The current HiGHS `time_limit`
therefore did not act as a complete wall-clock cap on the persistent solve call.
The global parent boundary still held because sufficient reserve remained.

## Why another round would still be needed

After the time-limited round-2 incumbent returned, a provisional exhaustive
screen again checked all 171,488,782 sides in 2.058 seconds. It found 8 new
violated pairs, with maximum violation 0.233254 p.u. Because the restricted
master was not optimal, those pairs were not added and are not asserted to be
the exact violation set of an optimal round-2 solution.

This run gives direct evidence for the rounds: adding the 322 pairs changed the
commitment and dispatch, and that new incumbent exposed 8 pairs that were not
violated by the round-1 dispatch. The expensive operation is re-solving the
tighter MILP, not discovering the new pairs.

## Evidence boundary

The ignored raw result, registry, JSONL event log, and native HiGHS log were
hashed into the tracked structured profile. The result is not N-1 secure, not
optimal at the requested tolerance, and not independently verified. V3 is
closed to reruns.
