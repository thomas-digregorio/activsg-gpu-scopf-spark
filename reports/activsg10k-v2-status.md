# ACTIVSg10k v2 one-shot status

## Outcome

The explicitly authorized second laptop CPU benchmark ran exactly once. The
parent watchdog stopped it at 300.010 seconds with
`hard_deadline_exceeded`. Round 2 had not returned, and final independent
verification had not started. This is a failed acceptance gate, not an
N-1-secure solution. No retry or DGX Spark run was performed.

The frozen identity is commit
`1ab1342d627d8499ed7aafdecd699eb47ac618ad`, tag `benchmark-10k-v2`, and
configuration SHA-256
`4a6a1b9703de6919947c472bea9d6c7c768972a667918f4cd54f2a71119c8d9e`.
The v1 and ACTIVSg500 tags and evidence remain unchanged.

## Unchanged mathematical contract

The v2 raw-input and model sections are identical to v1 when parsed as JSON.
The immutable source hashes, one-hour preventive formulation, exact
source PMIN/PMAX values, 10 equal-MW PWL segments, tolerances, and complete
branch-N-1 contingency catalog did not change. Only the HiGHS session strategy
and measured-time allocation changed.

The case has 1,937 eligible source-online generators and 548 unavailable
source-offline generators. Eligible PMIN totals 85,764.93 MW and PMAX totals
170,021.33 MW. The maximum absolute error of the source-derived 10-segment PWL
curves is 2.587250178 in the source cost units. These are production-cost
curves, not submitted market offers.

## Last complete optimization evidence

Round 1 returned HiGHS `Optimal` in 59.680 seconds at objective
2,201,957.040411, bound 2,201,955.195613, and relative gap `8.38e-7`. Its
incumbent committed 1,520 generators. The NumPy screen then exhaustively checked
171,488,782 monitored sides in 2.006 seconds, found 322 violated contingency
pairs, and added all of them in deterministic pair-ID order. The maximum
violation was 8.287 p.u. Round 1 therefore passed the restricted-master solve
but failed the security screen; it was never a final accepted solution.

Round 2 was a materially tighter MILP because it included those 322 security
rows. It began at 70.971 elapsed seconds with 179.029 seconds of solver budget,
but HiGHS did not return before the global deadline. The preserved result has no
round-2 incumbent, bound, gap, or screen. Peak process RSS was 1,341,493,248
bytes (about 1.25 GiB).

## Deadline defect and post-run correction

The frozen adapter set the round-2 native HiGHS limit to the prior HiGHS runtime
plus the remaining call budget: about `59.680 + 179.029 = 238.709` seconds. That
native limit exceeded the actual solve budget by about 59.680 seconds, so the
300-second parent watchdog was the first hard stop. This is the concrete v2
deadline-accounting defect. The evidence is consistent with HiGHS applying the
configured limit to the new run, but no claim about undocumented internal clock
semantics is needed to establish that the adapter's cap was too large.

After preserving v2, the adapter was changed to cap each native call at no more
than its current remaining budget after incremental setup. Solve duration is now
measured with an independent wall clock, and the configured native limit is
recorded. Tiny tests cover the cap. This correction is after the frozen v2 tag
and does not alter or replace the failed result.

Any further ACTIVSg10k benchmark must use a separately approved identity. The
Spark gate remains closed because v2 did not finish as `optimal_verified`.
