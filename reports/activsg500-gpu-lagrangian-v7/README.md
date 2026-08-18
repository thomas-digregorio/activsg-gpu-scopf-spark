# ACTIVSg500 GPU Lagrangian v7 result

The one authorized v7 DGX Spark run completed successfully with status
`optimality_gap_certified_gpu_lagrangian`. It found an independently verified
preventive N-1 dispatch and certified a `0.057702%` upper/lower-bound gap,
inside the requested `0.1%` limit. End-to-end wall time was 58.682906 seconds.

V7 used frozen commit `ffc29cea4345545627191cfa8230cfbe6394c96b`
and tag `experiment-500-gpu-lagrangian-v7`. It used no integer solver, no CPU
branch-and-bound, and no custom CUDA kernel. The mathematical model, exact
source PMIN/PMAX values, source-derived PWL costs, raw inputs, and tolerances
were unchanged from v5.

## Result and runtime comparison

| System/run | Status | Objective | Lower bound | Gap | End-to-end seconds |
|---|---|---:|---:|---:|---:|
| Laptop HiGHS | optimal, independently verified | 79,410.651432182 | 79,410.651432182 | 0 | 2.702380 |
| DGX Spark v5 | GPU Lagrangian gap certified | 79,410.651432180 | 79,364.830084234 | 0.057702% | 139.691012 |
| DGX Spark v7 | GPU Lagrangian gap certified | 79,410.651432180 | 79,364.830084232 | 0.057702% | 58.682906 |

The Phase-I-first change reduced the DGX workflow by 81.008105 seconds, or
57.99%, and made v7 2.380 times as fast as v5. The registered laptop run was
still 21.715 times faster end to end. These are system-to-system comparisons
between different hardware, algorithms, and solver stacks, not pure GPU kernel
speedups.

The GPU result is a valid `1e-3`-gap solution, not a claim of exact global
optimality. Its absolute objective/bound gap is 45.821348 dollars. The certified
bound is 33.589303 dollars above the minimum bound needed for the requested
gap.

## Phase-I-first result

The controller performed a short Phase I immediately after creating each of
the two children at all nine splits:

- 18 pre-cost-LP Phase-I attempts were made;
- eight off-children were certified infeasible and pruned without an ordinary
  cost-LP attempt;
- ten zero-violation Phase-I primals were eligible as cost-LP starts;
- all nine active on-child cost solves consumed those starts; and
- one off-child needed an ordinary cost-LP sequence followed by fallback
  Phase I.

The one exception was the first off-child, `r_s001_0`. Its initial Phase-I
model contained 167 logical security pairs and had a zero-violation solution,
so the precheck could not prune it. The warm-started cost solve then found two
new pairs, `c0225_m0230_upper` and `c0226_m0229_upper`, with a maximum violation
of 2.359427 p.u. After adding them, two short cost-LP attempts encountered dual
divergence and reached their local limits. Fallback Phase I on the resulting
169-pair model certified a conservative violation lower bound of 0.0548642303
p.u., safely above the `1e-6` p.u. prune threshold.

Consequently, ordinary infeasible-child adapter time fell from 91.077957
seconds in v5 to 10.522671 seconds in v7, saving 80.555286 seconds. This explains
almost the entire 81.008105-second end-to-end reduction. It also explains why
the measured result was about ten seconds above the earlier optimistic
48.6-second estimate: one child still needed contingency discovery before
Phase I could prove it infeasible.

Adding security screening and row generation inside the short Phase-I precheck
is the clearest remaining version of this optimization. It is a performance
opportunity, not an outstanding correctness failure in v7.

## Primal solution and prices

The raw result serializes all 90 source generator rows, including source status,
commitment, exact PMIN/PMAX, MW and p.u. dispatch, and ten PWL segment dispatch
values. It also serializes 500 bus prices in dollars/MWh and dollars/p.u.-hour.
There are 56 source-online eligible generators, of which 50 are committed.
Total dispatch is 7,750.66 MW, or 77.5066 p.u.

The GPU and registered laptop results have:

- identical commitment vectors, with zero unit differences;
- maximum generator-dispatch difference of `1.02e-9` MW;
- objective difference of `-2.36e-9` dollars; and
- maximum bus-price difference of `3.45e-11` dollars/MWh.

GPU prices range from 6.87 to 48.23781158 dollars/MWh. Both price vectors are
from a secure fixed-commitment continuous pricing solve. The GPU values are
cuOpt PDLP demand-dual prices; they are not MILP duals or submitted offers.

The independent raw-input primal checker passed all gates:

- 337 valid branch outages and 401,704 monitored sides checked;
- maximum contingency violation: `9.67e-12` p.u.;
- maximum model residual: `5.46e-13` p.u.;
- conditional exact-PMIN/PMAX violation: `5.68e-16` p.u.;
- zero fractional commitments; and
- zero objective-recalculation difference.

## GPU lower-bound certificate

The root Lagrangian bound was 77,393.120159792, corresponding to a 2.540631%
gap. Nine deterministic generator disjunctions raised the bound as follows:

| Certified cover | Split generator source row | Lower bound | Gap |
|---|---:|---:|---:|
| Root | - | 77,393.120160 | 2.540631% |
| 1 split | 16 | 77,996.459657 | 1.780859% |
| 2 splits | 17 | 78,666.140270 | 0.937546% |
| 3 splits | 3 | 78,985.052450 | 0.535947% |
| 4 splits | 64 | 78,985.052450 | 0.535947% |
| 5 splits | 62 | 78,985.052450 | 0.535947% |
| 6 splits | 61 | 78,985.052450 | 0.535947% |
| 7 splits | 63 | 79,014.735899 | 0.498567% |
| 8 splits | 80 | 79,127.718842 | 0.356290% |
| 9 splits | 79 | 79,364.830084 | 0.057702% |

The final cover has one active Lagrangian leaf and nine Phase-I-pruned leaves.
The active leaf contains one fractional commitment in its continuous
relaxation. Every pruned leaf has a conservative Phase-I bound above the
registered `1e-6` p.u. threshold. No split failed or was rolled back.

The independent DGX replay reconstructed all ten leaves from raw inputs,
verified the disjoint exhaustive cover, and reproduced the global bound with
zero difference. A second read-only laptop replay also passed. It reproduced
the bound within `4.37e-11` dollars, Phase-I values within `1.57e-14` p.u., and
LODF values within `7.99e-15`, below the registered `1e-12` tolerance.

The laptop replay counted up to 602 additional near-zero sparse coefficients
at the `1e-14` cleanup boundary. Their maximum numerical certificate effect was
only `3.59e-11`, and all outward-relaxation and replay invariants passed. This is
recorded as cross-architecture floating-point variation, not hidden.

## Timing attribution

| Work | Seconds | Share of total |
|---|---:|---:|
| Two rejected root commitment candidates | 20.314373 | 34.6% |
| Successful fixed-commitment candidate | 1.032138 | 1.8% |
| One remaining off-child cost-LP sequence | 10.522671 | 17.9% |
| Eighteen prechecks plus one fallback Phase I | 3.899802 | 6.6% |
| Relaxation PDLP adapters | 5.809253 | 9.9% |
| CuPy relaxation screening | 0.993718 | 1.7% |
| GPU Lagrangian evaluation | 5.096802 | 8.7% |
| Final independent verification | 1.004995 | 1.7% |
| Full launcher boundary | 58.682906 | 100% |

Peak recorded process RSS was 1.384 GB and peak CUDA device-memory growth was
0.965 GB. The full result remains in ignored local and DGX result storage; its
hash and all supporting artifact hashes are recorded in
[evidence.json](evidence.json). Frozen pre-run evidence is in
[preflight.json](preflight.json). No retry or duplicate v7 full run was
performed.
