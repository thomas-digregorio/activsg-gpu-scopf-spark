# ACTIVSg MIP-gap sensitivity experiments

## Question and controlled variables

The registered experiments measure how the requested solver relative MIP gap
changes the final preventive solution. The five levels are `1e-3`, `1e-4`,
`1e-5`, `1e-6`, and `1e-7`. Within each case, every other mathematical input is
identical:
immutable source hashes, exact source PMIN/PMAX, ten equal-MW PWL segments,
single-hour demand, DC network equations, outage catalog, model residual
tolerance, and contingency-security tolerance.

Six identities are registered: the preserved unbounded
`activsg10k-gap-sensitivity-v2` suite and the bounded
`activsg500-gap-sensitivity-v1`, `activsg2000-gap-sensitivity-v1`, and
`activsg500-gpu-gap-sensitivity-v1` suites, plus the single-level
`activsg2000-gpu-gap-sensitivity-v1` and explicitly authorized replacement
`activsg2000-gpu-gap-sensitivity-v2` suites. The first three use laptop HiGHS
and NumPy. The GPU suites use DGX Spark cuOpt and CuPy. Both ACTIVSg2000 GPU
identities authorize only `1e-3`; no later GPU gap is registered.
Results never cross-seed between cases or gap levels.

Each level is one independent MIP run starting with only base-case constraints.
No learned or preloaded contingency-pair set is used, and results from one gap
do not seed another. Within a run, every violated pair is added in deterministic
order. The laptop's persistent HiGHS session and the Spark adapter both supply
the prior round commitment as a partial MIP start; Spark rebuilds the cuOpt
restricted master around that hint. Constraint-generation LP or MIP rounds are
internal to that one run.

The ACTIVSg10k v2 runs have no wall-clock deadline. Each bounded laptop or Spark
run has a hard 1,800-second end-to-end limit. Its solver receives
the decreasing global budget, with 120 seconds reserved for verification and
pricing and 15 seconds reserved for serialization. The parent worker watchdog
is the hard boundary. A durable ignored registry is written before each worker
starts and prevents a retry for the same gap label. A later bounded-suite gap is
blocked unless all prior gaps completed as `optimal_verified` with accepted
fixed-commitment pricing. The CPU and GPU suites have separate registries and
raw-result names, so new Spark evidence cannot overwrite completed laptop
evidence.

The preserved v1 `1e-3` attempt failed when HiGHS returned `kError` while
processing a round-2 partial commitment start. The user explicitly authorized
the v2 replacement. V2 still supplies the prior commitment. If and only if
HiGHS returns that internal start-processing error, the adapter records the
failed attempt, rebuilds the identical restricted master without the hint, and
continues cold. Other failures remain terminal.

## Recorded MIP quantities

For each level the raw result retains the final objective, lower bound, achieved
relative gap, commitment, dispatch, all constraint-generation rounds, complete
exhaustive screen, and independent verification. Every generator is keyed by
its immutable MATPOWER source row. Exact source PMIN, PMAX, and dispatch are
stored in MW and on the case base:

`P_pu = P_MW / baseMVA`.

## Pricing definition

A mixed-integer solution has no generally valid LP dual price. After the MIP
passes independent verification, the experiment fixes every commitment to the
returned binary value and solves a continuous preventive SCOPF. The pricing LP
begins with the final MIP security rows, is exhaustively screened, and adds any
new violated branch-outage pair until its own dispatch is N-1 secure.

The price at each bus is the HiGHS dual of that bus's nodal-balance equality,
with the equality written as generation minus net flow equals demand. It is the
objective derivative for one additional MW of demand during the one-hour
interval, reported as $/MWh. The equivalent derivative when demand is expressed
in p.u. is:

`price_per_pu_hour = price_per_mwh * baseMVA`.

The pricing-LP dispatch is stored separately because fixing commitment and
fully optimizing the continuous variables can improve upon the MIP incumbent's
dispatch. Price comparisons can also reflect LP dual degeneracy, so identical
primal commitment and dispatch do not guarantee bitwise-identical duals.

For the GPU suite, this LP runs with HiGHS 1.15.1 inside the same DGX Spark
container after the cuOpt/CuPy stages. Its time and memory remain inside the
same end-to-end Spark boundary. This is necessary for nodal row duals and does
not reclassify the MIP stage as a CPU solve.

## Evidence outputs

Ignored raw results live under `results/experiments/`. After all five one-shot
runs finish, `scripts/build-gap-sensitivity-report.py --suite <suite-id>`
produces a tracked summary, full generator table, all-bus price table, and exact
pairwise metrics under the corresponding directory in `reports/`. The
generator table includes exact source PMIN/PMAX, commitment, MIP dispatch,
fixed-commitment pricing dispatch, and nodal price for every gap level in both
MW-based and p.u.-based units.

After the Spark suite finishes, `scripts/build-activsg500-cpu-gpu-comparison.py`
builds a paired report against the already completed laptop suite. That report
is explicitly a laptop-system versus Spark-system comparison, not a pure GPU
speedup claim.

## ACTIVSg2000 v1 outcome

The bounded ACTIVSg2000 campaign stopped after `1e-4` did not meet its requested
gap before the solver budget ended. Its provisional screen retained 18 security
violations, so it was not independently verified or priced. The ordered gate
therefore prevented `1e-5`, `1e-6`, and `1e-7` from starting. The accepted
`1e-3` generator, dispatch, and price records and the clearly labeled partial
`1e-4` evidence are under `reports/activsg2000-gap-sensitivity-v1/`.

## ACTIVSg500 GPU v1 outcome

All five DGX Spark levels completed `optimal_verified` with accepted pricing
and zero reported MIP gap. Each selected the same 50-unit commitment, added the
same 169 contingency pairs over three MIP rounds, and reproduced the laptop
dispatch and all 500 prices within numerical precision. Spark end-to-end time
ranged from 3.671 to 4.917 seconds, versus 2.676 to 2.726 seconds for the prior
laptop results. This is a small-case system comparison and not a pure GPU
speedup measurement.

## ACTIVSg2000 GPU 1e-3 outcome

The one authorized DGX Spark run completed in 7.909 seconds but stopped after
the first restricted master. cuOpt returned native `FeasibleFound` with an
objective of 1,118,437.552574, bound of 1,118,297.105972, and reported gap of
`1.255739e-4`, which is below the requested `1e-3`. The frozen fail-closed
adapter nevertheless required native `Optimal` before adding screened rows.

The exhaustive incumbent screen found 165 violated contingency pairs and a
maximum violation of 2.041592 p.u. No rows were added, and independent
verification and fixed-commitment pricing were not reached in the run. A
post-hoc raw-input checker, run without another MIP solve, passed the basic
model and exact conditional PMIN/PMAX residual gates but failed exhaustive N-1
security over 17,563,400 monitored sides. Consequently the saved 325-unit
commitment and dispatch are provisional, GPU prices are unavailable, and the
lower objective must not be compared with the accepted laptop secure result as
an improvement. The run was not retried. Full evidence is under
`reports/activsg2000-gpu-1e-3-v1/`.

## ACTIVSg2000 GPU replacement acceptance rule

V2 preserves the v1 case hashes, exact PMIN/PMAX, model, requested `1e-3` gap,
security tolerance, 1,800-second boundary, dynamic pair generation, partial
integer MIP starts, and pricing definition. It changes only the frozen identity
and cuOpt restricted-master acceptance gate.

For v2, native `Optimal`, `FeasibleFound`, or `TimeLimit` can proceed to row
generation only when all of these are true:

- A finite incumbent objective and minimization dual bound exist.
- The independently calculated `abs(objective - bound) / abs(objective)` and
  cuOpt's reported gap are both at or below the requested tolerance.
- The bound is valid for minimization within numerical allowance.
- Native maximum constraint, integrality, and variable-bound residuals are all
  present and at most `1e-6`.

After every accepted restricted-master solve, including every re-solve after
rows are added, the complete valid outage-by-monitored-line set is screened.
The run cannot be successful until the requested gap is certified for the
current master and the final screen contains zero violations above `1e-5` p.u.
The independent checker then rereads the immutable raw files and repeats all
model and exhaustive-security gates before fixed-commitment pricing.
