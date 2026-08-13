# ACTIVSg10k MIP-gap sensitivity experiment

## Question and controlled variables

The v2 experiment measures how the requested HiGHS relative MIP gap changes the
final preventive ACTIVSg10k solution. The five levels are `1e-3`, `1e-4`,
`1e-5`, `1e-6`, and `1e-7`. Every other mathematical input is identical:
immutable source hashes, exact source PMIN/PMAX, ten equal-MW PWL segments,
single-hour demand, DC network equations, outage catalog, model residual
tolerance, and contingency-security tolerance.

Each level is one independent MIP run starting with only base-case constraints.
No learned or preloaded contingency-pair set is used, and results from one gap
do not seed another. Within a run, every violated pair is added in deterministic
order and the persistent HiGHS session supplies the prior round commitment as a
partial MIP start. Constraint-generation LP or MIP rounds are internal to that
one run.

The runs have no wall-clock deadline. A durable ignored registry is written
before each worker starts and prevents a retry for the same gap label.

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

## Evidence outputs

Ignored raw results live under `results/experiments/`. After all five one-shot
runs finish, `scripts/build-gap-sensitivity-report.py` produces a tracked
summary, full generator table, all-bus price table, and exact pairwise metrics
under `reports/activsg10k-gap-sensitivity-v2/`.
