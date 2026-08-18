# ACTIVSg500 MIP-gap sensitivity v1

Each gap level is one independent laptop MIP run with a hard 1,800-second limit from the base master. Within each run, dynamic N-1 constraint generation retains the previous commitment as a partial MIP start. Gap levels are not seeded from one another. Prices are from a separately identified fixed-commitment, N-1-secure LP and are not MILP duals.

Every completed result passes independent exhaustive branch-N-1 verification and includes accepted fixed-commitment pricing. The controller prevents a later gap from starting if an earlier gap times out, fails, or lacks pricing.

| Requested gap | Achieved gap | Objective | Bound | Committed | MIP rounds | Wall (s) | Mean price ($/MWh) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 0.000e+00 | 79,410.651432 | 79,410.651432 | 50 | 3 | 2.702 | 26.877180 |
| 1e-4 | 0.000e+00 | 79,410.651432 | 79,410.651432 | 50 | 3 | 2.685 | 26.877180 |
| 1e-5 | 0.000e+00 | 79,410.651432 | 79,410.651432 | 50 | 3 | 2.726 | 26.877180 |
| 1e-6 | 0.000e+00 | 79,410.651432 | 79,410.651432 | 50 | 3 | 2.676 | 26.877180 |
| 1e-7 | 0.000e+00 | 79,410.651432 | 79,410.651432 | 50 | 3 | 2.710 | 26.877180 |

## Differences from the 1e-7 result

| Gap | Commitment flips | MIP dispatch L1 (MW) | Maximum dispatch change (MW) | Mean absolute bus-price change ($/MWh) | Maximum bus-price change ($/MWh) |
|---:|---:|---:|---:|---:|---:|
| 1e-3 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1e-4 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1e-5 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1e-6 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

For this ACTIVSg500 formulation, the requested MIP-gap setting never became binding. HiGHS proved objective equal to bound with a reported zero gap in every restricted-master round at every requested setting. Consequently, `1e-3` produced exactly the same generator commitment, MIP dispatch, fixed-commitment pricing dispatch, and all nodal prices as `1e-7`.

The common solution commits 50 of the 56 source-online generators and dispatches 7,750.660000 MW (77.506600 p.u.). Source-online rows 5, 6, 27, 35, 39, 40 are uncommitted; all 34 source-offline generators remain unavailable. The fixed-commitment pricing dispatch differs from the MIP dispatch by at most 1.455e-11 MW.

Prices range from $6.87000000/MWh to $48.23781158/MWh, with a mean of $26.87717951/MWh. On the 100 MVA base, the minimum, maximum, and mean are $687.00000000, $4,823.78115797, and $2,687.71795052 per p.u.-hour, respectively.

Thus `1e-3` is sufficient for this specific ACTIVSg500 instance and solver path. This does not establish that `1e-3` is sufficient for a larger ACTIVSg case or a different unit-commitment formulation.

Full source-row generator data are in `generator-detail.csv`; all 500 bus prices are in `bus-prices.csv`; exact metrics and evidence hashes are in `comparison.json`.
