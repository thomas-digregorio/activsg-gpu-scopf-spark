# ACTIVSg500 laptop CPU versus DGX Spark gap comparison

This is an end-to-end system comparison, not a pure GPU speedup claim. The laptop path uses HiGHS and NumPy; the Spark path uses cuOpt and CuPy for the MIP and contingency screens, followed by HiGHS inside the same Spark container for fixed-commitment nodal pricing.

Both suites use the same immutable ACTIVSg500 source hashes, exact source PMIN/PMAX, one-hour preventive model, ten-segment source-derived cost curves, security tolerance, and five requested gap levels. Every included result is optimal, independently verified, exhaustively N-1 secure, and priced.

| Gap | CPU objective | Spark objective | Commitment flips | Dispatch L1 (MW) | Mean price difference ($/MWh) | CPU wall (s) | Spark wall (s) | CPU/Spark wall ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-3 | 79,410.651432 | 79,410.651432 | 0 | 0.000000 | 0.000000 | 2.702 | 4.917 | 0.550 |
| 1e-4 | 79,410.651432 | 79,410.651432 | 0 | 0.000000 | 0.000000 | 2.685 | 3.931 | 0.683 |
| 1e-5 | 79,410.651432 | 79,410.651432 | 0 | 0.000000 | 0.000000 | 2.726 | 3.909 | 0.697 |
| 1e-6 | 79,410.651432 | 79,410.651432 | 0 | 0.000000 | 0.000000 | 2.676 | 3.671 | 0.729 |
| 1e-7 | 79,410.651432 | 79,410.651432 | 0 | 0.000000 | 0.000000 | 2.710 | 3.945 | 0.687 |

## Interpretation

The two solver stacks selected the same commitment at every gap. No generator dispatch or nodal-price difference exceeded `1e-6` in its reported unit. The largest absolute objective difference was 5.821e-10, the largest generator dispatch difference was 1.722e-11 MW, and the largest bus-price difference was 1.602e-12 $/MWh.

On this small case, the DGX Spark end-to-end path took 1.372x to 1.819x the laptop wall time. This is a latency-dominated system result, not evidence that GPU acceleration is intrinsically slower for larger instances.

`generator-detail.csv` contains exact PMIN/PMAX, commitment, MIP dispatch, pricing dispatch, and generator-bus prices in MW and p.u. units for both systems at every gap. `bus-prices.csv` contains all 500 paired nodal prices. `comparison.json` records exact vector-difference metrics, frozen identities, and raw-result SHA-256 hashes.
