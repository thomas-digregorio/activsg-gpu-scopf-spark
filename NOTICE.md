# Notices and attribution

## ACTIVSg input data

The benchmarks consume the synthetic ACTIVSg500 and ACTIVSg10k cases published by the Texas A&M University Electric Grid Test Case Repository. Raw cases are not tracked in this repository.

- ACTIVSg500: https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg500/
- ACTIVSg10k: https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg10k/
- MATPOWER case format: https://matpower.org/docs/ref/matpower7.1/lib/caseformat.html

Retain the attribution and license notices distributed with the source data.
The distributed MATPOWER source identifies the case as synthetic, cites
Birchfield et al., IEEE Transactions on Power Systems 32(4), 2017,
DOI 10.1109/TPWRS.2016.2616385, and states that the case is licensed under
Creative Commons Attribution 4.0 International.

## Solver and numerical dependencies

The implementation interoperates with HiGHS, NumPy, SciPy, psutil, CuPy, and
NVIDIA cuOpt. Each dependency and container image remains subject to its own
license and terms.

## Original code

No license is granted for the original code in this repository at this time.
