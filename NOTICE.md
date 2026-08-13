# Notices and attribution

## ACTIVSg500 input data

The benchmark consumes the synthetic ACTIVSg500 case published by the Texas A&M University Electric Grid Test Case Repository. The raw case is not tracked in this repository.

- Repository page: https://electricgrids.engr.tamu.edu/electric-grid-test-cases/activsg500/
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
