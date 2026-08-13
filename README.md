# ACTIVSg500 GPU DC-SCOPF on DGX Spark

Fresh, auditable prototype for a single-hour preventive branch-N-1 DC security-constrained commitment and dispatch problem on the synthetic TAMU ACTIVSg500 case.

The project compares exactly one end-to-end laptop CPU run with exactly one end-to-end DGX Spark run. Each official run has a hard five-minute limit. The first implementation uses HiGHS and NumPy/SciPy on the laptop, and NVIDIA cuOpt plus CuPy on the Spark. It does not contain custom CUDA kernels.

## Scope guardrails

- ACTIVSg500 only. Larger ACTIVSg cases require explicit approval.
- Exact source-case `PMIN` is enforced whenever a generator is committed.
- Source polynomial production costs are converted to ten equal-MW piecewise-linear segments; they are not represented as submitted market offers.
- Branch contingencies only in the first version. Generator outages are deferred.
- Raw cases, derived matrices, solver work, and benchmark results are not committed.
- The repository and every writable runtime path must be outside OneDrive.

## Repository status

Implementation is developed on `codex/initial-implementation` and published through a draft pull request. Usage instructions will be added with the implementation.

## Licensing

No license is granted for the original code at this time. See [NOTICE.md](NOTICE.md) for data and dependency attribution.

