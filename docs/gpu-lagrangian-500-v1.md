# ACTIVSg500 GPU Lagrangian experiment v1

This experiment asks whether a GPU continuous solver plus a replayable
Lagrangian/disjunctive certificate can replace CPU branch-and-bound for the
registered one-hour ACTIVSg500 preventive DC-SCOPF at a relative gap of
`1e-3`.

## Mathematical boundary

The model retains the immutable source case, source-online eligibility, exact
source PMIN and PMAX, ten equal-MW PWL production-cost segments, `PD + GS`,
transformer taps and phase shifts, applicable angle limits, base `RATE_A`, and
all registered non-islanding branch outages. Generator outages and corrective
redispatch remain outside the model.

Bus angles and branch flows are eliminated with the same FP64 reduced DC
factorization used by the independent checker. The reduced coupling system is

`sum_g p_g = D`, `H p <= h`,

where `H` contains base limits, applicable angle limits, and dynamically found
post-contingency limits. Every omitted security row relaxes the problem, so it
can weaken but cannot invalidate a lower bound.

For canonical row duals `y` (free on balance and non-positive on upper rows),
the certificate evaluates

`q(y) = y' h + sum_g min_{X_g} (C_g - (H' y)_g p_g)`.

`X_g` is the exact nonconvex local set: the unit is off at zero cost/output, or
on at exact PMIN plus bounded PWL segments. Its minimum is evaluated by
comparing off with on and filling every segment whose adjusted slope is
negative. Thus any sign-valid `y` gives a global MILP lower bound without a
branch-and-bound tree. The cuOpt PDLP multiplier seeds a projected Polyak
supergradient loop whose coupling matrix, multipliers, generator subproblems,
and best-bound state remain in FP64 GPU memory. CuPy performs that loop, and an
independent NumPy replay rebuilds the winning certificate from the raw case.

If the root bound is insufficient, the algorithm splits one commitment and
solves the two child relaxations with cuOpt PDLP. The active leaves form a
disjoint exhaustive cover; the global lower bound is the minimum leaf bound.
This is disjunctive refinement, not a claim that combinatorial worst-case
complexity disappeared.

## Primal and acceptance path

cuOpt PDLP treats commitment columns continuously to generate candidates. A
candidate commitment is then fixed exactly and the dispatch LP is solved with
cuOpt PDLP. CuPy performs a complete outage/monitored-line screen after every
solve, and all violations are added before re-solving. No primal is accepted
without a final exhaustive zero-violation screen and the existing raw-input
checker.

Success requires all of the following in the single registered DGX run:

- a secure fixed-commitment primal independently verified from raw inputs;
- an independently replayed Lagrangian certificate for every frontier leaf;
- a verified disjoint/exhaustive commitment-region cover;
- `(upper_bound - lower_bound) / abs(upper_bound) <= 1e-3`;
- completion within the frozen 600-second end-to-end boundary.

The comparison uses the already-recorded, canonical-JSON-hashed laptop HiGHS result from
the same case/model contract. It is a system-to-system comparison, not a pure
GPU kernel speedup. No new laptop full-model run is authorized by this
experiment.

## Residency statement

PDLP numerical optimization, contingency screening, and the primary local
generator Lagrangian evaluation execute on the DGX GPU. Python/CPU code loads
the immutable inputs, constructs sparse model metadata, selects disjunctive
splits, checkpoints evidence, and independently replays/verifies the final
certificate. No custom CUDA kernel is needed for 56 local generator
subproblems; the configuration permits one later if profiling justifies it.
