# Texas2k Series24 cold CPU scenario experiment

## Authorized comparison

Run one laptop CPU HiGHS experiment for each official Texas2k Series24
MATPOWER scenario, in source order:

1. 2016 summer peak
2. 2016 low load
3. 2024 summer peak
4. 2024 low load
5. 2024 high renewables
6. 2024 low load with grid-forming inverters

Every scenario is one hour, uses a `1e-3` relative MIP-gap target, and has a
hard 1,800-second end-to-end deadline. Raw loading, model construction, all
constraint-generation rounds, independent exhaustive verification, and result
serialization are inside that boundary. The optional
fixed-commitment pricing LP is disabled because this experiment requests solve
time and MILP result, not prices.

## Cold-start rule

Each scenario launches in a new Python worker with a new HiGHS instance. Round
1 receives no solution from any earlier scenario. Later constraint-generation
rounds within the same scenario may pass that scenario's preceding commitment
as a partial MIP start; those rounds remain part of one run.

No operating-system file-cache flush is claimed. Here, cold means no
cross-scenario incumbent, basis, solver model, or solver process is reused.

## Immutable sources

Official source page:
<https://electricgrids.engr.tamu.edu/activsg2000-dynamics-cases-2024/>

The downloaded archive is not tracked by Git.

| Source | SHA-256 |
|---|---|
| `Texas2k_series24_cases_with_dynamics.zip` | `954c515a55c186bf1d987efdde0e2873b4a694bc4bf3522c47bb2b1ab58a0fb9` |
| Case 1 MATPOWER file | `4eb378bb7c5abeba02a1be61752983d90a17dcaac3460ea2bb0c4986d5aa00b2` |
| Case 2 MATPOWER file | `29e8a640fc7f0e6660746a7784ae1168ac7f17b5e5741dae449247cd9525b819` |
| Case 3 MATPOWER file | `3f2243b14d73bf923ac0b5deee3bf56f9e93aa0f6d6afbfc5618bf9b9d761e46` |
| Case 4 MATPOWER file | `f69585bcca1b2bc601a257d3ef2c8181d052e7ecc5dd739962c64b32c6ee313f` |
| Case 5 MATPOWER file | `c7d96c0dfe560509eaeecf0de0d9767274cd7b79318cd4b5bf0ebff1468b1d85` |
| Case 6 MATPOWER file | `8710b40e9eafb38463d9a1135a59e70f14fbad38c99ca90af598928a5d0e7b86` |

The source files are Windows-1252 text. They are decoded without changing
their bytes, and the SHA-256 check precedes parsing.

## Mathematical contract

- Source-online generators are commitment eligible; source-offline generators
  remain unavailable.
- Every source `PMIN` and `PMAX` is retained exactly.
- Each source polynomial production cost is converted to ten equal-MW convex
  chord segments over exact `[PMIN, PMAX]`.
- The model is preventive: commitment and dispatch are common to the base case
  and every included branch outage.
- There is no load shedding, spillage, overload slack, reserve requirement,
  corrective redispatch, or feasibility repair.
- The archive contains no MATPOWER `contab` files. Therefore, each scenario
  generates one deterministic outage candidate per immutable source branch
  row, excludes source-offline and islanding branches with recorded reasons,
  and exhaustively screens every remaining branch outage on both flow sides.
- Generator outages remain deferred.

Aggregate PMIN/PMAX checks and LODF-versus-explicit-DC validation are setup
checks, not benchmark solves. Success requires a HiGHS-certified `1e-3` gap,
zero final contingency violations above `1e-5` p.u., and an independent
verification pass. An aggregate capacity check alone is not a feasibility
certificate.

## One-shot commands

After committing and tagging the frozen implementation as
`experiment-texas2k-series24-cpu-v1`, invoke `benchmark` exactly once for each
configuration:

```powershell
.\.venv\Scripts\activsg-scopf benchmark `
  --config "configs\texas2k-series24-case1-1e-3.json" `
  --platform laptop_cpu `
  --output "results\experiments\texas2k-series24-case1-cpu.json"
```

Invoke the corresponding command separately for cases 2 through 6 only after
the preceding worker exits and its result is inspected. Every benchmark ID has
its own durable one-run registry.
