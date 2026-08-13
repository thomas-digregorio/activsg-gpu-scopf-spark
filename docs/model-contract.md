# Version 1 model contract

## Scope and variables

The model contains one one-hour interval and one preventive decision vector.
For each source-online generator `g`, it has binary commitment `u_g`, dispatch
`p_g`, and ten incremental cost-segment variables `y_gs`. Source-offline rows do
not receive decision variables and are serialized as unavailable.

For each bus it has a voltage angle in radians, with the first source reference
bus fixed to zero. For each in-service branch it has one from-to MW flow.

## Exact conditional PMIN and PWL cost

For every eligible generator:

```text
p_g = PMIN_g u_g + sum_s y_gs
0 <= y_gs <= segment_width_gs u_g
```

The ten widths partition the exact source `[PMIN, PMAX]` interval equally. The
objective contribution is:

```text
f_g(PMIN_g) u_g + sum_s chord_slope_gs y_gs
```

Thus an off unit costs zero and produces zero; an on unit cannot produce below
the unmodified TAMU `PMIN`. The original high-to-low polynomial coefficients,
segment widths/slopes, committed base cost, and maximum analytic chord error are
included in each result. These are source-derived production-cost curves, not
submitted market offers.

## DC network

For active branch `l=(i,j)`, MATPOWER tap `tau_l` defaults to one when its source
value is zero:

```text
b_l = 1 / (x_l tau_l)
f_l = baseMVA b_l (theta_i - theta_j - shift_l)
```

Nodal balance is `generation - C^T f = PD + GS`. Positive `GS` is therefore MW
demand at nominal voltage. Positive `RATE_A` bounds both flow directions.
Non-default MATPOWER angle-difference bounds are applied when present. There are
no loss terms or balancing/slack variables.

## Branch contingencies

The parser groups contingency-table rows by source label. Version 1 accepts only
a single `CT_TBRCH` replacement setting `BR_STATUS` to zero for an in-service
branch with finite nonzero reactance. Removing it must leave the active network
connected. Every exclusion and every deferred `CT_TGEN` label is recorded with
a reason.

FP64 PTDF/LODF columns use the same tap-aware DC susceptance matrix. Three
deterministically spaced columns are validated against explicit post-outage DC
solutions before optimization. For valid outage `k` and monitored line `m`:

```text
f_m_after_k = f_m + LODF_mk f_k
```

The restricted master begins with only base constraints. After each optimal
solve, NumPy or CuPy screens every `(outage, monitored line, lower/upper side)`.
Every violation above `1e-5` p.u. is appended in deterministic contingency-label,
monitored-source-row, side order. No violated pair is sampled or truncated.

## Independent acceptance checker

The checker rereads and rehashes both raw files. It does not trust solver row
activities. It verifies all generator rows, binary bounds/integrality, exact
conditional PMIN/PMAX, all PWL segments, objective reconstruction, `PD + GS`
balance, reference angle, DC equations, base limits, and active angle limits.
It then performs an explicit post-outage sparse DC solve for every eligible
branch outage and checks every applicable monitored-line side.

