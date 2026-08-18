# ACTIVSg2000 GPU LP-certificate v13 failed attempt

The single approved v13 replacement stopped after its first continuous
restricted-master solve. It did not time out and it did not report
infeasibility. cuOpt translated all 432 commitment columns as continuous,
selected FP64 PDLP Stable3, and returned `Optimal` with no branch-and-bound:

- cuOpt solve time: `1.388015313` seconds
- end-to-end wall time: `3.023588047` seconds
- primal objective: `1,118,296.700341088`
- dual objective candidate: `1,118,296.699947574`
- native absolute/relative gap: `3.94e-4 / 1.76e-10`
- native absolute/relative primal infeasibility: `3.05e-7 / 9.73e-9`
- native absolute/relative dual infeasibility: `2.98e-5 / 2.78e-10`

The experiment correctly withheld a bound and stopped before contingency
screening because the independent checker rejected the certificate. Two
checker assumptions were wrong:

1. It attempted to reproduce cuOpt's absolute primal threshold using the
   exposed pre-presolve row RHS norm. cuOpt reports the relative primal
   residual as `9.73e-9`, within the configured `1e-8`, while the independent
   maximum native row violation was `1.129119e-7`, within the model's `1e-6`
   acceptance tolerance.
2. One syntactically free angle column, canonical column 5516
   (`theta_b7095`), had a nonzero numerical stationarity residual. The angle is
   physically bounded by the existing DC flow equations, finite `RATE_A`
   limits, and fixed reference angle, but those implied bounds were not
   available to the box-dual reconstruction.

The root restricted-master dual candidate is not an exhaustive N-1 bound. No
contingency screen ran, so it cannot be compared with the requested
`1,131,914.820566` threshold as a final certificate.

Version 0.20.2 derives finite, redundant angle bounds from shortest paths of
the existing per-branch inequality
`|theta_i-theta_j| <= |shift| + RATE_A/|baseMVA*b|`. For ACTIVSg2000, all
2,000 buses are connected by 3,206 finite-RATE_A branches; the construction
tightens only the 1,999 non-reference angle domains and changes no physical
feasible point. It also uses the configured `1e-6` model tolerance for primal
screening and enables cuOpt's documented per-constraint PDLP residual mode.
Any separately approved replacement must use v14 paths and identity.

## Frozen evidence hashes

- result JSON: `cac181b54e8c1c35dbbba8ed2e344a45aa3c2b1c1fa86ebca60e1f91c56dd225`
- one-shot registry: `ef6e6b477ed8bef588d2c012780fbb6e0546694aa968187b2fc0267857a1ef61`
- worker/native console: `8f56f8f72fab58e7c27f553557e38b18e331786eb325e5c2186de760f245b78e`
- launcher console: `b063716a65ca4f04d897a7b5ac34eb2e7791e7b32a19e79b0b1da8208d629fcd`
- frozen commit/tag: `9c2fe161c3cece10ac11a967af902558f5acfbfb` /
  `experiment-2000-gpu-lp-certificate-v13`
