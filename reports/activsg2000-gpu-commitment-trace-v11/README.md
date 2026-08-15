# ACTIVSg2000 GPU commitment trace v11

## Result

The single authorized DGX Spark run completed in
`601.954` seconds end to end. Its final incumbent is
exhaustively N-1 secure, but the requested `1e-3` MIP gap was **not**
certified: objective `$1,133,047.868434`, bound
`$1,130,541.728576`, gap `0.002211857`. The correct run
status is `incomplete_restricted_master_gap_not_certified` and fixed-commitment pricing is withheld.

Independent raw-input verification passed exact conditional PMIN/PMAX,
integrality, objective reconstruction, DC physics, base limits, and all
`17,563,400` contingency sides. Its maximum
security violation was `1.827e-10`
p.u., below the `1e-5` p.u. tolerance.

## Did unit commitment stabilize?

No -- not by the end of this run. The evidence shows both inter-round and
within-round movement:

- Round 1 ended with 325 committed source rows.
- Round 2 ended with 321; 46 rows changed from round 1 (21 off-to-on and 25
  on-to-off). It produced 33 distinct callback commitments.
- Round 3 ended with 328; 39 rows changed from round 2 (23 off-to-on and 16
  on-to-off). It produced 30 distinct callback commitments.
- Round 3 had a long `395.274`-second plateau, but then six
  later commitment transitions occurred. The last binary change arrived at
  `507.135` solver
  seconds, only
  `7.287`
  seconds before solver return.

The practical conclusion is that the incumbent often looked stable for
minutes, while the lower-bound proof moved slowly, but late commitment changes
still mattered. A long unchanged objective is therefore not enough to declare
the unit commitment stable on this model.

## MIP-start handling

The later-round start policy was exercised correctly. Before each rebuild, a
bounded HiGHS fixed-commitment LP tested whether the prior GPU commitment could
be extended to the newly constrained master. Round 2's prior commitment was
infeasible after 177 security rows were added. Round 3's precheck reached its
10-second limit after another 24 rows. Consequently, neither prior commitment
was submitted to cuOpt; both rounds solved cold. This is a guarded rejection,
not the earlier native-column-size translation bug.

## Evidence files

- `generator-round-detail.csv`: all 544 source rows, exact PMIN/PMAX, every
  round endpoint, final dispatch in MW and p.u., and v9/CPU comparisons.
- `incumbent-transitions.csv`: every distinct within-solve commitment state.
- `incumbent-flips.csv`: every exact unit flip between callback states.
- `round-flips.csv`: exact endpoint flips after each security update.
- `round-summary.csv` and `summary.csv`: compact timing and acceptance data.
- `evidence.json`, `independent-verification.json`, `events.jsonl`, and the
  console logs: immutable provenance and verification evidence.
- `v10-prelaunch-error.log`: the preserved registry failure that occurred
  before case loading or optimization; v11 is the sole computational run.
- `mip-start-smoke.json`: a one-time frozen-image component check proving that
  a five-column canonical full start is translated into the correct native
  vector despite free-variable splitting and column elimination.

The raw ACTIVSg case and large solver result remain ignored by repository
policy. Their source and result hashes are recorded in `evidence.json`.
