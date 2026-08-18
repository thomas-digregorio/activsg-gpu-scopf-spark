# ACTIVSg2000 round-2 scaled diagnostic v3

## Outcome

The per-unit diagonal reformulation fixed the cuOpt root-LP numerical failure.
The root relaxation finished by dual simplex in 1.67 seconds with a finite
objective. cuOpt then explored 2,709 nodes and returned a finite incumbent,
finite lower bound, and certified `0.0008422865` relative gap within the
requested `1e-3` tolerance.

This diagnostic is nevertheless closed as
`diagnostic_incumbent_returned_but_acceptance_gate_failed`. Its improved
dispatch satisfied the fixed 173-row master but the required exhaustive screen
found 14 new contingency pairs, with maximum violation `0.1517224017` p.u. The
independent raw-input checker confirmed the same security failure. This is the
expected reason for another constraint-generation round; it is not a numerical
failure and not an infeasibility result.

## Frozen identity and hashes

- Commit: `877e3ed43a5a5e686d6f8fb8d01738c794f8329d`
- Tag: `diagnostic-2000-gpu-round2-cpu-seed-v3`
- Config SHA-256:
  `5aafdcaff964d65142eb3cfffe4980e07bd8646fcd8af5f6403cffa61f5178f3`
- Raw result SHA-256:
  `95754cc479008a4032143f50acd0f02700d0b06c2aabf1d052948e0cbc1d454d`
- Console SHA-256:
  `a9181dec00be93e80354c071fd863ab5a6675839c8b1d33bfeb616c5f5ebb938`
- Registry SHA-256:
  `d77077d9bb716d649d7e602a44d3823bd8377e305ed3cfe6a7714523d8a4ef93`

## Solver and acceptance facts

- Objective: `1,131,794.3544796335` dollars.
- Lower bound: `1,130,841.0593988446` dollars.
- Certified relative gap: `0.0008422864781158988`.
- Native solve time: `20.03793261` seconds.
- End-to-end time: `23.85321776` seconds.
- Native maximum constraint/integer/bound residuals:
  `1.5606e-11`, `6.6613e-16`, and `1.3323e-15` in scaled native units.
- Exhaustive screen: 17,563,400 sides, 14 violations above `1e-5` p.u.;
  maximum pair `c2344_m2007_lower` at `0.1517224017` p.u.
- Independent maximum model residual: `1.0140e-9` p.u.
- Independent maximum security violation: `0.1517224017` p.u.

The next run restores the required dynamic loop: solve, exhaustively screen,
add every newly violated pair in stable order, and re-solve while time remains.
It retains the same exact-PMIN model and partial prior-commitment MIP starts.
