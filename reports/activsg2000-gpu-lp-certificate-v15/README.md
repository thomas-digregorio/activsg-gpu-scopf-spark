# ACTIVSg2000 GPU LP-certificate v15 result

The single approved v15 DGX Spark run completed in `129.385` seconds. It fixed
the earlier deadline, time-limit-vector, warm-start, and dual-coordinate
problems and reached an `Optimal` exhaustive continuous LP after three
solve/screen rounds. It did not use branch-and-bound or custom CUDA kernels.

The result is definitive but does **not** certify the requested `1e-3` MIP gap.
The verified continuous lower bound is `$1,128,529.120570`, while the known
secure integer incumbent of `$1,133,047.868434` requires a lower bound of at
least `$1,131,914.820566`. The resulting certified gap is `0.003988135`
(`0.3988%`), and the lower bound misses the requirement by `$3,385.699995`.
Because round 3 solved the exhaustive LP to optimality, additional PDLP time on
this unchanged relaxation cannot close that gap. A tighter formulation, valid
cuts, or integer search is required for a `0.1%` proof.

## What the fixes changed

- cuOpt LP presolve was explicitly disabled in all three rounds. The native log
  contains three `presolve=0` parameter records and no presolver invocation.
- The independently reconstructed and cuOpt-reported dual objectives now agree:
  their absolute differences were `1.94e-6`, `1.88e-5`, and `8.69e-6` dollars.
  All three stricter numerical dual certificates passed.
- Rounds 2 and 3 received both the prior PDLP primal and row dual. The adapter
  read them back in native coordinates and appended `84` then `20` zero duals
  for the newly generated security rows.
- The dynamic controller offered each solve up to `480` seconds while retaining
  follow-up time. Round 3 finished in `81.674` seconds instead of being stopped
  at v14's 120-second cap.
- A frozen-image component check confirmed that `TimeLimit` vectors are
  retrieved and validated. The production run did not need that fallback
  because every round returned `Optimal`.

The earlier MILP MIP-start column-translation fix remains separate and intact.
This v15 path is a continuous PDLP experiment, so it uses LP primal/dual warm
starts rather than a binary MIP start.

## Solve and screen trace

| Round | Rows | PDLP solve (s) | Primal objective | Reconstructed dual | Added pairs | Maximum screen violation (p.u.) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,788 | 1.326914 | 1,118,296.700187 | 1,118,296.700185 | 84 | 1.566919344 |
| 2 | 8,872 | 43.344190 | 1,127,661.401732 | 1,127,661.401719 | 20 | 0.234883018 |
| 3 | 8,892 | 81.673646 | 1,128,529.234057 | 1,128,529.233423 | 0 | 2.430113e-11 |

The final CuPy screen checked all `17,563,400` outage/monitored-line sides and
found no violation above `1e-5` p.u. A separate independent raw-input checker,
run after the measured one-shot boundary because the frozen v15 controller
stopped on the definitive bound-insufficiency gate, also passed all
`17,563,400` sides. Its maximum security violation was `4.355e-9` p.u. and its
maximum model residual was `1.031e-7` p.u.

The final LP solution has 53 fractional commitment variables among 432
source-online generators. It is a lower-bound certificate, not an implementable
unit commitment or a pricing run.

## Timing and memory

- end-to-end measured wall time: `129.385188561` seconds
- cumulative cuOpt PDLP time: `126.871869842` seconds
- cumulative exhaustive GPU screening: `1.177935993` seconds
- peak process RSS: `1,321,668,608` bytes
- peak CUDA device-memory delta: `869,404,672` bytes
- peak CuPy pool use: `70,301,696` bytes

## Frozen evidence

- compute commit/tag: `50ff750ff5e320a14b556fedc77bbf3d7dbe66f6` /
  `experiment-2000-gpu-lp-certificate-v15`
- result JSON SHA-256:
  `80032f42288f80725632e837d434bf6fc2c7a993db193aac63758f2bf971c7cf`
- post-run independent verification SHA-256:
  `a660a4109b64d0eaf83a268d9d045ea03378fdb246b9776039a44431405645b5`

The complete compact evidence and all remaining artifact hashes are in
[`evidence.json`](evidence.json). Raw cases, full solver results, and native
logs remain ignored by repository policy.
