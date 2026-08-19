# ACTIVSg2000 GPU Lagrangian v33

## Outcome

V33 is a preserved failed experiment. It is not a successful 0.1% solve and
must not replace a certified benchmark result.

- Frozen commit: `a761552bd5c7f5bca5393e394f354e16d65e0131`
- Frozen tag: `experiment-2000-gpu-lagrangian-v33`
- Config SHA-256: `cfe981ab0cac573c75d918c649aa6a641e606617f909d50a147955e8838ba513`
- Image ID: `sha256:640aeb14feb7ee174ef9706676d10ab3f020b97d58c84981c91a4f402a542674`
- Status: `failed_exception`
- Wall time: `574.2317364840128` seconds
- Error: `Hard at-least cardinality region is empty`

No V34 or replacement run was launched after the user directed the work to
stop.

## What the numerical fix established

The v32 failed-state preflight reproduced six asymmetric epigraph bounds below
the registered `1e-4` proposal floor. V33 snapped those proposal-only bounds to
zero, solved the exact first sibling pair, and replayed both child certificates
with zero GPU/host difference.

The official run then completed 144 block-diagonal sibling batches (288 child
regions) without a numerical solver warning, child failure, or certificate
replay failure:

- 1,662 post-normalization proposal bounds were snapped across the children.
- Minimum retained matrix coefficient: `1.00000270654084e-4`.
- Minimum retained nonzero finite bound: `1.01438026296768e-4`.
- Maximum finite bound: `1.0`.
- Maximum exact GPU/host certificate replay difference: `4.65661287307739e-10`
  dollars.
- Native large-coefficient-range advisories: `0`.
- Native numerical warnings: `0`.

This closes the v32 scaling-validation failure. The v33 stop was a separate
branch-state logic error.

## Primal and bound state at failure

The secure incumbent was independently verified:

- Objective: `$1,142,133.305099282`
- Committed source-online generators: `331`
- Commitment SHA-256:
  `ca2bedaf17974833d9725adab18773c10ed9d322e6b42c5ce87079cba083c6b0`
- Maximum model residual: `2.46018316829577e-12` p.u.
- Maximum exhaustive N-1 violation: `4.67490141176086e-6` p.u., below the
  `1e-5` security tolerance.
- Pricing status: `gpu_pdlp_fixed_commitment_exact_cost_epigraph_dual_certified`.

The proof state was incomplete:

- In-memory frontier bound: `$1,130,041.1601321911`
- Incumbent-relative gap: `0.010587332418293917`
- Frontier regions: `145`
- Completed disjunctive splits / GPU sibling batches: `144`
- Bound status: `gpu_generated_pending_independent_replay`
- Requested `0.001` gap: not met
- Mandatory final raw-input frontier replay: not reached
- Final success gate: failed

The in-memory bound and gap are diagnostic only because the run stopped before
the mandatory final raw-input replay.

## Timing evidence

- Bounded alternative-primal beam: 31 unique attempts, 123.354347235 seconds,
  zero incumbent improvements.
- 144 proof batches: 125.073325885 seconds summed GPU solver wall time and
  210.516187934 seconds summed build/solve/replay batch wall time.
- Proof batch wall-time range: 1.307475722 to 3.211594140 seconds.
- Peak process RSS: 3,515,609,088 bytes.
- Peak CuPy pool use: 70,301,696 bytes.
- CUDA device-memory delta: 3,098,951,680 bytes.

The checkpoint grew to about 113 MiB as the frontier expanded, so evidence
construction and serialization also became material. This run did not reach a
valid system-to-system timing comparison despite ending before the CPU
baseline, because it did not certify the requested gap or complete final
verification.

## Root cause and unimplemented correction

The refinement controller excludes supports already occupied by hard
cardinality cuts, but it chooses a new subset using the inherited root
fractional commitment without first applying the candidate region's binary
`fixed_on` and `fixed_off` masks. Deep in the frontier, a selected at-least
branch can therefore require more online units than remain free in that
region. The exact evaluator correctly detects the empty child, but the
controller currently treats that expected branch-state condition as a fatal
`ScopfError`.

A future revision should form the branching reference after forcing every
fixed commitment to its mask value, require both proposed children to be
nonempty under the masks, and handle an independently proven empty child as a
prune rather than a solver failure. That correction was intentionally not
implemented or run after the user requested a clean stop.

## Artifact hashes

- Result:
  `21d3a800e52e4956036838aaeddbfbf9e553cb337f5f862ca9c1bffeecd08645`
- Checkpoint:
  `86a9ed170354c99930a6f452acb8ee5dce011bd1bd59278ae28b079877abe80c`
- Worker console:
  `d080ac10995ee565047dfb584643bda2903627be7d8d343f17aeefbf6bac2a6e`
- Launcher log:
  `2f620be74f5283dc12b4d1b5fe6ebd0456d182f4044ab0c02415be0632ac8078`
- Run registry:
  `2c073c7cb11b3f582af1a7e68cc59172767fd6a79b84b01130fe9d01ac9de03c`

Large raw result, checkpoint, cache, and solver-log files remain ignored by
Git. The small `evidence.json` in this directory is the tracked summary.
