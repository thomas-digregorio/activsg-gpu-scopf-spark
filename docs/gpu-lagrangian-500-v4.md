# ACTIVSg500 GPU Lagrangian experiment v4

V4 preserves the v3 ACTIVSg500 raw inputs, exact source `PMIN`, ten-segment
source-derived costs, security/model tolerances, 600-second deadline, and
`1e-3` gap target. It is a separately frozen one-shot experiment; v1-v3
evidence and paths are immutable.

## Corrected security-row identity

Every screened contingency-side pair remains a logical constraint identity and
is serialized for provenance. If two pairs produce an exactly identical FP64
dispatch coefficient vector and right-hand side after the registered
coefficient cleanup, only the first deterministic representative is submitted
to cuOpt. The equivalence class and pair-to-representative map are serialized
and independently reconstructed. This is exact row coalescing, not
tolerance-based constraint removal. During cross-architecture replay only, the
recorded equivalence is accepted within `1e-12` after independently rebuilding
each alias; this does not change the solver model that was run.

The portable verifier accepts a serialized LODF only when it differs from the
fresh raw-case operator by at most `1e-12`. The solver row is reconstructed
using the serialized value after that check, allowing harmless cross-CPU FP64
differences while preserving the frozen constraint.

## Bounded child solves and transactional rollback

Each disjunctive child receives at most 15 seconds, in slices of at most five
seconds. A prior primal/dual vector warm-starts the first re-solve after rows
are appended. Divergence or stagnation triggers exactly one cold restart. A
still-unusable child cannot consume the global deadline indefinitely.

A failed child does not make its parent disappear. V4 next solves a GPU PDLP
Phase-I model that minimizes one common per-unit violation of all rows. Any
finite row dual is projected into the valid upper-row dual cone and evaluated
against finite variable bounds. A child is pruned only when the resulting
box-dual lower bound, after a `1e-8` p.u. safety deduction, exceeds `1e-6`
p.u. The complete dual vector, row identities, fixed-unit masks, and
contingency identities are checkpointed and replayed from raw inputs on CPU.
This is a numerical FP64 certificate, not an exact rational Farkas proof.

If Phase I cannot certify the child, the tentative split is rolled back and
the next deterministic generator is tried. A split enters the cover only when
both children are either solved with Lagrangian certificates or Phase-I
certified infeasible. The independent cover checker includes both active and
certified-pruned leaves.

## Durable secure incumbent

As soon as a fixed commitment passes its exhaustive GPU screen, V4 serializes
the complete generator commitment, MW and per-unit dispatch, base flows, bus
angles, and fixed-commitment PDLP prices. The raw-input verifier immediately
rechecks exact conditional `PMIN`/`PMAX`, balance, base limits, and every valid
branch outage before lower-bound refinement continues. A later timeout
therefore cannot erase the secure incumbent.

## Acceptance

Success still requires all of the following in the sole full v4 run:

- an immediately and finally independently verified secure dispatch;
- a disjoint exhaustive leaf cover with every active Lagrangian bound and every
  pruned Phase-I certificate independently replayed;
- a relative upper/lower-bound gap no larger than `1e-3`;
- no integer solver, CPU branch-and-bound, or unregistered model relaxation.

Development validation uses unit fixtures and one tiny DGX smoke only. The full
run is launched only from tag `experiment-500-gpu-lagrangian-v4` and refuses
pre-existing one-shot paths.
