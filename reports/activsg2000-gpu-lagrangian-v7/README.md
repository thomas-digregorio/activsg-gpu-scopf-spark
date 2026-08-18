# ACTIVSg2000 GPU Lagrangian v7

This is the compact evidence index for the one-shot DGX Spark v7 run. The
55 MB raw result remains on the Spark and is intentionally excluded from Git.

## Frozen identity

- Commit: `436dbaa068eac240a0bd616c8ea403e6a86203df`
- Tag: `experiment-2000-gpu-lagrangian-v7`
- Config SHA-256: `ebc6474f3eab030a1696a63f3e64aa7407b9764144a11889f9adafe1aed90cf1`
- Raw result SHA-256: `2e148c985b41b19b377d196880776ed53623f1cb46535224c38599502173adac`
- Spark raw result: `/home/dgxsparktd/activsg-gpu-scopf-spark/results/experiments/activsg2000-gpu-lagrangian-v7-dgx-spark.json`

## Outcome

The run stopped cleanly with `deadline_budget_exhausted`. Total end-to-end
wall time was 924.836317954 seconds, below the laptop CPU comparison boundary
of 1007.3702709000063 seconds. This is **not** a speedup result because v7 did
not produce a secure primal, objective, or certified relative gap.

The independently replayed current-frontier lower bound was
`1128529.2124109394`, with zero replay difference. The run solved 41 regions,
left 21 frontier regions, and completed 42 child Phase-I checks.

## Attribution

The v6 numerical defects did not recur:

- zero native MIP-start rejections;
- zero native barrier warnings (the barrier path was disabled);
- no malformed Stable2-state resubmission exception;
- the root Lagrangian certificate replay passed.

The failed gate was primal scheduling. Twelve commitments consumed the global
repair cap and were rejected by exact fixed-commitment Phase I. The best had a
certified positive minimum violation of `0.0014754097912082827` p.u. Forty-five
later commitments were recorded as `skipped_repair_limit`, so no secure
incumbent existed when the solve budget ended.

The lower-bound frontier also remained weak: bounds ranged from
`1128529.2124109394` to `1142774.2474378317`, but the minimum did not improve
over the root bound. A later revision must address both candidate scheduling
and lower-bound refinement; v7 must not be rerun or relabeled as successful.

