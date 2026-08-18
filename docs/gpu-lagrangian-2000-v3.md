# ACTIVSg2000 GPU Lagrangian v3 correction contract

The frozen v2 run stopped after the secure root relaxation because its first
333-unit binary candidate made contingency row `c0603_m0609_lower` constant and
violated after fixed dispatch substitution. That is a proof that this one
commitment cannot be feasible; it is not a failure of the full SCOPF model.

V3 changes only exception scope. The dispatch projection records the row name
and shifted bounds, rejects that fixed candidate before calling PDLP, and
continues the deterministic GPU-generated candidate queue. The full-model
infeasibility flag remains false. Exact source PMIN/PMAX, ten-segment costs,
network equations, contingency rows, tolerances, 1,800-second runtime policy,
no-CPU-seed policy, and GPU Lagrangian lower-bound algorithm are unchanged.

The v2 result and tag remain immutable. A v3 run requires its own config, suite
paths, frozen tag, preflight, and explicit approval because it would be a new
one-shot run rather than a retry that overwrites v2.
