# ACTIVSg2000 GPU Lagrangian v4 result

## Outcome

The one authorized DGX Spark v4 run stopped fail-closed after 14.864 seconds.
cuOpt returned the continuous root LP as `Optimal`, but the independently
evaluated canonical model residual was `2.110661e-6` p.u., above the registered
`1e-6` p.u. tolerance. The controller correctly refused to screen or use that
vector.

This is a numerical acceptance failure, not an infeasible SCOPF result. The run
did not reach a contingency screen, an integer commitment candidate, a global
gap calculation, pricing, or the two-context child frontier. Therefore no
objective, lower bound, commitment, dispatch, price, or GPU speedup result is
claimed, and the run was not retried.

## Numerical attribution

The root LP had 9,995 rows, 4,014 continuous columns, and 2,369,600 nonzeros.
PDLP took 9.730 seconds and reported a primal objective of
`$1,118,296.7052230693` and a dual objective of `$1,118,296.700186604`.
Its scaled native log reported absolute primal infeasibility `3.16e-9` and
relative primal infeasibility `3.92e-11`, while the canonical checker found
`2.110661e-6` p.u. (`0.0002110661 MW` on the 100-MVA base).

The v4 `power_system_equilibrated_v2` reformulation could scale an already
per-unit row downward. Although that positive diagonal transformation is
mathematically exact, it can make a native absolute residual tolerance map to
a weaker residual tolerance in canonical units. v4 did not record the worst
canonical row identity, so the frozen evidence cannot attribute the residual
more narrowly without rerunning it.

## Replacement correction

The prepared v5 correction preserves the model and exact TAMU PMIN/PMAX values.
It uses a scale-safe equilibration that may strengthen an under-scaled row but
never reduces its v1 per-unit row multiplier. If an `Optimal` root vector still
misses the canonical residual gate, the controller performs one same-master,
primal-and-dual warm-started PDLP refinement at `1e-10` optimality tolerance
inside the same constraint-generation round. Each solve now records the worst
canonical row name, side, activity, bound, violation, and native row scale.

## Evidence identity

The failed run remains frozen at commit
`57e9ebfd5afd29fbb10cc50927b4ccf1d0707ae6`, tag
`experiment-2000-gpu-lagrangian-v4`, config SHA-256
`f7489c311d01608de172dc4a3ab57906ea1a18a3631dbf1a19ffa665618399ad`,
and image SHA-256
`5e79445c9fd8a103896c3fa20637f3d14e5ca5b1e64b9a612988e12550219c99`.
All artifact hashes and exact metrics are in [evidence.json](evidence.json).
