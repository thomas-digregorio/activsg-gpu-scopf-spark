# ACTIVSg500 GPU Lagrangian v6 failed run

The one authorized v6 DGX Spark run ended `failed_exception` after 0.754109
seconds. It loaded the immutable ACTIVSg500 inputs and built the reduced model,
then stopped before the root LP with:

```text
ScopfError: Prepared region master commitment bounds changed
```

There is no v6 objective, lower bound, commitment, dispatch, price vector,
security screen, or gap result. No cuOpt optimization was launched, no Phase-I
precheck ran, and the run cannot be used to evaluate the requested speedup.

## Cause

The new prepared-master validator iterated
`commitment_by_generator` as though its dictionary keys were canonical column
indices. They are immutable generator source-row indices. The tiny fixture's
only eligible generator happened to have both source row 0 and commitment
column 0, so it did not expose the identity error. ACTIVSg500 has noncontiguous
eligible source rows and commitment columns spaced by the PWL variables; for
example, its first eligible source rows map to commitment columns
`0, 12, 24, 36, ...`.

The correction resolves every source row through
`commitment_by_generator[source_row]`. A regression fixture now uses a nonzero
source row mapped to commitment column zero, and a read-only build of the real
ACTIVSg500 model validated all 56 mappings. This correction requires a new
frozen experiment identity; v6 was not retried or relabeled.

## Preserved identity

- Commit: `ee36e78f5f67b09d66493a07ef564c9bed62edf5`
- Tag: `experiment-500-gpu-lagrangian-v6`
- Config hash: `f369c75ccaaaf299cc0007336e78587912194c614238369970cc3cd45af24be3`
- Status: `failed_exception`
- Full launcher boundary: 0.754108831 seconds
- Raw loading: 0.003446278 seconds
- Network/reduced-model build: 0.072635017 seconds
- Root LP calls: 0
- Phase-I prechecks: 0
- Full authorized v6 run count: 1

Frozen pre-run evidence is in [preflight.json](preflight.json), and hashes of
the ignored result, registry, checkpoint, and logs are in
[evidence.json](evidence.json).
