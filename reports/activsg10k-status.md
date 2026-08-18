# ACTIVSg10k one-shot status

## Outcome

The one authorized laptop CPU benchmark ran once and stopped after 220.824
seconds with `failed_exception`. The DGX Spark benchmark was not started because
the required laptop `optimal_verified` gate did not pass. No retry, tuning run,
or replacement benchmark was performed.

The frozen identity is commit
`e957c09027a2f17eafcabd957272f3c4e6afcea6`, tag `benchmark-10k-v1`, and
configuration SHA-256
`b73964691b3829aefd484bfa68e285197cee3e7f498f22dc810d94b0609e3bf1`.
The earlier ACTIVSg500 tag and results were not changed.

## Source and model evidence

- The immutable MATPOWER 8.1 source contains 10,000 buses, 2,485 generators,
  and 12,706 branches.
- Exactly 1,937 source-online generators were eligible; 548 source-offline rows
  remained unavailable. Exact source PMIN values were retained row by row. The
  eligible PMIN sum was 85,764.93 MW and PMAX sum was 170,021.33 MW.
- The source contingency table contained 11,806 branch-outage changes. The
  registered logic included 8,371 non-islanding outages and enumerated 3,435
  excluded islanding bridges. There were no generator-outage rows in this
  table.
- The base MILP had 35,840 columns, 33,913 rows, and 97,141 nonzeros.
- The FP64 LODF occupied 850,895,408 bytes. Three deterministic columns matched
  explicit post-outage DC solves to `2.78e-13` p.u.

## Last complete optimization evidence

Round 1 returned HiGHS `Optimal` in 59.081 seconds at objective
2,201,957.040411, bound 2,201,955.195613, and relative gap `8.38e-7`. Its
incumbent committed 1,520 generators. The NumPy screen then exhaustively checked
171,488,782 monitored sides in 2.031 seconds, found 322 violated pairs, and
added all of them in deterministic pair-ID order. The largest violation was
8.287 p.u., so this round was not a secure final solution.

Round 2 started with about 149.7 seconds of solver budget. At the end of that
budget, HiGHS returned `HighsStatus.kWarning`. The
[HiGHS enum documentation](https://ergo-code.github.io/HiGHS/previews/PR1223/python/enums/)
defines `kWarning` to include termination on a time or iteration limit. The exact timing makes a
round-2 time limit the evidence-supported inference, but the frozen adapter
raised immediately instead of reading the model status. Consequently, the
round-2 incumbent, bound, and gap were not preserved. Peak memory and final
independent verification are also unavailable.

## Post-run correction and boundary

After preserving the failed v1 evidence, the adapter was corrected to accept a
HiGHS warning long enough to serialize the model status, incumbent, bound, and
gap. Checkpoints now also record the active round/stage and sampled peak memory.
This correction is after the frozen tag and was validated only with tiny
fixtures. It does not alter, replace, or retroactively pass the official v1
result.

A second laptop run requires an explicitly approved new benchmark identity and
configuration. A Spark run remains disallowed until a matching laptop result
finishes within 300 seconds as `optimal_verified`.
