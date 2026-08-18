# ACTIVSg2000 DGX Spark 1e-3 v2 failure

The single authorized v2 run ended `failed_exception` after 6.980 seconds. It
is preserved as failed evidence and was not retried under the v2 identity.

cuOpt completed restricted-master round 1 with native `FeasibleFound`:

| Quantity | Value |
|---|---:|
| Objective | 1,118,442.3846114445 |
| Dual bound | 1,118,297.1059733980 |
| Reported relative gap | 1.298937165149922e-4 |
| Requested relative gap | 1e-3 |
| cuOpt solve wall time | 5.65025721899292 s |

The new certificate record contained NumPy boolean scalar values. The first
checkpoint after the solve therefore raised `TypeError: Object of type bool is
not JSON serializable`. The failure happened before the first exhaustive
contingency screen. No security pair was added, no commitment or dispatch was
serialized, independent verification was not reached, and no GPU prices exist.
The round-1 objective, bound, and gap are diagnostic solver evidence only; they
are not an accepted grid solution.

The correction converts every certificate flag to a built-in Python `bool` and
adds a direct `json.dumps` regression test over the exact certificate structure.
The separately authorized v3 identity retains the same raw hashes, model,
tolerances, deadline, and acceptance rule.

## Frozen identity

- Commit: `de2ff0eaed30a87a8dd20edfbec4e31086879e3e`
- Tag: `experiment-2000-gpu-gap-v2`
- Config SHA-256: `59b2c0412dc1723ee712bb2ef01b71460f36978bc3751dcd47a240f855925938`
- Started: `2026-08-14T00:31:18.966117+00:00`
- Finished: `2026-08-14T00:31:25.953156+00:00`

## Ignored raw evidence

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Result | 714620 | `898dbb5380d8b939d21767bfd9d4c15ead01867325723b04ec90a30656b90c5d` |
| One-shot registry | 936 | `d9e7d36594f10f119af1e5624b909e998e3986073aa56007c515ae24be3d93ef` |
| Last checkpoint | 713877 | `353b0884efb7f9def67b7be8ca5c7536a37f74b8e270375dcc22acd79af3c2d9` |
| Diagnostic events | 2377 | `eaa9e418b0b6c8aaae7724af7d3635461d415764e4a729c1c62f2b3828464fb9` |

The raw files remain ignored and local to the laptop and Spark results trees.
