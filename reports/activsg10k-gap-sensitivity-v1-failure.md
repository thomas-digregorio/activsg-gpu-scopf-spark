# ACTIVSg10k gap-sensitivity v1 failure

The registered v1 `1e-3` attempt ended as `failed_exception` after 22.687
seconds. Round 1 was accepted by HiGHS at objective 2,202,010.103352, bound
2,201,934.864469, and gap `3.41683e-5`. Its exhaustive screen found 333
violated contingency pairs with maximum violation 8.287 p.u.

Round 2 appended those 333 rows and supplied the prior generator commitment as
a partial MIP start. HiGHS began the fixed-discrete feasibility LP, then returned
`kError` with `Highs::optimizeModel() error trying to find feasible solution`.
The adapter correctly preserved the failure rather than treating it as a model
status. This error did not prove that the round-2 restricted master was
infeasible; the unrestricted MIP search never began.

The result is not N-1 secure, was not independently verified, and has no pricing
result. It is closed to unapproved reruns. The user subsequently authorized a
v2 replacement campaign. V2 retains the partial-start attempt and rebuilds the
identical restricted master cold only when this specific internal error occurs.

The frozen v1 identity is commit
`dce44dee1b6c0da34f1e09478de61bf17147ba2f`, tag
`experiment-10k-gap-v1`, and configuration SHA-256
`fefff168bc773519dcba5b6b55db9350c14c3592b50a6090ad657b0ca803421c`.
Raw ignored evidence hashes are retained in the accompanying JSON report.
