# Environment contract

## Laptop CPU

The registered laptop environment is Python 3.13 with the exact packages in
`requirements-cpu.lock`. Development-only tools are pinned separately in
`requirements-dev.lock`. The laptop profile uses HiGHS and NumPy/SciPy. It does
not initialize or use the laptop GPU.

## DGX Spark

The registered Spark environment is built from the ARM64 image
`nvidia/cuopt:26.6.0-cuda13.2-py3.14`, pinned by digest in `Dockerfile.spark`.
That image currently provides Python 3.14.5, cuOpt 26.06.00, CuPy 14.1.1,
NumPy 2.4.6, and SciPy 1.17.1. The image digest—not the mutable tag—is the
benchmark identity.

The derived image also installs the ARM64 `highspy==1.15.1` wheel. cuOpt and
CuPy remain the registered Spark MIP and screening path. HiGHS is used only for
the post-MIP fixed-commitment pricing LP because the cuOpt adapter does not
expose the nodal-balance row duals required for prices. The environment checker
fails closed if that HiGHS version differs.

The repository is mounted read-only at `/workspace`; only its ignored
`results/` directory is mounted read-write. Raw TAMU files are copied separately
to the approved local Spark checkout and are never included in the image or Git
history.

The derived image installs `git` so the controller can bind every result to the
mounted checkout commit. Runtime scripts place CuPy and CUDA caches only under
the ignored, writable `results/` mount; the repository and raw inputs remain
read-only inside the container.

cuOpt's mixed-integer solver uses both GPU and CPU components. Consequently,
the comparison is a laptop-CPU system versus a DGX-Spark cuOpt/CuPy system,
not a claim of pure GPU kernel speedup.
