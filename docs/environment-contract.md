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

The repository is mounted read-only at `/workspace`; only its ignored
`results/` directory is mounted read-write. Raw TAMU files are copied separately
to the approved local Spark checkout and are never included in the image or Git
history.

cuOpt's mixed-integer solver uses both GPU and CPU components. Consequently,
the comparison is a laptop-CPU system versus a DGX-Spark cuOpt/CuPy system,
not a claim of pure GPU kernel speedup.
