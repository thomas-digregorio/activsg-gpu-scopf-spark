"""Domain-specific failures with stable messages for CLI results."""


class ScopfError(RuntimeError):
    """Base failure for invalid inputs or an incomplete run."""


class ScopeViolation(ScopfError):
    """The request escaped the explicitly approved ACTIVSg case scope."""


class ProvenanceError(ScopfError):
    """An immutable source does not match its registered identity."""


class DeadlineExceeded(ScopfError):
    """The global end-to-end deadline no longer permits more work."""


class MipStartSolveError(ScopfError):
    """A native solver rejected or errored while processing a supplied MIP start."""
