class FunctionCallingError(Exception):
    """Base class for expected, user-facing errors in this project.

    Anything raised as this (or a subclass) is caught at the top level and
    reported as a clean error message instead of a stack trace, per the
    "never crash unexpectedly" requirement.
    """


class InputFileError(FunctionCallingError):
    """Raised when an input file is missing, unreadable, or invalid."""


class GenerationError(FunctionCallingError):
    """Raised when the constrained decoding pipeline cannot produce a call."""
