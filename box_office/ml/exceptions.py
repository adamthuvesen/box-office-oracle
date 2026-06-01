"""Typed exceptions raised by the ``box_office.ml`` package."""

from __future__ import annotations


class CrossValidationFailed(RuntimeError):
    """Raised when cross-validation produced zero successful folds.

    The most recent fold-level exception is preserved as ``__cause__`` so the
    original traceback is reachable from the caller.
    """


class OOFIndexCollision(AssertionError):
    """Raised when the OOF accumulator sees duplicate ``(fold, idx)`` pairs.

    Duplicate fold/index pairs would corrupt out-of-fold metrics, so insertion
    asserts uniqueness before storing predictions.
    """
