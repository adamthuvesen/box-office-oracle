"""The evaluation year windows. Every training and evaluation path reads them here.

Tuning and model selection use eval years ``START_EVAL_YEAR`` to
``ITERATION_END_YEAR`` only. ``CONFIRMATION_YEARS`` is a spent confirmation set:
the v9 contract was adopted on the iteration window, so a CV run that scores
those years turns the confirmation set into a validation set. Only an explicit
holdout-burning run may select them.

This module imports nothing from the package, so ``box_office.config`` can read
it without a layer violation.
"""

from __future__ import annotations

START_EVAL_YEAR = 2015
ITERATION_END_YEAR = 2023
CONFIRMATION_YEARS: tuple[int, ...] = (2024, 2025)
CONFIRMATION_START_YEAR = CONFIRMATION_YEARS[0]
CONFIRMATION_END_YEAR = CONFIRMATION_YEARS[-1]

ITERATION_CV_FOLDS = ITERATION_END_YEAR - START_EVAL_YEAR + 1


class ConfirmationSetTouched(ValueError):
    """A tuning or CV path tried to score a confirmation-set year."""


def refuse_confirmation_years(end_year: int, *, allow_confirmation_years: bool = False) -> None:
    """Raise unless the eval window ends before the confirmation set, or the caller burns it."""
    if end_year >= CONFIRMATION_START_YEAR and not allow_confirmation_years:
        raise ConfirmationSetTouched(
            f"eval window ends in {end_year}, inside the spent confirmation set "
            f"{CONFIRMATION_START_YEAR}-{CONFIRMATION_END_YEAR}. Iterate on years "
            f"<= {ITERATION_END_YEAR}; only a deliberate holdout-burning run may "
            "pass allow_confirmation_years=True."
        )
