# The cloud CV uses the 2015-2023 iteration window and refuses confirmation years

Status: accepted (2026-11-12)

The cloud config trained and evaluated through 2024 (`end_year: 2024`), while local training stopped at 2023. Release years 2024-2025 are the confirmation set. The v9 feature contract was adopted on the 2015-2023 window, so a cloud run that scores 2024 turns the confirmation set into a validation set and spends it for good. The cloud CV now defaults to the iteration window (`CrossValidationWindow` in `box_office/config.py`). `refuse_confirmation_years` in `box_office/eval_windows.py` raises `ConfirmationSetTouched` when a window ends in 2024 or later, unless the caller passes `allow_confirmation_years=True` for a deliberate holdout-burning run.

## Consequences

- The promotion gate keeps its threshold (pooled out-of-fold R² of at least 0.55 in dollar space, `promotion_threshold` in `box_office/config.py`). It now measures 2015-2023 only, where it used to include 2024. Its value is not directly comparable with older cloud runs.
- The cloud split keeps 2024-2025 out of the training frame. A model trained for production therefore does not see the two most recent years. Training on them is a holdout-burning run, and it cannot be undone.
- Spending the confirmation set is one-way. That is why the refusal sits in code and not in a doc.
