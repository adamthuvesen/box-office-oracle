"""The model's target transform: ``log1p`` of worldwide gross in USD.

The model trains and predicts on the log scale. Every conversion between
dollars and the log target goes through this pair, so training, evaluation,
and serving cannot drift apart.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

type TargetValues = pd.Series | npt.NDArray[np.floating] | float


def to_log_target[T: TargetValues](gross_usd: T) -> T:
    """Worldwide gross in USD to the model's log target."""
    return np.log1p(gross_usd)


def from_log_target[T: TargetValues](log_target: T) -> T:
    """The model's log target back to worldwide gross in USD."""
    return np.expm1(log_target)
