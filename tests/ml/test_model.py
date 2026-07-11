"""Tests for ML model components.

Bare imports on purpose: a broken model module must fail at collection
time, not silently skip.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from box_office.ml.cv import ModelEvaluator
from box_office.ml.model import BoxOfficeXGBoostModel


class TestBoxOfficeXGBoostModel(unittest.TestCase):
    """Test BoxOfficeXGBoostModel essential functionality."""

    def test_model_initialization(self):
        model = BoxOfficeXGBoostModel()
        self.assertFalse(model.is_fitted)
        self.assertIsNotNone(model.params)

    @patch("box_office.ml.model.xgb.XGBRegressor")
    def test_model_fit_basic(self, mock_xgb):
        """Test basic model fitting.

        Asserts the underlying XGBRegressor receives ``X_train, y_train`` in
        that order.
        """
        mock_model_instance = MagicMock()
        mock_xgb.return_value = mock_model_instance

        model = BoxOfficeXGBoostModel()
        X_train = pd.DataFrame({"feature1": [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])

        result = model.fit(X_train, y_train)

        self.assertTrue(model.is_fitted)
        # Validate the call shape, not just that it happened.
        assert mock_model_instance.fit.call_count == 1
        call_args, call_kwargs = mock_model_instance.fit.call_args
        # X must be the first positional argument and identical to X_train.
        pd.testing.assert_frame_equal(call_args[0], X_train)
        # y must be passed somewhere — positionally or as a keyword.
        if len(call_args) > 1:
            pd.testing.assert_series_equal(pd.Series(call_args[1]), y_train)
        else:
            assert "y" in call_kwargs
            pd.testing.assert_series_equal(pd.Series(call_kwargs["y"]), y_train)
        self.assertEqual(result, model)  # Should return self


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator utility class."""

    def test_evaluate_oof_performance_basic(self):
        cv_results = {"oof_predictions": {"0": 2.0, "1": 3.0, "2": 4.0}}

        y_train_log = pd.Series([1.8, 3.2, 3.9], index=[0, 1, 2])

        result = ModelEvaluator.evaluate_oof_performance(cv_results, y_train_log)

        self.assertIsInstance(result, dict)
        self.assertIn("oof_r2", result)
        self.assertIn("oof_mae", result)


if __name__ == "__main__":
    unittest.main()
