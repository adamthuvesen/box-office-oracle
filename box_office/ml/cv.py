"""Time-series cross-validation and OOF evaluation."""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from box_office.config import MODEL_HYPERPARAMETERS
from box_office.eval_windows import (
    ITERATION_CV_FOLDS,
    ITERATION_END_YEAR,
    START_EVAL_YEAR,
    refuse_confirmation_years,
)
from box_office.ml.exceptions import CrossValidationFailed, OOFIndexCollision
from box_office.ml.metrics_models import (
    CrossValidationResults,
    FoldResult,
    OofEvaluation,
    OofRecord,
)
from box_office.ml.regression_metrics import rmse_on_log_scale, spearman_rank_corr
from box_office.ml.target import from_log_target

logger = logging.getLogger(__name__)


@dataclass
class _CrossValidationState:
    oof_preds: np.ndarray
    oof_indices: list[int] = field(default_factory=list)
    oof_records: list[OofRecord] = field(default_factory=list)
    oof_seen_keys: set[tuple[int, int]] = field(default_factory=set)
    cv_scores: list[float] = field(default_factory=list)
    cv_rmsle_scores: list[float] = field(default_factory=list)
    best_iterations: list[int] = field(default_factory=list)
    fold_importances: list[Any] = field(default_factory=list)
    fold_results: list[FoldResult] = field(default_factory=list)

    @classmethod
    def for_row_count(cls, row_count: int) -> _CrossValidationState:
        return cls(oof_preds=np.zeros(row_count))

    def record_oof_predictions(
        self, fold_number: int, val_indices: Any, y_pred: np.ndarray
    ) -> None:
        self.oof_preds[val_indices] = y_pred
        self.oof_indices.extend(val_indices)

        for idx, pred in zip(val_indices, y_pred, strict=True):
            key = (fold_number, int(idx))
            if key in self.oof_seen_keys:
                raise OOFIndexCollision(f"Duplicate (fold, idx) pair detected: {key}")
            self.oof_seen_keys.add(key)
            self.oof_records.append(OofRecord(fold=fold_number, idx=int(idx), pred=float(pred)))

    def build_results(self, feature_names: list[str]) -> CrossValidationResults:
        return CrossValidationResults(
            mean_cv_mae=np.mean(self.cv_scores),
            std_cv_mae=np.std(self.cv_scores),
            mean_cv_rmsle=np.mean(self.cv_rmsle_scores),
            std_cv_rmsle=np.std(self.cv_rmsle_scores),
            cv_scores=self.cv_scores,
            cv_rmsle_scores=self.cv_rmsle_scores,
            mean_best_iteration=np.mean(self.best_iterations),
            oof_predictions={
                str(idx): pred
                for idx, pred in zip(
                    self.oof_indices, self.oof_preds[self.oof_indices], strict=True
                )
            },
            oof_records=self.oof_records,
            feature_importances=(
                np.mean(self.fold_importances, axis=0).tolist() if self.fold_importances else None
            ),
            feature_names=feature_names,
            fold_results=self.fold_results,
        )


class TimeSeriesCrossValidator:
    """Time series cross-validator for box office prediction models."""

    def __init__(
        self,
        cv_folds: int = ITERATION_CV_FOLDS,
        start_eval_year: int = START_EVAL_YEAR,
        end_year: int = ITERATION_END_YEAR,
        early_stopping_rounds: int = MODEL_HYPERPARAMETERS.early_stopping_rounds,
        *,
        allow_confirmation_years: bool = False,
    ):
        refuse_confirmation_years(end_year, allow_confirmation_years=allow_confirmation_years)
        self.cv_folds = cv_folds
        self.start_eval_year = start_eval_year
        self.end_year = end_year
        self.early_stopping_rounds = early_stopping_rounds

    def _log_cv_summary(
        self,
        fold_results: list[FoldResult],
        cv_scores: list[float],
        cv_rmsle_scores: list[float],
        best_iterations: list[int],
    ) -> None:
        """Log CV results as single summary instead of per-fold."""
        logger.info("Cross-validation completed: %d folds", len(fold_results))

        if cv_scores:
            mean_mae = np.mean(cv_scores)
            std_mae = np.std(cv_scores)
            logger.info(f"CV MAE: {mean_mae:.4f} (+/- {std_mae:.4f})")
            # Output metric for SageMaker metric extraction
            logger.info(f"CV Mean MAE: {mean_mae:.4f}")

        if cv_rmsle_scores:
            mean_rmsle = np.mean(cv_rmsle_scores)
            std_rmsle = np.std(cv_rmsle_scores)
            logger.info(f"CV RMSLE: {mean_rmsle:.4f} (+/- {std_rmsle:.4f})")
            # Output metric for SageMaker metric extraction
            logger.info(f"CV Mean RMSLE: {mean_rmsle:.4f}")

        if best_iterations:
            mean_iterations = np.mean(best_iterations)
            std_iterations = np.std(best_iterations)
            logger.info(f"Best iterations: {mean_iterations:.0f} (+/- {std_iterations:.0f})")

    def cross_validate(
        self,
        model_class,
        X_train,
        y_train_log,
        dates,
        preprocessor_factory=None,
        **model_kwargs,
    ) -> CrossValidationResults:
        """Perform time series cross-validation with forward chaining.

        When ``preprocessor_factory`` is given (a callable returning a fresh,
        unfitted preprocessor with fit_transform/transform), ``X_train`` is the
        RAW frame: each fold fits the preprocessor on train-years rows only and
        transforms both splits, so frequency encodings cannot see the eval
        year. When omitted, ``X_train`` is already engineered and behavior is
        unchanged.
        """
        logger.info("Starting time series cross-validation with RMSLE objective...")

        # Reset indices to ensure positional alignment with numpy arrays
        # This prevents corruption when DataFrames have non-RangeIndex
        X_train = X_train.reset_index(drop=True)
        y_train_log = y_train_log.reset_index(drop=True)
        dates = dates.reset_index(drop=True)

        sort_indices = dates.argsort()
        X_sorted = X_train.iloc[sort_indices]
        y_sorted = y_train_log.iloc[sort_indices]
        dates_sorted = dates.iloc[sort_indices]

        state = _CrossValidationState.for_row_count(len(X_train))
        # With a preprocessor_factory the raw columns are not the model's
        # features; each fold overwrites this with the engineered names.
        feature_names = X_train.columns.tolist()

        unique_years = sorted(dates_sorted.unique())
        logger.info(
            f"Available years in dataset: {unique_years[:5]}...{unique_years[-5:]} (showing first/last 5)"
        )
        eval_years = [
            year for year in range(self.start_eval_year, self.end_year + 1) if year in unique_years
        ]
        actual_folds = min(self.cv_folds, len(eval_years))

        logger.info(f"CV evaluation years: {eval_years}")
        logger.info(f"Performing CV on {actual_folds} folds from years {eval_years[:actual_folds]}")

        for i, eval_year in enumerate(eval_years[:actual_folds]):
            train_mask = dates_sorted < eval_year
            val_mask = dates_sorted == eval_year

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                logger.warning(f"Skipping year {eval_year} - insufficient data")
                continue

            X_fold_train, X_fold_val = X_sorted[train_mask], X_sorted[val_mask]
            y_fold_train, y_fold_val = y_sorted[train_mask], y_sorted[val_mask]

            try:
                if preprocessor_factory is not None:
                    fold_preprocessor = preprocessor_factory()
                    X_fold_train = fold_preprocessor.fit_transform(X_fold_train)
                    X_fold_val = fold_preprocessor.transform(X_fold_val)
                    feature_names = X_fold_train.columns.tolist()

                state.fold_results.append(
                    self._run_fold(
                        fold_number=i + 1,
                        eval_year=eval_year,
                        model_class=model_class,
                        X_fold_train=X_fold_train,
                        X_fold_val=X_fold_val,
                        y_fold_train=y_fold_train,
                        y_fold_val=y_fold_val,
                        model_kwargs=model_kwargs,
                        state=state,
                    )
                )
            except Exception as e:
                # A partial CV would score a different window than the one
                # asked for, so any fold failure fails the run. Only the fold
                # context is added here.
                raise CrossValidationFailed(
                    f"CV fold {i + 1} (eval_year={eval_year}) failed: {e}"
                ) from e

        self._log_cv_summary(
            state.fold_results,
            state.cv_scores,
            state.cv_rmsle_scores,
            state.best_iterations,
        )

        if not state.fold_results:
            raise CrossValidationFailed(
                "No CV folds were attempted; check eval-year range vs. dataset years."
            )
        return state.build_results(feature_names)

    def _run_fold(
        self,
        fold_number: int,
        eval_year: int,
        model_class: Any,
        X_fold_train: Any,
        X_fold_val: Any,
        y_fold_train: Any,
        y_fold_val: Any,
        model_kwargs: dict[str, Any],
        state: _CrossValidationState,
    ) -> FoldResult:
        model_kwargs_with_early_stopping = {
            **model_kwargs,
            "early_stopping_rounds": self.early_stopping_rounds,
        }

        fold_model = model_class(**model_kwargs_with_early_stopping)
        fold_model.fit(
            X_fold_train,
            y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False,
        )

        y_pred = fold_model.predict(X_fold_val)
        state.record_oof_predictions(fold_number, y_fold_val.index, y_pred)

        fold_mae = mean_absolute_error(y_fold_val, y_pred)
        fold_rmsle = rmse_on_log_scale(y_fold_val, y_pred)
        state.cv_scores.append(fold_mae)
        state.cv_rmsle_scores.append(fold_rmsle)

        # Log-space R² and rank correlation are the stable lenses: the model
        # minimizes squared error on log1p(target), so log-space R² matches the
        # objective, and Spearman captures ordering quality regardless of the
        # dollar-level calibration.
        fold_model_r2_log = float(r2_score(y_fold_val, y_pred))
        fold_model_spearman = spearman_rank_corr(y_fold_val.to_numpy(), y_pred)
        fold_model_r2_dollars, fold_model_median_ape = self._dollar_scale_metrics(
            y_fold_val, y_pred
        )

        fold_best_iteration = (
            fold_model.best_iteration
            if hasattr(fold_model, "best_iteration")
            else model_kwargs.get("n_estimators", MODEL_HYPERPARAMETERS.n_estimators)
        )
        state.best_iterations.append(fold_best_iteration)
        state.fold_importances.append(fold_model.feature_importances_)

        return FoldResult(
            fold_number=fold_number,
            eval_year=eval_year,
            mae_score=fold_mae,
            rmsle_score=fold_rmsle,
            model_r2_log=fold_model_r2_log,
            model_spearman=fold_model_spearman,
            model_r2_dollars=fold_model_r2_dollars,
            model_median_ape=fold_model_median_ape,
            best_iteration=fold_best_iteration,
            train_samples=len(X_fold_train),
            val_samples=len(X_fold_val),
        )

    @staticmethod
    def _dollar_scale_metrics(y_fold_val: Any, y_pred: np.ndarray) -> tuple[float, float]:
        # Inputs above log(~1e308) overflow expm1; emit NaN rather than break CV.
        with np.errstate(over="ignore", invalid="ignore"):
            y_true_dollars = from_log_target(y_fold_val.to_numpy())
            y_pred_dollars = from_log_target(y_pred)
        if not (np.all(np.isfinite(y_true_dollars)) and np.all(np.isfinite(y_pred_dollars))):
            return float("nan"), float("nan")

        r2_dollars = float(r2_score(y_true_dollars, y_pred_dollars))
        median_ape = float(
            np.median(np.abs(y_pred_dollars - y_true_dollars) / np.maximum(y_true_dollars, 1.0))
        )
        return r2_dollars, median_ape


class ModelEvaluator:
    @staticmethod
    def evaluate_oof_performance(cv_results: CrossValidationResults, y_train_log) -> OofEvaluation:
        """Evaluate out-of-fold predictions."""
        logger.info("Evaluating out-of-fold performance...")

        oof_preds_dict = cv_results.oof_predictions
        if not oof_preds_dict:
            raise ValueError("CV results carry no out-of-fold predictions")

        indices = [int(idx) for idx in oof_preds_dict]
        oof_preds_log = np.array([oof_preds_dict[str(idx)] for idx in indices])

        # Reset index to ensure positional alignment with CV results
        y_train_log = y_train_log.reset_index(drop=True)
        y_true_log = y_train_log.iloc[indices].values

        oof_preds = from_log_target(oof_preds_log)
        y_true = from_log_target(y_true_log)

        oof_r2 = r2_score(y_true, oof_preds)
        oof_mae = mean_absolute_error(y_true, oof_preds)
        oof_rmsle = rmse_on_log_scale(y_true_log, oof_preds_log)

        logger.info(f"OOF R²: {oof_r2:.4f}")
        logger.info(f"OOF MAE: ${oof_mae:,.0f}")
        logger.info(f"OOF RMSLE: {oof_rmsle:.4f}")

        # Output metrics for SageMaker metric extraction
        logger.info(f"OOF R² Score: {oof_r2:.4f}")
        logger.info(f"OOF MAE: {oof_mae:.2f}")
        logger.info(f"OOF RMSLE: {oof_rmsle:.4f}")

        return OofEvaluation(
            oof_r2=oof_r2,
            oof_mae=oof_mae,
            oof_rmsle=oof_rmsle,
            num_oof_samples=len(indices),
        )
