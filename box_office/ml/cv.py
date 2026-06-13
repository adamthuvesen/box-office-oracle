"""Time-series cross-validation and OOF evaluation."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from box_office.ml.exceptions import CrossValidationFailed, OOFIndexCollision
from box_office.ml.regression_metrics import rmse_on_log_scale, spearman_rank_corr

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """Time series cross-validator for box office prediction models."""

    def __init__(
        self,
        cv_folds: int = 10,
        start_eval_year: int = 2015,
        end_year: int = 2024,
        early_stopping_rounds: int = 50,
    ):
        self.cv_folds = cv_folds
        self.start_eval_year = start_eval_year
        self.end_year = end_year
        self.early_stopping_rounds = early_stopping_rounds

    def _log_cv_summary(
        self,
        fold_results: List[Dict],
        cv_scores: List[float],
        cv_rmsle_scores: List[float],
        best_iterations: List[int],
    ) -> None:
        """Log CV results as single summary instead of per-fold."""
        successful_folds = [r for r in fold_results if r["error"] is None]
        failed_folds = [r for r in fold_results if r["error"] is not None]

        logger.info("Cross-validation completed!")
        logger.info(
            f"Completed {len(successful_folds)}/{len(fold_results)} folds successfully"
        )

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
            logger.info(
                f"Best iterations: {mean_iterations:.0f} (+/- {std_iterations:.0f})"
            )

        # Log individual fold failures with details
        if failed_folds:
            logger.error(f"Failed folds ({len(failed_folds)}):")
            for fold in failed_folds:
                logger.error(
                    f"  Fold {fold['fold_number']} (year {fold['eval_year']}): {fold['error']}"
                )

    def cross_validate(
        self, model_class, X_train, y_train_log, dates, **model_kwargs
    ) -> Dict[str, Any]:
        """Perform time series cross-validation with forward chaining."""
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

        oof_preds = np.zeros(len(X_train))
        oof_indices = []
        # (fold, idx, pred) records — a str(idx)-keyed dict would silently
        # overwrite when expanding-window CV revisits the same row.
        oof_records: List[Dict[str, Any]] = []
        oof_seen_keys: set = set()
        cv_scores = []
        cv_rmsle_scores = []
        best_iterations = []
        fold_importances = []
        fold_results = []
        last_fold_exception: Optional[BaseException] = None

        unique_years = sorted(dates_sorted.unique())
        logger.info(
            f"Available years in dataset: {unique_years[:5]}...{unique_years[-5:]} (showing first/last 5)"
        )
        eval_years = [
            year
            for year in range(self.start_eval_year, self.end_year + 1)
            if year in unique_years
        ]
        actual_folds = min(self.cv_folds, len(eval_years))

        logger.info(f"CV evaluation years: {eval_years}")
        logger.info(
            f"Performing CV on {actual_folds} folds from years {eval_years[:actual_folds]}"
        )

        for i, eval_year in enumerate(eval_years[:actual_folds]):
            train_mask = dates_sorted < eval_year
            val_mask = dates_sorted == eval_year

            if train_mask.sum() == 0 or val_mask.sum() == 0:
                logger.warning(f"Skipping year {eval_year} - insufficient data")
                continue

            X_fold_train, X_fold_val = X_sorted[train_mask], X_sorted[val_mask]
            y_fold_train, y_fold_val = y_sorted[train_mask], y_sorted[val_mask]

            try:
                model_kwargs_with_early_stopping = model_kwargs.copy()
                model_kwargs_with_early_stopping["early_stopping_rounds"] = (
                    self.early_stopping_rounds
                )

                fold_model = model_class(**model_kwargs_with_early_stopping)
                fold_model.fit(
                    X_fold_train,
                    y_fold_train,
                    eval_set=[(X_fold_val, y_fold_val)],
                    verbose=False,
                )

                y_pred = fold_model.predict(X_fold_val)
                val_indices = y_fold_val.index
                oof_preds[val_indices] = y_pred
                oof_indices.extend(val_indices)

                fold_number = i + 1
                for idx, pred in zip(val_indices, y_pred):
                    key = (fold_number, int(idx))
                    if key in oof_seen_keys:
                        raise OOFIndexCollision(
                            f"Duplicate (fold, idx) pair detected: {key}"
                        )
                    oof_seen_keys.add(key)
                    oof_records.append(
                        {
                            "fold": fold_number,
                            "idx": int(idx),
                            "pred": float(pred),
                        }
                    )

                fold_mae = mean_absolute_error(y_fold_val, y_pred)
                fold_rmsle = rmse_on_log_scale(y_fold_val, y_pred)
                cv_scores.append(fold_mae)
                cv_rmsle_scores.append(fold_rmsle)

                # Log-scale R² and rank correlation are the stable lenses: the
                # model minimizes squared error on log1p(target), so log-space
                # R² matches the objective, and Spearman captures ordering
                # quality regardless of the dollar-level calibration.
                fold_model_r2_log = float(r2_score(y_fold_val, y_pred))
                fold_model_spearman = spearman_rank_corr(y_fold_val.to_numpy(), y_pred)

                # Per-fold dollar-scale metrics so the report can also speak in
                # USD R² / APE. Inputs above log(~1e308) overflow expm1; emit
                # NaN rather than break CV.
                with np.errstate(over="ignore", invalid="ignore"):
                    y_true_dollars = np.expm1(y_fold_val.to_numpy())
                    y_pred_dollars = np.expm1(y_pred)
                if np.all(np.isfinite(y_true_dollars)) and np.all(
                    np.isfinite(y_pred_dollars)
                ):
                    fold_model_r2_dollars = float(
                        r2_score(y_true_dollars, y_pred_dollars)
                    )
                    fold_model_median_ape = float(
                        np.median(
                            np.abs(y_pred_dollars - y_true_dollars)
                            / np.maximum(y_true_dollars, 1.0)
                        )
                    )
                else:
                    fold_model_r2_dollars = float("nan")
                    fold_model_median_ape = float("nan")

                fold_best_iteration = (
                    fold_model.best_iteration
                    if hasattr(fold_model, "best_iteration")
                    else model_kwargs.get("n_estimators", 2000)
                )
                best_iterations.append(fold_best_iteration)
                fold_importances.append(fold_model.feature_importances_)

                fold_results.append(
                    {
                        "fold_number": i + 1,
                        "eval_year": eval_year,
                        "mae_score": fold_mae,
                        "rmsle_score": fold_rmsle,
                        "model_r2_log": fold_model_r2_log,
                        "model_spearman": fold_model_spearman,
                        "model_r2_dollars": fold_model_r2_dollars,
                        "model_median_ape": fold_model_median_ape,
                        "best_iteration": fold_best_iteration,
                        "train_samples": len(X_fold_train),
                        "val_samples": len(X_fold_val),
                        "error": None,
                    }
                )

            except Exception as e:
                # Capture the original exception chain so we can re-raise from
                # the last failure if every fold blows up. ``exc_info=True``
                # ensures the per-fold traceback ends up in CloudWatch even
                # though we deliberately keep going.
                last_fold_exception = e
                logger.error(
                    "CV fold %d (eval_year=%s) failed: %s",
                    i + 1,
                    eval_year,
                    e,
                    exc_info=True,
                )
                fold_results.append(
                    {
                        "fold_number": i + 1,
                        "eval_year": eval_year,
                        "mae_score": None,
                        "rmsle_score": None,
                        "best_iteration": None,
                        "train_samples": len(X_fold_train),
                        "val_samples": len(X_fold_val),
                        "error": str(e),
                    }
                )

        self._log_cv_summary(fold_results, cv_scores, cv_rmsle_scores, best_iterations)

        if cv_scores:
            mean_cv_mae = np.mean(cv_scores)
            std_cv_mae = np.std(cv_scores)
            mean_cv_rmsle = np.mean(cv_rmsle_scores)
            std_cv_rmsle = np.std(cv_rmsle_scores)

            cv_results = {
                "mean_cv_mae": mean_cv_mae,
                "std_cv_mae": std_cv_mae,
                "mean_cv_rmsle": mean_cv_rmsle,
                "std_cv_rmsle": std_cv_rmsle,
                "cv_scores": cv_scores,
                "cv_rmsle_scores": cv_rmsle_scores,
                "mean_best_iteration": (
                    np.mean(best_iterations)
                    if best_iterations
                    else model_kwargs.get("n_estimators", 2000)
                ),
                "oof_predictions": {
                    str(idx): pred
                    for idx, pred in zip(oof_indices, oof_preds[oof_indices])
                },
                "oof_records": oof_records,
                "feature_importances": (
                    np.mean(fold_importances, axis=0).tolist()
                    if fold_importances
                    else None
                ),
                "feature_names": X_train.columns.tolist(),
                "fold_results": fold_results,
            }

            return cv_results
        else:
            attempted = len(fold_results)
            failed = sum(1 for r in fold_results if r["error"] is not None)
            message = (
                f"All {attempted} CV folds failed (errors={failed}); see per-fold "
                "tracebacks logged above."
            )
            if last_fold_exception is not None:
                raise CrossValidationFailed(message) from last_fold_exception
            # No folds were even attempted — the caller passed years that didn't
            # intersect the dataset. Surface that clearly without a misleading chain.
            raise CrossValidationFailed(
                "No CV folds were attempted; check eval-year range vs. dataset years."
            )


class ModelEvaluator:

    @staticmethod
    def evaluate_oof_performance(
        cv_results: Dict[str, Any], y_train_log
    ) -> Dict[str, float]:
        """Evaluate out-of-fold predictions."""
        logger.info("Evaluating out-of-fold performance...")

        oof_preds_dict = cv_results["oof_predictions"]
        if not oof_preds_dict:
            logger.warning("No OOF predictions available")
            return {}

        indices = [int(idx) for idx in oof_preds_dict.keys()]
        oof_preds_log = np.array([oof_preds_dict[str(idx)] for idx in indices])

        # Reset index to ensure positional alignment with CV results
        y_train_log = y_train_log.reset_index(drop=True)
        y_true_log = y_train_log.iloc[indices].values

        # expm1 inverts log1p applied to the target during training.
        oof_preds = np.expm1(oof_preds_log)
        y_true = np.expm1(y_true_log)

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

        return {
            "oof_r2": oof_r2,
            "oof_mae": oof_mae,
            "oof_rmsle": oof_rmsle,
            "num_oof_samples": len(indices),
        }
