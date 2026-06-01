"""Feature-selection study for box-office-oracle.

Reproducible justification for
``box_office.ml.feature_pipeline.constants.SELECTED_FEATURES``.

It reduces the full engineered feature set (66 columns) to a compact, decorrelated,
axis-balanced subset via importance-ranked greedy selection with a correlation
ceiling, validated against the production scoring setup: XGBoost on
``log1p(worldwide_gross)``, RMSE / R**2 on the log scale, repeated K-fold CV.

Leakage controls (the offline snapshot predates the v2 leakage fix):
  - drop ``social_media_buzz`` + its derivatives (synthesized from the target),
  - drop the few rows carrying the ``production_budget = 0.4 * worldwide_gross``
    imputation signature,
  - restrict candidates to features the *current* pipeline still emits, excluding
    ``InteractionTransformer`` outputs (removed as redundant in this change).

Run:  uv run python analysis/feature_selection_study.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from box_office.ml.feature_preprocessor import FeaturePreprocessorHigh

DATA_DIR = "analysis/datasets_high"
CORR_CEILING = 0.70  # max |Spearman| allowed between any two selected features
MAX_FEATURES = 15  # hard cap
R2_TOLERANCE = 0.01  # accept the smallest set within this CV R**2 of the full model
CV_SEEDS = (42, 7, 123)

# Synthesized from the target pre-v2; scientifically invalid.
LEAKED = {
    "social_media_buzz",
    "viral_potential",
    "social_buzz_to_budget",
    "buzz_to_votes_ratio",
    "marketing_efficiency",
}
# InteractionTransformer outputs — removed in this change, so not selectable.
INTERACTION_OUTPUTS = {
    "franchise_strength",
    "franchise_budget_confidence",
    "blockbuster_budget_multiplier",
    "action_budget_interaction",
    "comedy_budget_efficiency",
    "horror_low_budget_advantage",
    "director_budget_confidence",
    "star_power_premium",
    "budget_seasonal_boost",
    "summer_budget_interaction",
    "holiday_budget_interaction",
    "weekend_rating_boost",
    "covid_budget_impact",
    "covid_rating_impact",
    "covid_votes_impact",
}
# Derived re-expressions that are near-duplicates of raw signals (votes, budget,
# year). Demoted in ranking so the interpretable raw feature wins its cluster.
DERIVED_DEMOTE = {
    "total_budget",
    "budget_to_votes_ratio",
    "votes_per_budget",
    "rating_per_budget",
    "rating_votes_interaction",
    "year_to_budget_ratio",
    "year_to_votes_ratio",
    "budget_inflation_adjusted",
    "votes_era_adjusted",
    "budget_per_actor_freq",
    "years_since_2000",
}

# Coarse axis map for coverage reporting (lowercase CSV names).
AXIS = {
    "demand": {"votes"},
    "budget": {
        "production_budget",
        "ad_budget",
        "total_budget",
        "budget_inflation_adjusted",
    },
    "marketing": {"ad_to_prod_ratio"},
    "ip": {"franchise_rating"},
    "quality": {"rating", "mpaa_encoded"},
    "format": {"runtime"},
    "industry_power": {
        "director_freq",
        "company_freq",
        "avg_actor_freq",
        "lead_actor_freq",
        "max_actor_freq",
    },
    "genre": {
        "genre_action",
        "genre_comedy",
        "genre_drama",
        "genre_adventure",
        "genre_thriller",
        "genre_horror",
        "genre_romance",
        "super_genre_encoded",
    },
    "era_time": {
        "release_year",
        "years_since_2000",
        "is_covid_era",
        "is_pre_streaming_era",
        "is_streaming_mature_era",
    },
}


def make_model() -> XGBRegressor:
    # n_jobs=1: single-threaded XGBoost is deterministic run-to-run, so the
    # selected set is reproducible (multithreaded `hist` is not).
    return XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=42,
        n_jobs=1,
        objective="reg:squarederror",
    )


def cv_r2(X: pd.DataFrame, y: pd.Series, cols: list[str]) -> float:
    scores = []
    for seed in CV_SEEDS:
        kf = KFold(5, shuffle=True, random_state=seed)
        oof = cross_val_predict(make_model(), X[cols], y, cv=kf, n_jobs=-1)
        scores.append(r2_score(y, oof))
    return float(np.mean(scores))


def cv_rmse(X: pd.DataFrame, y: pd.Series, cols: list[str]) -> float:
    kf = KFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(make_model(), X[cols], y, cv=kf, n_jobs=-1)
    return float(np.sqrt(mean_squared_error(y, oof)))


def max_pairwise_corr(X: pd.DataFrame, cols: list[str]) -> float:
    if len(cols) < 2:
        return 0.0
    cc = X[cols].corr(method="spearman").abs()
    np.fill_diagonal(cc.values, 0.0)
    return float(cc.max().max())


def greedy_decorrelated(
    X: pd.DataFrame, y: pd.Series, candidates: list[str]
) -> list[str]:
    """Importance-ranked greedy selection with a correlation ceiling.

    Ranks by (canonical-tier, gain): raw/simple features outrank derived
    re-expressions, so the interpretable representative wins each correlated
    cluster while the selection stays importance- and correlation-driven.
    """
    gain = make_model().fit(X[candidates], y).feature_importances_
    rank = pd.DataFrame({"feat": candidates, "gain": gain})
    rank["tier"] = rank["feat"].apply(lambda f: 1 if f.lower() in DERIVED_DEMOTE else 0)
    rank = rank.sort_values(["tier", "gain"], ascending=[True, False])
    corr = X[candidates].corr(method="spearman").abs()
    ordered: list[str] = []
    for feat in rank["feat"]:
        if all(corr.loc[feat, s] < CORR_CEILING for s in ordered):
            ordered.append(feat)
    return ordered


def main() -> None:
    emitted = FeaturePreprocessorHigh().get_feature_names()
    emitted_by_lower = {e.lower(): e for e in emitted}

    X = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    y_raw = pd.read_csv(f"{DATA_DIR}/y_train.csv").iloc[:, 0]
    y = np.log1p(y_raw)

    # Drop rows with the production_budget = 0.4 * worldwide_gross imputation signature.
    imputed = (X["production_budget"] / y_raw - 0.4).abs() < 1e-3
    X, y, y_raw = (
        X[~imputed].reset_index(drop=True),
        y[~imputed].reset_index(drop=True),
        y_raw[~imputed].reset_index(drop=True),
    )

    present = [c for c in X.columns if c.lower() in emitted_by_lower]
    full = [c for c in present if c.lower() not in LEAKED]  # incl. interactions
    candidates = [
        c for c in full if c.lower() not in INTERACTION_OUTPUTS
    ]  # selectable pool

    full_r2 = cv_r2(X, y, full)
    target = full_r2 - R2_TOLERANCE

    ordered = greedy_decorrelated(X, y, candidates)
    ordered = ordered[:MAX_FEATURES]

    # Smallest prefix within tolerance of the full model; else the cap.
    chosen = ordered
    table = []
    for k in range(3, len(ordered) + 1):
        r2k = cv_r2(X, y, ordered[:k])
        table.append((k, r2k))
        if r2k >= target and chosen is ordered:
            chosen = ordered[:k]

    print(f"dropped {int(imputed.sum())} imputed rows; n={len(y)}")
    print(
        f"full model: {len(full)} feats  CV_R2={full_r2:.4f}  (target ≥ {target:.4f})"
    )
    print(f"candidate pool (no leakage, no interactions): {len(candidates)}")
    print("\nprefix sweep (greedy, decorrelated):")
    for k, r2k in table:
        flag = "  <- chosen" if k == len(chosen) else ""
        print(f"  k={k:2d}  CV_R2={r2k:.4f}{flag}")

    sel_r2 = cv_r2(X, y, chosen)
    sel_rmse = cv_rmse(X, y, chosen)
    sel_maxr = max_pairwise_corr(X, chosen)
    no_budget = [c for c in chosen if c not in AXIS["budget"]]
    print(
        f"\nselected n={len(chosen)}  CV_R2={sel_r2:.4f}  CV_RMSE={sel_rmse:.4f}  max|r|={sel_maxr:.2f}"
    )
    print(f"  vs full CV_R2={full_r2:.4f}  (Δ={sel_r2 - full_r2:+.4f})")
    print(
        f"  CV_R2 without budget axis (leak sensitivity): {cv_r2(X, y, no_budget):.4f}"
    )

    covered = {ax for ax, feats in AXIS.items() if feats & {c.lower() for c in chosen}}
    print(f"  axis coverage: {sorted(covered)}")
    missing = set(AXIS) - covered
    if missing:
        print(f"  axes NOT covered: {sorted(missing)}")

    selected_emitted = [emitted_by_lower[c.lower()] for c in chosen]
    assert all(
        c in emitted for c in selected_emitted
    ), "selected feature not emitted by pipeline"
    print("\nSELECTED_FEATURES (production casing, paste into constants.py):")
    print("SELECTED_FEATURES: tuple[str, ...] = (")
    for name in selected_emitted:
        print(f'    "{name}",')
    print(")")


if __name__ == "__main__":
    main()
