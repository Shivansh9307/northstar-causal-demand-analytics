"""
Demand forecasting models and time-based evaluation.

Model ladder, in the order PROJECT_ARCHITECTURE.md §6 Phase 5 asks for:

1. **Seasonal naive** - predict the same weekday from the last fully observed
   week. At a seven-day horizon that is both the most recent available
   observation and the matching day of week, which makes it the honest
   comparator rather than a straw man.
2. **Ridge** on the engineered features, with one-hot categoricals.
3. **Gradient boosting** - LightGBM where available, falling back to
   scikit-learn's histogram booster (the same algorithm family) so the pipeline
   runs on a machine without OpenMP.

Metrics
-------
MAPE is reported because §8 asks for it, but it is not the primary metric and
the report says why: daily store x SKU demand includes small counts, and MAPE
divides by them, so a one-unit miss on a two-unit day counts fifty times a
one-unit miss on a hundred-unit day. WAPE (sum of absolute errors over sum of
actuals) is the retail standard and is what the model comparison is judged on.
MAPE is computed on non-zero actuals only, which is stated wherever it appears.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOGGER = logging.getLogger("promopulse.ml.forecast")


def lightgbm_available() -> bool:
    """LightGBM needs libomp on macOS; report cleanly rather than crashing."""
    try:
        import lightgbm  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Forecast accuracy on a common scale, with MAPE's caveat made explicit."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    non_zero = actual > 0

    return {
        "wape": float(np.abs(error).sum() / max(actual.sum(), 1e-9)),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mape_nonzero": float((np.abs(error[non_zero]) / actual[non_zero]).mean()),
        "bias": float(error.mean()),
        "n": int(len(actual)),
        "zero_share": float((~non_zero).mean()),
    }


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

@dataclass
class GradientBooster:
    """
    Thin wrapper so the pipeline is identical whichever backend is present.

    LightGBM is preferred because it handles categoricals natively and exposes
    exact TreeSHAP through `pred_contrib=True` - which is how this project gets
    SHAP values without the `shap` package, whose `llvmlite` dependency does not
    build on Python 3.14.
    """

    categorical: Sequence[str]
    params: Dict[str, object] = field(default_factory=dict)
    backend: str = field(init=False)
    model: object = field(init=False, default=None)
    feature_names: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.backend = "lightgbm" if lightgbm_available() else "sklearn_hist"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_valid: Optional[pd.DataFrame] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> "GradientBooster":
        self.feature_names = list(X_train.columns)
        if self.backend == "lightgbm":
            import lightgbm as lgb

            params = {
                "objective": "tweedie",  # non-negative, right-skewed counts
                "tweedie_variance_power": 1.3,
                "learning_rate": 0.06,
                "num_leaves": 96,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "bagging_freq": 1,
                "verbose": -1,
                "num_threads": 0,
                **self.params,
            }
            train_set = lgb.Dataset(
                X_train, label=y_train, categorical_feature=list(self.categorical)
            )
            valid_sets = []
            callbacks = []
            if X_valid is not None:
                valid_sets = [lgb.Dataset(X_valid, label=y_valid, reference=train_set)]
                callbacks = [lgb.early_stopping(50, verbose=False)]
            self.model = lgb.train(
                params, train_set, num_boost_round=600,
                valid_sets=valid_sets, callbacks=callbacks,
            )
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            categorical_mask = [c in self.categorical for c in X_train.columns]
            self.model = HistGradientBoostingRegressor(
                loss="poisson",  # closest available to the count target
                learning_rate=0.06,
                max_iter=400,
                max_leaf_nodes=96,
                min_samples_leaf=200,
                categorical_features=categorical_mask,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            ).fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, None)

    def shap_values(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Exact TreeSHAP contributions, one column per feature.

        Only available on the LightGBM backend; returns None otherwise so the
        caller can substitute permutation importance and say so.
        """
        if self.backend != "lightgbm":
            return None
        contributions = self.model.predict(X, pred_contrib=True)
        # Last column is the base value.
        return pd.DataFrame(contributions[:, :-1], columns=self.feature_names, index=X.index)


def fit_ridge(
    X_train: pd.DataFrame, y_train: np.ndarray, categorical: Sequence[str]
):
    """Regularised linear baseline: one-hot categoricals, standardised numerics."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in X_train.columns if c not in categorical]
    pipeline = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), list(categorical)),
        ])),
        # sparse_cg keeps a 1.8M-row one-hot design tractable.
        ("model", Ridge(alpha=1.0, solver="sparse_cg")),
    ])
    return pipeline.fit(X_train, y_train)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def run_cross_validation(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    categorical: Sequence[str],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    target: str = "units_sold",
    max_train_rows: Optional[int] = 800_000,
    seed: int = 42,
    naive_column: str = "sales_lag_7",
) -> pd.DataFrame:
    """
    Expanding-window CV across the model ladder.

    `max_train_rows` caps the training sample *within* each chronological window,
    which keeps the sweep tractable without touching the time ordering - the
    sample is drawn from days already in that fold's past. Measured cost: holding
    out the last fold, capping at 600k moved WAPE from 0.3827 to 0.3839, so the
    model ranking is unaffected. The final holdout in `run_holdout` uses every
    available training row.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for fold_number, (train_index, valid_index) in enumerate(folds, start=1):
        if max_train_rows is not None and len(train_index) > max_train_rows:
            train_index = rng.choice(train_index, size=max_train_rows, replace=False)
        train = frame.iloc[train_index]
        valid = frame.iloc[valid_index]
        X_train, y_train = train[list(feature_names)], train[target].to_numpy()
        X_valid, y_valid = valid[list(feature_names)], valid[target].to_numpy()

        LOGGER.info("Fold %d: train %d, valid %d", fold_number, len(train), len(valid))

        naive = valid[naive_column].to_numpy()
        rows.append({"fold": fold_number, "model": "Seasonal naive", **evaluate(y_valid, naive)})

        ridge = fit_ridge(X_train, y_train, categorical)
        rows.append({
            "fold": fold_number, "model": "Ridge",
            **evaluate(y_valid, np.clip(ridge.predict(X_valid), 0, None)),
        })

        booster = GradientBooster(categorical=categorical).fit(X_train, y_train, X_valid, y_valid)
        rows.append({
            "fold": fold_number, "model": f"Gradient boosting ({booster.backend})",
            **evaluate(y_valid, booster.predict(X_valid)),
        })
    return pd.DataFrame(rows)


def run_holdout(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    categorical: Sequence[str],
    holdout_index: np.ndarray,
    target: str = "units_sold",
    naive_column: str = "sales_lag_7",
) -> Tuple[pd.DataFrame, GradientBooster, pd.DataFrame, np.ndarray]:
    """
    Final evaluation on the untouched holdout period.

    Returns the metric table, the fitted booster (for SHAP), the holdout feature
    frame and the booster's predictions.
    """
    mask = np.zeros(len(frame), dtype=bool)
    mask[holdout_index] = True
    train, holdout = frame[~mask], frame[mask]

    X_train, y_train = train[list(feature_names)], train[target].to_numpy()
    X_holdout, y_holdout = holdout[list(feature_names)], holdout[target].to_numpy()

    rows: List[Dict[str, object]] = []
    naive = holdout[naive_column].to_numpy()
    rows.append({"model": "Seasonal naive", **evaluate(y_holdout, naive)})

    ridge = fit_ridge(X_train, y_train, categorical)
    rows.append(
        {"model": "Ridge", **evaluate(y_holdout, np.clip(ridge.predict(X_holdout), 0, None))}
    )

    booster = GradientBooster(categorical=categorical).fit(X_train, y_train)
    predictions = booster.predict(X_holdout)
    rows.append(
        {
            "model": f"Gradient boosting ({booster.backend})",
            **evaluate(y_holdout, predictions),
        }
    )

    return pd.DataFrame(rows), booster, X_holdout, predictions


def error_by_segment(
    holdout: pd.DataFrame, predictions: np.ndarray, column: str, target: str = "units_sold"
) -> pd.DataFrame:
    """Where the forecast is weak, which is what a planner actually needs to know."""
    work = holdout.copy()
    work["prediction"] = predictions
    rows = []
    for value, group in work.groupby(column, observed=True):
        metrics = evaluate(group[target].to_numpy(), group["prediction"].to_numpy())
        rows.append({column: value, "rows": metrics["n"], "wape": metrics["wape"],
                     "mae": metrics["mae"], "bias": metrics["bias"]})
    return pd.DataFrame(rows).sort_values("wape", ascending=False).reset_index(drop=True)
