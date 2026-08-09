"""
Stockout-risk classification.

Stockouts occur on 0.28% of store x SKU days. At that base rate a model that
predicts "never" is 99.7% accurate and completely useless, which is why
PROJECT_ARCHITECTURE.md §6 Phase 5 requires precision and recall rather than
accuracy. Accuracy is reported here exactly once, next to the always-negative
baseline, to make the point and then set it aside.

Threshold choice is a business decision, not a modelling one
------------------------------------------------------------
The default 0.5 cut-off is arbitrary for a rare event. Two costs matter:

* a **missed stockout** loses the margin on the demand that went unserved, and
  Phase 2 showed 94% of lost sales happen on promoted days;
* a **false alarm** triggers an unnecessary expedite or over-order, costing
  holding and waste on a perishable line.

`threshold_by_expected_cost` picks the cut-off that minimises expected cost
given those two numbers, and the report shows how the recommendation moves as
the ratio changes rather than presenting one threshold as objective.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.forecast import lightgbm_available  # noqa: E402

LOGGER = logging.getLogger("northstar.ml.stockout")

TARGET = "stockout_flag"


def fit_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    categorical: Sequence[str],
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[np.ndarray] = None,
):
    """
    Gradient-boosted classifier, class-weighted for the rare positive.

    Weighting the positive class does not change the ranking a threshold sweep
    explores, but it keeps the fitted probabilities in a usable range instead of
    collapsing everything towards zero.
    """
    positive_rate = float(np.mean(y_train))
    scale = (1 - positive_rate) / max(positive_rate, 1e-9)

    if lightgbm_available():
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "learning_rate": 0.06,
            "num_leaves": 64,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.85,
            "scale_pos_weight": scale,
            "verbose": -1,
        }
        train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=list(categorical))
        valid_sets, callbacks = [], []
        if X_valid is not None:
            valid_sets = [lgb.Dataset(X_valid, label=y_valid, reference=train_set)]
            callbacks = [lgb.early_stopping(50, verbose=False)]
        model = lgb.train(params, train_set, num_boost_round=400,
                          valid_sets=valid_sets, callbacks=callbacks)
        return model, "lightgbm"

    from sklearn.ensemble import HistGradientBoostingClassifier

    categorical_mask = [c in categorical for c in X_train.columns]
    weights = np.where(y_train == 1, scale, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=64, min_samples_leaf=200,
        categorical_features=categorical_mask, early_stopping=True,
        validation_fraction=0.1, random_state=42,
    ).fit(X_train, y_train, sample_weight=weights)
    return model, "sklearn_hist"


def predict_proba(model, backend: str, X: pd.DataFrame) -> np.ndarray:
    if backend == "lightgbm":
        return model.predict(X)
    return model.predict_proba(X)[:, 1]


def classification_metrics(
    actual: np.ndarray, scores: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Precision, recall, F1 and the confusion counts at a given cut-off."""
    predicted = scores >= threshold
    actual = actual.astype(bool)

    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-9)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alerts": tp + fp,
        "accuracy": float((tp + tn) / max(len(actual), 1)),
    }


def precision_recall_curve(actual: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Precision-recall curve and average precision (PR-AUC)."""
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import precision_recall_curve as sk_curve

    precision, recall, thresholds = sk_curve(actual, scores)
    curve = pd.DataFrame({
        "precision": precision[:-1],
        "recall": recall[:-1],
        "threshold": thresholds,
    })
    curve.attrs["average_precision"] = float(average_precision_score(actual, scores))
    curve.attrs["base_rate"] = float(np.mean(actual))
    return curve


def threshold_by_expected_cost(
    actual: np.ndarray,
    scores: np.ndarray,
    cost_missed: float,
    cost_false_alarm: float,
    grid: int = 200,
) -> pd.DataFrame:
    """
    Sweep thresholds and score each by expected cost.

    cost_missed x false negatives + cost_false_alarm x false positives.
    """
    candidates = np.quantile(scores, np.linspace(0.5, 0.9999, grid))
    rows: List[Dict[str, float]] = []
    for threshold in np.unique(candidates):
        metrics = classification_metrics(actual, scores, threshold)
        rows.append({
            **metrics,
            "expected_cost": metrics["false_negatives"] * cost_missed
            + metrics["false_positives"] * cost_false_alarm,
        })
    return pd.DataFrame(rows).sort_values("expected_cost").reset_index(drop=True)


def baseline_comparison(actual: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """
    The always-negative baseline beside the model, to retire accuracy explicitly.
    """
    actual = actual.astype(bool)
    never = classification_metrics(actual, np.zeros_like(scores), 0.5)
    curve = precision_recall_curve(actual, scores)
    base_rate = float(actual.mean())

    # Threshold giving roughly 50% recall, a planner-friendly operating point.
    at_recall = classification_metrics(
        actual, scores, float(np.quantile(scores, 1 - base_rate * 4))
    )
    return pd.DataFrame([
        {"model": "Always predict 'no stockout'", "precision": 0.0, "recall": 0.0,
         "f1": 0.0, "accuracy": never["accuracy"], "alerts": 0},
        {"model": "Gradient boosting", "precision": at_recall["precision"],
         "recall": at_recall["recall"], "f1": at_recall["f1"],
         "accuracy": at_recall["accuracy"], "alerts": at_recall["alerts"]},
    ]), curve


def lift_by_decile(actual: np.ndarray, scores: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """
    Stockout rate by predicted-risk decile - how a planner would actually use it.
    """
    frame = pd.DataFrame({"actual": actual.astype(int), "score": scores})
    frame["decile"] = pd.qcut(frame["score"].rank(method="first"), bins, labels=False) + 1
    base = frame["actual"].mean()
    summary = frame.groupby("decile").agg(
        rows=("actual", "size"), stockouts=("actual", "sum"), rate=("actual", "mean")
    ).reset_index()
    summary["lift_vs_base"] = summary["rate"] / max(base, 1e-9)
    summary["share_of_all_stockouts"] = summary["stockouts"] / max(frame["actual"].sum(), 1)
    return summary.sort_values("decile", ascending=False).reset_index(drop=True)
