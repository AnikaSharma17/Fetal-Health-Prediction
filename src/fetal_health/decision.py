"""Probability calibration and a decision threshold for the Pathological class.

All fitting uses grouped CV on the training set only, so the test set stays untouched
until the final evaluation.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

from .data import CLASS_NAMES
from .train import RANDOM_STATE, grouped_cv, limited_threads

LABELS = list(CLASS_NAMES)
PATHOLOGICAL = 3
PATH_COL = LABELS.index(PATHOLOGICAL)


def calibrated_model(estimator, X, y, groups, method="sigmoid", n_splits=5,
                     random_state=RANDOM_STATE):
    """Fit estimator plus a probability calibrator, with grouped inner CV.

    Each inner fold fits the model on some recordings and the calibrator on the
    held-out recordings; predictions average the fold models.
    """
    splits = list(grouped_cv(n_splits, random_state).split(X, y, groups))
    with limited_threads():
        return CalibratedClassifierCV(clone(estimator), method=method, cv=splits).fit(X, y)


def oof_proba(estimator, X, y, groups, method=None, n_splits=5, random_state=RANDOM_STATE):
    """Out-of-fold class probabilities for every row of (X, y), with grouped CV.

    method=None uses the estimator's own predict_proba; "sigmoid" or "isotonic" wraps
    it in calibrated_model, fit on the training folds only.
    """
    proba = np.zeros((len(X), len(LABELS)))
    for train, test in grouped_cv(n_splits, random_state).split(X, y, groups):
        if method is None:
            with limited_threads():
                model = clone(estimator).fit(X.iloc[train], y.iloc[train])
        else:
            model = calibrated_model(estimator, X.iloc[train], y.iloc[train], groups.iloc[train],
                                     method, n_splits, random_state)
        proba[test] = model.predict_proba(X.iloc[test])
    return proba


def probability_scores(y, proba) -> dict:
    """Log loss and multiclass Brier score (lower is better for both)."""
    onehot = (np.asarray(y)[:, None] == np.array(LABELS)).astype(float)
    return {
        "log_loss": float(log_loss(y, proba, labels=LABELS)),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
    }


def pathological_threshold(y, proba, target_recall=0.95):
    """Largest threshold on P(Pathological) that reaches target_recall on (y, proba).

    Pick it on out-of-fold training predictions, never on the test set.
    """
    p_path = np.asarray(proba)[np.asarray(y) == PATHOLOGICAL, PATH_COL]
    if len(p_path) == 0:
        raise ValueError("no Pathological rows to set a threshold on")
    for t in np.sort(np.unique(p_path))[::-1]:
        if (p_path >= t).mean() >= target_recall:
            return float(t)
    return 0.0


def predict_with_threshold(proba, threshold):
    """Pathological if P(Pathological) >= threshold, otherwise the likelier of the other two."""
    proba = np.asarray(proba)
    others = [i for i in range(len(LABELS)) if i != PATH_COL]
    pred = np.array(LABELS)[others][proba[:, others].argmax(axis=1)]
    return np.where(proba[:, PATH_COL] >= threshold, PATHOLOGICAL, pred)


def recall_threshold_curve(y, proba, thresholds=None) -> pd.DataFrame:
    """Pathological recall and precision across thresholds (for plotting the trade-off)."""
    thresholds = np.linspace(0.05, 0.95, 19) if thresholds is None else thresholds
    y = np.asarray(y)
    rows = []
    for t in thresholds:
        flagged = np.asarray(proba)[:, PATH_COL] >= t
        tp = np.sum(flagged & (y == PATHOLOGICAL))
        rows.append({"threshold": t,
                     "recall": tp / max(np.sum(y == PATHOLOGICAL), 1),
                     "precision": tp / flagged.sum() if flagged.any() else np.nan})
    return pd.DataFrame(rows)
