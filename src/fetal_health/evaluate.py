"""Metrics that expose per-class performance, and CV protocol comparisons."""
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data import CLASS_NAMES
from .train import RANDOM_STATE, grouped_cv

LABELS = list(CLASS_NAMES)


def summary_metrics(y_true, y_pred) -> dict:
    """Headline metrics plus recall for each class."""
    _, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
    }
    metrics.update({f"recall_{CLASS_NAMES[c]}": float(r) for c, r in zip(LABELS, recall)})
    return metrics


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, plus a macro-average row."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                          index=[CLASS_NAMES[c] for c in LABELS])
    report.loc["macro avg"] = [p.mean(), r.mean(), f.mean(), s.sum()]
    report["support"] = report["support"].astype(int)
    return report


def compare_cv_protocols(estimator, X, y, groups, n_splits=5, random_state=RANDOM_STATE, n_jobs=4):
    """Score one estimator under plain stratified CV and under grouped CV.

    A large drop under grouped CV means rows from the same recording were inflating
    the stratified score.
    """
    protocols = {
        "stratified": (StratifiedKFold(n_splits, shuffle=True, random_state=random_state), None),
        "grouped": (grouped_cv(n_splits, random_state), groups),
    }
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    rows = []
    for name, (cv, g) in protocols.items():
        res = cross_validate(clone(estimator), X, y, groups=g, cv=cv, scoring=scoring, n_jobs=n_jobs)
        rows.append({"protocol": name,
                     **{f"{m}_mean": res[f"test_{m}"].mean() for m in scoring},
                     **{f"{m}_std": res[f"test_{m}"].std() for m in scoring}})
    return pd.DataFrame(rows).set_index("protocol")
