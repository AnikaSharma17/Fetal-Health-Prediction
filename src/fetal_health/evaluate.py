"""Metrics that expose per-class performance, uncertainty, and CV protocol comparisons."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data import CLASS_NAMES
from .train import N_JOBS, RANDOM_STATE, grouped_cv, limited_threads

LABELS = list(CLASS_NAMES)


def summary_metrics(y_true, y_pred) -> dict:
    """Headline metrics plus precision and recall for each class."""
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS,
                                                              zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
    }
    metrics.update({f"recall_{CLASS_NAMES[c]}": float(r) for c, r in zip(LABELS, recall)})
    metrics.update({f"precision_{CLASS_NAMES[c]}": float(p) for c, p in zip(LABELS, precision)})
    return metrics


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, plus a macro-average row."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                          index=[CLASS_NAMES[c] for c in LABELS])
    report.loc["macro avg"] = [p.mean(), r.mean(), f.mean(), s.sum()]
    report["support"] = report["support"].astype(int)
    return report


def _metrics_from_confusion(cm) -> dict:
    """summary_metrics computed from a confusion matrix (rows: true, columns: predicted).

    Much faster than calling the sklearn metrics, which matters for the bootstrap.
    Assumes every class occurs in y_true.
    """
    tp = np.diag(cm).astype(float)
    recall = tp / cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    metrics = {"accuracy": tp.sum() / cm.sum(), "balanced_accuracy": recall.mean(),
               "macro_f1": f1.mean()}
    metrics.update({f"recall_{CLASS_NAMES[c]}": r for c, r in zip(LABELS, recall)})
    metrics.update({f"precision_{CLASS_NAMES[c]}": p for c, p in zip(LABELS, precision)})
    return {k: float(v) for k, v in metrics.items()}


def bootstrap_ci(y_true, y_pred, groups, n_boot=2000, level=0.95, random_state=RANDOM_STATE):
    """Confidence intervals for summary_metrics, resampling whole recordings.

    Rows of one recording are correlated, so resampling rows would make the intervals
    too narrow; each bootstrap sample draws recordings with replacement instead.
    Samples that miss a class entirely are redrawn, since per-class metrics are
    undefined there.

    Returns a DataFrame indexed by metric with columns estimate, lower, upper.
    """
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    k = len(LABELS)
    code = {label: i for i, label in enumerate(LABELS)}
    cell = np.array([code[t] for t in y_true]) * k + np.array([code[p] for p in y_pred])
    recordings, rec_index = np.unique(groups, return_inverse=True)
    # confusion-matrix counts contributed by each recording, flattened
    per_recording = np.zeros((len(recordings), k * k))
    np.add.at(per_recording, (rec_index, cell), 1)

    rng = np.random.default_rng(random_state)
    samples = []
    while len(samples) < n_boot:
        counts = np.bincount(rng.integers(0, len(recordings), len(recordings)),
                             minlength=len(recordings))
        cm = (counts @ per_recording).reshape(k, k)
        if (cm.sum(axis=1) == 0).any():
            continue
        samples.append(_metrics_from_confusion(cm))
    samples = pd.DataFrame(samples)
    alpha = (1 - level) / 2
    return pd.DataFrame({
        "estimate": pd.Series(summary_metrics(y_true, y_pred)),
        "lower": samples.quantile(alpha),
        "upper": samples.quantile(1 - alpha),
    })


def compare_cv_protocols(estimator, X, y, groupings, n_splits=5, random_state=RANDOM_STATE,
                         n_jobs=N_JOBS):
    """Score one estimator under plain stratified CV and under grouped CV.

    groupings is a Series of group labels or a {name: Series} dict to compare several
    ways of grouping (e.g. real vs proxy recording IDs). A clearly higher stratified
    score means rows of the same recording were inflating it.
    """
    if isinstance(groupings, pd.Series):
        groupings = {"grouped": groupings}
    protocols = {"stratified": (StratifiedKFold(n_splits, shuffle=True, random_state=random_state), None)}
    protocols.update({name: (grouped_cv(n_splits, random_state), g) for name, g in groupings.items()})
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    rows = []
    for name, (cv, g) in protocols.items():
        with limited_threads():
            res = cross_validate(clone(estimator), X, y, groups=g, cv=cv, scoring=scoring,
                                 n_jobs=n_jobs)
        rows.append({"protocol": name,
                     **{f"{m}_mean": res[f"test_{m}"].mean() for m in scoring},
                     **{f"{m}_std": res[f"test_{m}"].std() for m in scoring}})
    return pd.DataFrame(rows).set_index("protocol")
