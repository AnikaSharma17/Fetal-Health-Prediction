"""Repeat the whole selection-and-evaluation procedure over several train/test splits.

One test set with 9 Pathological recordings gives wide intervals. Re-running the full
procedure (model search, threshold, test evaluation) on different recording-grouped
splits shows how much the reported numbers depend on which recordings land in the
test set.
"""
import pandas as pd

from .decision import calibrated_model, oof_proba, pathological_threshold, predict_with_threshold
from .evaluate import summary_metrics
from .train import N_JOBS, SCORING, holdout_split, search_models

# Metrics summarized across splits
KEY_METRICS = ["macro_f1", "balanced_accuracy", "accuracy", "recall_Pathological",
               "precision_Pathological", "recall_Suspect", "precision_Suspect"]


def run_holdout(X, y, groups, seed, grid=None, target_recall=0.95, calibration="sigmoid",
                n_jobs=N_JOBS) -> dict:
    """Run the notebook's procedure on the train/test split given by seed.

    Everything except the split is the same as in the notebook. The one simplification
    is that the calibration method is fixed (sigmoid, which the notebook selects on its
    own split) instead of being chosen per split. With seed=42 this reproduces the
    notebook's numbers.

    Returns one flat row: split sizes, the selected model, test metrics for the
    model's most likely class (default_*) and for the calibrated Pathological
    threshold (threshold_*).
    """
    X_train, X_test, y_train, y_test, g_train, g_test = holdout_split(X, y, groups, random_state=seed)
    results, best = search_models(X_train, y_train, g_train, grid=grid, n_jobs=n_jobs)
    name, strategy, cv_score = results.loc[0, ["model", "strategy", f"cv_{SCORING}"]]
    clf = best[name]

    row = {
        "seed": seed,
        "test_rows": len(X_test),
        "test_recordings": g_test.nunique(),
        "test_pathological_rows": int((y_test == 3).sum()),
        "test_pathological_recordings": g_test[y_test == 3].nunique(),
        "model": name,
        "strategy": strategy,
        "cv_macro_f1": float(cv_score),
    }
    row.update({f"default_{k}": v for k, v in summary_metrics(y_test, clf.predict(X_test)).items()})

    threshold = pathological_threshold(
        y_train, oof_proba(clf, X_train, y_train, g_train, method=calibration), target_recall)
    calibrated = calibrated_model(clf, X_train, y_train, g_train, method=calibration)
    y_pred = predict_with_threshold(calibrated.predict_proba(X_test), threshold)
    row["threshold"] = threshold
    row.update({f"threshold_{k}": v for k, v in summary_metrics(y_test, y_pred).items()})
    return row


def summarize_repeats(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, minimum and maximum of the key test metrics across splits."""
    rows = {}
    for rule in ("default", "threshold"):
        for metric in KEY_METRICS:
            values = runs[f"{rule}_{metric}"]
            rows[rule, metric] = {"mean": values.mean(), "std": values.std(ddof=1),
                                  "min": values.min(), "max": values.max()}
    table = pd.DataFrame(rows).T
    table.index.names = ["decision_rule", "metric"]
    return table
