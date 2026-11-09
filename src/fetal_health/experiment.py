"""Repeat the whole selection-and-evaluation procedure over several train/test splits.

One test set with 9 Pathological recordings gives wide intervals. Re-running the full
procedure (model search, calibration choice, thresholds, test evaluation) on different
recording-grouped splits shows how much the reported numbers depend on which recordings
land in the test set.
"""
import numpy as np
import pandas as pd

from .data import CLASS_NAMES
from .decision import (DEFAULT_COSTS, LABELS, choose_calibration, cost_thresholds,
                       fit_probability_model, mean_cost, method_name, pathological_threshold,
                       predict_with_threshold, predict_with_thresholds)
from .evaluate import summary_metrics
from .train import N_JOBS, SCORING, holdout_split, search_models

# Metrics summarized across splits
KEY_METRICS = ["macro_f1", "balanced_accuracy", "accuracy", "recall_Pathological",
               "precision_Pathological", "recall_Suspect", "precision_Suspect"]
# Decision rules: most likely class, 95%-recall Pathological threshold, cost-based thresholds
RULES = ("default", "threshold", "cost")
PROBA_COLUMNS = [f"p_{CLASS_NAMES[c]}" for c in LABELS]


def run_holdout(X, y, groups, seed, grid=None, target_recall=0.95, costs=DEFAULT_COSTS,
                n_jobs=N_JOBS) -> tuple[dict, pd.DataFrame]:
    """Run the notebook's procedure on the train/test split given by seed.

    Everything except the split is the same as in the notebook: model search, then the
    calibration method (none, sigmoid or isotonic) chosen on out-of-fold log loss, then
    both thresholds chosen on out-of-fold training predictions. With seed=42 this
    reproduces the notebook's numbers.

    Returns (row, predictions):
      row: one flat dict with split sizes, the selected model and calibration, and test
        metrics for the model's most likely class (default_*), the Pathological threshold
        for target_recall (threshold_*) and the cost-minimising thresholds (cost_*),
        plus the mean cost of each rule under `costs`.
      predictions: one test row per row of X_test, with its position in X (`row`),
        recording, true label, the three rules' predictions and the class probabilities.
    """
    X_train, X_test, y_train, y_test, g_train, g_test = holdout_split(X, y, groups, random_state=seed)
    results, best = search_models(X_train, y_train, g_train, grid=grid, n_jobs=n_jobs)
    name, strategy, cv_score = results.loc[0, ["model", "strategy", f"cv_{SCORING}"]]
    clf = best[name]

    calibration, scores, oof = choose_calibration(clf, X_train, y_train, g_train)
    proba = fit_probability_model(clf, X_train, y_train, g_train, calibration).predict_proba(X_test)
    oof_chosen = oof[calibration]

    threshold = pathological_threshold(y_train, oof_chosen, target_recall)
    cost_path, cost_suspect = cost_thresholds(y_train, oof_chosen, costs)
    preds = {
        "default": clf.predict(X_test),
        "threshold": predict_with_threshold(proba, threshold),
        "cost": predict_with_thresholds(proba, cost_path, cost_suspect),
    }

    row = {
        "seed": seed,
        "test_rows": len(X_test),
        "test_recordings": g_test.nunique(),
        "test_pathological_rows": int((y_test == 3).sum()),
        "test_pathological_recordings": g_test[y_test == 3].nunique(),
        "model": name,
        "strategy": strategy,
        "cv_macro_f1": float(cv_score),
        "calibration": method_name(calibration),
        **{f"oof_log_loss_{m}": float(v) for m, v in scores["log_loss"].items()},
        "threshold": threshold,
        "cost_path_threshold": cost_path,
        "cost_suspect_threshold": cost_suspect,
    }
    for rule, y_pred in preds.items():
        row.update({f"{rule}_{k}": v for k, v in summary_metrics(y_test, y_pred).items()})
        row[f"{rule}_mean_cost"] = mean_cost(y_test, y_pred, costs)

    predictions = pd.DataFrame({
        "seed": seed,
        "row": X.index.get_indexer(X_test.index),
        "recording": np.asarray(g_test),
        "y_true": np.asarray(y_test),
        **{f"pred_{rule}": np.asarray(p) for rule, p in preds.items()},
        **dict(zip(PROBA_COLUMNS, proba.T)),
    })
    return row, predictions


def summarize_repeats(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation, minimum and maximum of the key test metrics across splits,
    for each decision rule (plus the mean cost, where the runs have it)."""
    rows = {}
    for rule in RULES:
        for metric in KEY_METRICS + ["mean_cost"]:
            column = f"{rule}_{metric}"
            if column not in runs:
                continue
            values = runs[column]
            rows[rule, metric] = {"mean": values.mean(), "std": values.std(ddof=1),
                                  "min": values.min(), "max": values.max()}
    table = pd.DataFrame(rows).T
    table.index.names = ["decision_rule", "metric"]
    return table
