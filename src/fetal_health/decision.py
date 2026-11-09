"""Probability calibration and decision rules for the Pathological and Suspect classes.

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
SUSPECT_COL = LABELS.index(2)
CALIBRATION_METHODS = (None, "sigmoid", "isotonic")  # None: keep the model's own probabilities

# Illustrative misclassification costs (rows: true class, columns: predicted class, both in
# the order Normal, Suspect, Pathological). Any false alarm costs 1; calling a Suspect case
# Normal costs 2; calling a Pathological case Suspect costs 5, and Normal 10. These are an
# assumption for demonstration, not clinical guidance: real costs must come from clinicians.
DEFAULT_COSTS = np.array([[0, 1, 1],
                          [2, 0, 1],
                          [10, 5, 0]], dtype=float)


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


def method_name(method) -> str:
    return "none" if method is None else method


def choose_calibration(estimator, X, y, groups, methods=CALIBRATION_METHODS, n_splits=5,
                       random_state=RANDOM_STATE):
    """Pick the calibration method (or none) with the lowest out-of-fold log loss.

    "No calibration" competes on equal terms, so a calibrator is only added when it
    improves held-out probabilities within the training set.

    Returns (method, scores DataFrame indexed by method name, {method: OOF probabilities}).
    """
    if not hasattr(estimator, "predict_proba"):
        methods = [m for m in methods if m is not None]
    oof = {m: oof_proba(estimator, X, y, groups, m, n_splits, random_state) for m in methods}
    scores = pd.DataFrame({method_name(m): probability_scores(y, p) for m, p in oof.items()}).T
    best = min(oof, key=lambda m: scores.loc[method_name(m), "log_loss"])
    return best, scores, oof


def fit_probability_model(estimator, X, y, groups, method, n_splits=5, random_state=RANDOM_STATE):
    """The estimator refit on (X, y), wrapped in a calibrator unless method is None."""
    if method is None:
        with limited_threads():
            return clone(estimator).fit(X, y)
    return calibrated_model(estimator, X, y, groups, method, n_splits, random_state)


def costs_with_pathological_miss(miss_cost, base=DEFAULT_COSTS):
    """DEFAULT_COSTS with a different cost for calling a Pathological case Normal
    (calling it Suspect costs half as much)."""
    costs = base.copy()
    costs[PATH_COL] = [miss_cost, miss_cost / 2, 0]
    return costs


def predict_with_thresholds(proba, path_threshold, suspect_threshold):
    """Flag the most severe class whose probability passes its threshold.

    Pathological if P(Pathological) >= path_threshold, else Suspect if
    P(Suspect) >= suspect_threshold, else Normal.
    """
    proba = np.asarray(proba)
    return np.where(proba[:, PATH_COL] >= path_threshold, PATHOLOGICAL,
                    np.where(proba[:, SUSPECT_COL] >= suspect_threshold, 2, 1))


def mean_cost(y_true, y_pred, costs=DEFAULT_COSTS) -> float:
    """Average misclassification cost per segment under the cost matrix."""
    index = {label: i for i, label in enumerate(LABELS)}
    true_i = np.array([index[v] for v in np.asarray(y_true)])
    pred_i = np.array([index[v] for v in np.asarray(y_pred)])
    return float(costs[true_i, pred_i].mean())


def cost_thresholds(y, proba, costs=DEFAULT_COSTS, grid=None):
    """Pathological and Suspect thresholds that minimise the mean cost on (y, proba).

    Fit on out-of-fold training predictions, never on the test set. Among equally good
    pairs, the highest thresholds (fewest alarms) win.

    Both thresholds are searched jointly over the grid. For a fixed Suspect threshold, a
    higher Pathological miss cost never raises the Pathological threshold; with the joint
    search this usually holds too but is not guaranteed, because catching a Pathological
    case as Suspect also gets cheaper relative to missing it.

    Returns (path_threshold, suspect_threshold).
    """
    grid = np.round(np.arange(0.99, 0.0, -0.01), 2) if grid is None else np.sort(grid)[::-1]
    proba = np.asarray(proba)
    index = {label: i for i, label in enumerate(LABELS)}
    true_i = np.array([index[v] for v in np.asarray(y)])
    p_path, p_susp = proba[:, PATH_COL], proba[:, SUSPECT_COL]
    suspect_flags = p_susp[None, :] >= grid[:, None]  # (thresholds, rows)
    best = (np.inf, None, None)
    for tp in grid:
        flag_path = p_path >= tp
        pred_i = np.where(flag_path[None, :], PATH_COL, np.where(suspect_flags, SUSPECT_COL, 0))
        cost = costs[true_i[None, :], pred_i].mean(axis=1)
        j = int(np.argmin(cost))  # first minimum = highest Suspect threshold
        if cost[j] < best[0] - 1e-12:
            best = (cost[j], float(tp), float(grid[j]))
    return best[1], best[2]
