import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from fetal_health.data import pseudo_recording_groups
from fetal_health.evaluate import (bootstrap_ci, compare_cv_protocols, per_class_report,
                                   summary_metrics)
from fetal_health.train import make_pipeline

Y_TRUE = [1, 1, 1, 1, 2, 2, 3, 3]
Y_PRED = [1, 1, 1, 2, 2, 1, 3, 1]


def test_summary_metrics_expose_per_class_recall_and_precision():
    m = summary_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["recall_Normal"] == pytest.approx(3 / 4)
    assert m["recall_Suspect"] == pytest.approx(1 / 2)
    assert m["recall_Pathological"] == pytest.approx(1 / 2)
    assert m["precision_Normal"] == pytest.approx(3 / 5)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2 + 1 / 2) / 3)


def test_per_class_report_shape_and_support():
    report = per_class_report(Y_TRUE, Y_PRED)
    assert list(report.index) == ["Normal", "Suspect", "Pathological", "macro avg"]
    assert report["support"].tolist() == [4, 2, 2, 8]
    assert report.loc["Pathological", "precision"] == pytest.approx(1.0)


def test_majority_predictions_score_zero_on_minority_classes():
    m = summary_metrics(Y_TRUE, [1] * len(Y_TRUE))
    assert m["recall_Suspect"] == 0 and m["recall_Pathological"] == 0
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)


def test_bootstrap_ci_brackets_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y_true = np.repeat([1, 2, 3], 40)
    y_pred = np.where(rng.random(120) < 0.8, y_true, 1)
    groups = np.arange(120) // 3  # recordings of three rows each
    ci = bootstrap_ci(y_true, y_pred, groups, n_boot=200)
    assert (ci["lower"] <= ci["estimate"]).all() and (ci["estimate"] <= ci["upper"]).all()
    assert (ci.loc["macro_f1", "upper"] - ci.loc["macro_f1", "lower"]) > 0
    pd.testing.assert_frame_equal(ci, bootstrap_ci(y_true, y_pred, groups, n_boot=200))


def test_bootstrap_resamples_whole_recordings():
    """One recording per class: the only resample containing every class is the full
    data set, so if recordings (not rows) are resampled the interval has zero width."""
    ci = bootstrap_ci([1, 1, 2, 3], [1, 2, 2, 3], ["a", "a", "b", "c"], n_boot=20)
    assert (ci["lower"] == ci["upper"]).all()
    assert (ci["lower"] == ci["estimate"]).all()


def test_compare_cv_protocols_has_one_row_per_protocol(xyg, clean_df):
    X, y, groups = xyg
    table = compare_cv_protocols(make_pipeline(DecisionTreeClassifier(max_depth=3, random_state=0)),
                                 X, y, {"recording": groups, "proxy": pseudo_recording_groups(clean_df)},
                                 n_jobs=1)
    assert list(table.index) == ["stratified", "recording", "proxy"]
    assert table["macro_f1_mean"].between(0, 1).all()


def test_fast_bootstrap_metrics_match_sklearn():
    from sklearn.metrics import confusion_matrix

    from fetal_health.evaluate import _metrics_from_confusion
    y_true = [1, 1, 1, 2, 2, 3, 3, 3]
    for y_pred in (Y_PRED, [1] * 8, [3, 1, 2, 2, 3, 3, 1, 2]):
        fast = _metrics_from_confusion(confusion_matrix(y_true, y_pred[:8], labels=[1, 2, 3]))
        slow = summary_metrics(y_true, y_pred[:8])
        assert fast == pytest.approx(slow)
