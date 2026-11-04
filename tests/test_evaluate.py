import pytest

from fetal_health.evaluate import per_class_report, summary_metrics

Y_TRUE = [1, 1, 1, 1, 2, 2, 3, 3]
Y_PRED = [1, 1, 1, 2, 2, 1, 3, 1]


def test_summary_metrics_expose_per_class_recall():
    m = summary_metrics(Y_TRUE, Y_PRED)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["recall_Normal"] == pytest.approx(3 / 4)
    assert m["recall_Suspect"] == pytest.approx(1 / 2)
    assert m["recall_Pathological"] == pytest.approx(1 / 2)
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
