import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from fetal_health.decision import (calibrated_model, oof_proba, pathological_threshold,
                                   predict_with_threshold, probability_scores)
from fetal_health.train import holdout_split, make_pipeline

PROBA = np.array([[0.7, 0.2, 0.1],
                  [0.2, 0.5, 0.3],
                  [0.5, 0.1, 0.4],
                  [0.1, 0.1, 0.8]])
Y = np.array([1, 3, 3, 3])


def test_threshold_reaches_target_recall_on_fitting_data():
    t = pathological_threshold(Y, PROBA, target_recall=0.66)
    assert t == pytest.approx(0.4)  # keeps the two likeliest of three Pathological rows
    assert (predict_with_threshold(PROBA, t)[Y == 3] == 3).mean() >= 0.66
    assert pathological_threshold(Y, PROBA, target_recall=1.0) == pytest.approx(0.3)


def test_predict_with_threshold_overrides_argmax_both_ways():
    assert predict_with_threshold(PROBA, 0.25).tolist() == [1, 3, 3, 3]
    # above every P(Pathological): never Pathological, falls back to the likelier other class
    assert predict_with_threshold(PROBA, 0.9).tolist() == [1, 2, 1, 1]


def test_probability_scores_reward_confident_correct_predictions():
    perfect = np.eye(3)[[0, 2, 2, 2]]
    assert probability_scores(Y, perfect)["brier"] == pytest.approx(0)
    assert probability_scores(Y, PROBA)["brier"] > 0


@pytest.fixture(scope="module")
def small_rf():
    return make_pipeline(RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0))


def test_calibrated_probabilities_are_valid(xyg, small_rf):
    X_train, X_test, y_train, _, g_train, _ = holdout_split(*xyg)
    proba = calibrated_model(small_rf, X_train, y_train, g_train).predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_oof_proba_covers_every_row(xyg, small_rf):
    X_train, _, y_train, _, g_train, _ = holdout_split(*xyg)
    proba = oof_proba(small_rf, X_train, y_train, g_train)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
