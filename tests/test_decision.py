import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fetal_health.decision import (DEFAULT_COSTS, calibrated_model, choose_calibration,
                                   cost_thresholds, costs_with_pathological_miss,
                                   fit_probability_model, mean_cost, oof_proba,
                                   pathological_threshold, predict_with_threshold,
                                   predict_with_thresholds, probability_scores)
from fetal_health.train import holdout_split, make_pipeline

PROBA = np.array([[0.7, 0.2, 0.1],
                  [0.2, 0.5, 0.3],
                  [0.5, 0.1, 0.4],
                  [0.1, 0.1, 0.8]])
Y = np.array([1, 3, 3, 3])

# Hand-made case for the cost rule: columns are P(Normal), P(Suspect), P(Pathological)
COST_PROBA = np.array([[0.80, 0.10, 0.10],   # Normal
                       [0.50, 0.40, 0.10],   # Normal, but P(Suspect) 0.4
                       [0.40, 0.55, 0.05],   # Suspect
                       [0.30, 0.10, 0.60],   # Pathological
                       [0.50, 0.20, 0.30]])  # Pathological, only P = 0.3
COST_Y = np.array([1, 1, 2, 3, 3])


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


# --- cost-based decision rule -------------------------------------------------------

def test_default_costs_are_illustrative_and_well_formed():
    assert np.all(np.diag(DEFAULT_COSTS) == 0)
    assert np.all(DEFAULT_COSTS[~np.eye(3, dtype=bool)] > 0)
    # missing Pathological is the most expensive error, calling it Suspect is cheaper
    assert DEFAULT_COSTS[2, 0] == DEFAULT_COSTS.max() and DEFAULT_COSTS[2, 1] < DEFAULT_COSTS[2, 0]


def test_costs_with_pathological_miss_changes_only_the_pathological_row():
    costs = costs_with_pathological_miss(20)
    np.testing.assert_array_equal(costs[2], [20, 10, 0])
    np.testing.assert_array_equal(costs[:2], DEFAULT_COSTS[:2])
    np.testing.assert_array_equal(costs_with_pathological_miss(10), DEFAULT_COSTS)
    assert DEFAULT_COSTS[2, 0] == 10  # the base matrix is not modified in place


def test_predict_with_thresholds_precedence():
    proba = np.array([[0.1, 0.5, 0.4],   # both pass: Pathological wins
                      [0.1, 0.6, 0.3],   # only Suspect passes
                      [0.8, 0.1, 0.1],   # neither passes: Normal
                      [0.5, 0.1, 0.4]])  # Pathological passes although Normal is the argmax
    assert predict_with_thresholds(proba, 0.35, 0.3).tolist() == [3, 2, 1, 3]
    # thresholds above every probability: always Normal, even when Normal is not the argmax
    assert predict_with_thresholds(proba, 0.99, 0.99).tolist() == [1, 1, 1, 1]
    # thresholds of 0: always Pathological
    assert predict_with_thresholds(proba, 0.0, 0.0).tolist() == [3, 3, 3, 3]


def test_mean_cost_is_zero_for_perfect_predictions_and_weights_errors():
    y = np.array([1, 2, 3, 3])
    assert mean_cost(y, y) == 0
    assert mean_cost(y, np.array([2, 2, 3, 3])) == pytest.approx(1 / 4)
    assert mean_cost(y, np.array([1, 2, 1, 2])) == pytest.approx((10 + 5) / 4)
    assert mean_cost(pd.Series(y), pd.Series(y)) == 0


def test_cost_thresholds_known_answer_on_small_grid():
    # Path 0.2 flags both Pathological rows; Suspect 0.5 flags only the Suspect row: cost 0.
    # Path 0.5 misses the row with P = 0.3 (cost >= 5), Suspect 0.2 adds a false alarm.
    t_path, t_susp = cost_thresholds(COST_Y, COST_PROBA, grid=[0.2, 0.5, 0.8])
    assert (t_path, t_susp) == (0.2, 0.5)
    assert mean_cost(COST_Y, predict_with_thresholds(COST_PROBA, t_path, t_susp)) == 0


def test_cost_thresholds_known_answer_on_default_grid():
    # Zero cost needs 0.1 < path <= 0.3 and 0.4 < suspect <= 0.55; ties go to the highest.
    assert cost_thresholds(COST_Y, COST_PROBA) == (0.3, 0.55)


def test_cost_thresholds_ties_go_to_highest_thresholds():
    # Only Normal rows with small probabilities: every pair that flags nothing is optimal
    proba = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
    assert cost_thresholds(np.array([1, 1]), proba, grid=[0.2, 0.5, 0.8]) == (0.8, 0.8)
    assert cost_thresholds(np.array([1, 1]), proba) == (0.99, 0.99)


def test_cost_thresholds_minimise_cost_over_the_grid():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([2, 1, 1], size=300)
    y = np.array([rng.choice([1, 2, 3], p=p) for p in proba])
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    t_path, t_susp = cost_thresholds(y, proba, grid=grid)
    best = mean_cost(y, predict_with_thresholds(proba, t_path, t_susp))
    brute = min(mean_cost(y, predict_with_thresholds(proba, a, b)) for a in grid for b in grid)
    assert best == pytest.approx(brute)


MISS_COSTS = (2, 5, 10, 20, 50)


@pytest.mark.parametrize("seed", range(5))
def test_higher_pathological_miss_cost_never_raises_its_threshold(seed):
    # Guaranteed when the Suspect threshold cannot interact: P(Suspect) = 0, so nothing is
    # ever called Suspect and only the Pathological threshold matters. Then a higher miss
    # cost only adds weight to Pathological rows below the threshold.
    rng = np.random.default_rng(seed)
    p = rng.dirichlet([3, 1], size=400)
    proba = np.column_stack([p[:, 0], np.zeros(len(p)), p[:, 1]])
    y = np.where(rng.random(len(p)) < p[:, 1], 3, 1)
    thresholds = [cost_thresholds(y, proba, costs_with_pathological_miss(c))[0]
                  for c in MISS_COSTS]
    assert all(a >= b for a, b in zip(thresholds, thresholds[1:])), thresholds
    assert thresholds[0] > thresholds[-1]  # the test is not vacuous


def test_joint_optimum_can_raise_the_pathological_threshold():
    # With both thresholds chosen together the property is not guaranteed: a higher miss
    # cost can make catching Pathological rows as Suspect (half the cost) the cheaper
    # route, which lets the Pathological threshold rise. This small random case (found by
    # search; rare in practice) documents that the result is still the true optimum.
    rng = np.random.default_rng(844)
    n = rng.integers(5, 40)
    proba = rng.dirichlet(rng.uniform(0.3, 3, 3), size=n)
    y = rng.choice([1, 2, 3], size=n)
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    low, high = (cost_thresholds(y, proba, costs_with_pathological_miss(c), grid=grid)
                 for c in (2, 5))
    assert high[0] > low[0]
    costs = costs_with_pathological_miss(5)
    brute = min(mean_cost(y, predict_with_thresholds(proba, a, b), costs)
                for a in grid for b in grid)
    assert mean_cost(y, predict_with_thresholds(proba, *high), costs) == pytest.approx(brute)


# --- model-based tests ----------------------------------------------------------------

@pytest.fixture(scope="module")
def small_rf():
    return make_pipeline(RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0))


@pytest.fixture(scope="module")
def train_split(xyg):
    X_train, X_test, y_train, _, g_train, _ = holdout_split(*xyg)
    return X_train, X_test, y_train, g_train


def test_calibrated_probabilities_are_valid(train_split, small_rf):
    X_train, X_test, y_train, g_train = train_split
    proba = calibrated_model(small_rf, X_train, y_train, g_train).predict_proba(X_test)
    assert proba.shape == (len(X_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_oof_proba_covers_every_row(train_split, small_rf):
    X_train, _, y_train, g_train = train_split
    proba = oof_proba(small_rf, X_train, y_train, g_train)
    np.testing.assert_allclose(proba.sum(axis=1), 1)


def test_choose_calibration_lets_none_compete(train_split, small_rf):
    X_train, _, y_train, g_train = train_split
    method, scores, oof = choose_calibration(small_rf, X_train, y_train, g_train)
    assert method in (None, "sigmoid", "isotonic")
    assert set(scores.index) == {"none", "sigmoid", "isotonic"}
    assert set(oof) == {None, "sigmoid", "isotonic"}
    best = "none" if method is None else method
    assert scores.loc[best, "log_loss"] == scores["log_loss"].min()
    # the score table is computed from the returned OOF probabilities
    assert scores.loc["none", "log_loss"] == pytest.approx(
        probability_scores(y_train, oof[None])["log_loss"])


def test_fit_probability_model_with_and_without_calibration(train_split, small_rf):
    X_train, X_test, y_train, g_train = train_split
    for method in (None, "sigmoid"):
        proba = fit_probability_model(small_rf, X_train, y_train, g_train, method).predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1)
    plain = fit_probability_model(small_rf, X_train, y_train, g_train, None)
    np.testing.assert_allclose(plain.predict_proba(X_test),
                               small_rf.fit(X_train, y_train).predict_proba(X_test))
