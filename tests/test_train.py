import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fetal_health.data import FEATURES
from fetal_health.evaluate import summary_metrics
from fetal_health.train import (holdout_split, make_pipeline, model_grid, search_models,
                                strategy_table)


@pytest.fixture(scope="module")
def split(xyg):
    return holdout_split(*xyg)


def test_holdout_has_no_shared_rows_or_recordings(split):
    X_train, X_test, y_train, y_test, g_train, g_test = split
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(g_train).isdisjoint(g_test)
    shared = X_train.merge(X_test, on=FEATURES, how="inner")
    assert shared.empty, "identical feature rows on both sides of the split"


def test_holdout_is_stratified_and_sized(split, xyg):
    _, X_test, y_train, y_test, _, _ = split
    assert len(X_test) / len(xyg[0]) == pytest.approx(0.2, abs=0.03)
    for cls in (1, 2, 3):
        assert (y_test == cls).mean() == pytest.approx((y_train == cls).mean(), abs=0.03)


def test_test_set_contains_only_real_rows(split, clean_df):
    """Regression test for oversampling before the split: every test row must be a real record."""
    _, X_test, *_ = split
    matched = X_test.merge(clean_df[FEATURES].drop_duplicates(), on=FEATURES, how="left", indicator=True)
    assert (matched["_merge"] == "both").all()


def test_preprocessing_is_fit_on_training_data_only(split):
    X_train, X_test, y_train, *_ = split
    pipe = make_pipeline(DecisionTreeClassifier(random_state=0)).fit(X_train, y_train)
    scaler = pipe.named_steps["scale"]
    pd.testing.assert_series_equal(pd.Series(scaler.data_min_, index=FEATURES), X_train.min(),
                                   check_names=False)
    assert len(pipe.predict(X_test)) == len(X_test)  # SMOTE never touches prediction


@pytest.mark.parametrize("strategy, has_smote, class_weight", [
    ("smote", True, None), ("class_weight", False, "balanced"), ("none", False, None)])
def test_imbalance_strategies(strategy, has_smote, class_weight):
    model = LogisticRegression()
    pipe = make_pipeline(model, strategy)
    assert ("smote" in pipe.named_steps) == has_smote
    assert pipe.named_steps["clf"].class_weight == class_weight
    assert model.class_weight is None, "the caller's model must not be modified"


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        make_pipeline(LogisticRegression(), "undersample")


def test_search_covers_each_strategy_and_is_reproducible(split):
    X_train, _, y_train, _, g_train, _ = split
    grid = {"Decision Tree": {"model": DecisionTreeClassifier(random_state=0),
                              "params": {"max_depth": [3, 5]}}}
    first, best = search_models(X_train, y_train, g_train, grid=grid, n_jobs=1)
    second, _ = search_models(X_train, y_train, g_train, grid=grid, n_jobs=1)
    pd.testing.assert_frame_equal(first, second)
    assert sorted(first["strategy"]) == ["class_weight", "none", "smote"]
    assert list(best) == ["Decision Tree"]
    assert strategy_table(first).loc["Decision Tree"].max() == first["cv_f1_macro"].max()


def test_model_beats_majority_baseline(split):
    X_train, X_test, y_train, y_test, g_train, _ = split
    grid = {
        "Majority baseline": model_grid()["Majority baseline"],
        "Random Forest": {"model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0),
                          "params": {}, "strategies": ["smote"]},
    }
    _, best = search_models(X_train, y_train, g_train, grid=grid, n_jobs=1)
    baseline = summary_metrics(y_test, best["Majority baseline"].predict(X_test))
    model = summary_metrics(y_test, best["Random Forest"].predict(X_test))
    assert baseline["recall_Pathological"] == 0
    assert model["macro_f1"] >= 0.70
    assert model["macro_f1"] >= baseline["macro_f1"] + 0.30
    assert model["recall_Pathological"] >= 0.70
