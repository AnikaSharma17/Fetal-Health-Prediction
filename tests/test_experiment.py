import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fetal_health.decision import mean_cost
from fetal_health.evaluate import summary_metrics
from fetal_health.experiment import KEY_METRICS, PROBA_COLUMNS, RULES, run_holdout, summarize_repeats
from fetal_health.train import holdout_split, model_grid

GRID = {
    "Majority baseline": model_grid()["Majority baseline"],
    "Random Forest": {"model": RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0),
                      "params": {}, "strategies": ["class_weight"]},
}


@pytest.fixture(scope="module")
def result(xyg):
    return run_holdout(*xyg, seed=3, grid=GRID, n_jobs=1)


@pytest.fixture(scope="module")
def row(result):
    return result[0]


@pytest.fixture(scope="module")
def predictions(result):
    return result[1]


def test_run_holdout_uses_the_split_for_its_seed(row, xyg):
    _, X_test, *_ = holdout_split(*xyg, random_state=3)
    assert row["seed"] == 3
    assert row["test_rows"] == len(X_test)
    assert row["model"] == "Random Forest" and row["strategy"] == "class_weight"


def test_run_holdout_chooses_calibration_on_oof_log_loss(row):
    assert row["calibration"] in ("none", "sigmoid", "isotonic")
    losses = {m: row[f"oof_log_loss_{m}"] for m in ("none", "sigmoid", "isotonic")}
    assert losses[row["calibration"]] == min(losses.values())


def test_run_holdout_reports_all_decision_rules(row):
    for rule in RULES:
        for metric in KEY_METRICS:
            assert 0 <= row[f"{rule}_{metric}"] <= 1
        assert row[f"{rule}_mean_cost"] >= 0
    for key in ("threshold", "cost_path_threshold", "cost_suspect_threshold"):
        assert 0 <= row[key] <= 1


def test_predictions_match_the_test_set_and_the_row(row, predictions, xyg):
    X, y, groups = xyg
    _, X_test, _, y_test, _, g_test = holdout_split(X, y, groups, random_state=3)
    assert len(predictions) == len(X_test)
    assert (predictions["seed"] == 3).all()
    # `row` indexes the deduplicated data; recording and label agree with it
    np.testing.assert_array_equal(y.iloc[predictions["row"]], predictions["y_true"])
    np.testing.assert_array_equal(groups.iloc[predictions["row"]], predictions["recording"])
    assert set(predictions["row"]) == set(X.index.get_indexer(X_test.index))
    np.testing.assert_allclose(predictions[PROBA_COLUMNS].sum(axis=1), 1)
    for rule in RULES:
        y_pred = predictions[f"pred_{rule}"]
        assert row[f"{rule}_macro_f1"] == pytest.approx(
            summary_metrics(predictions["y_true"], y_pred)["macro_f1"])
        assert row[f"{rule}_mean_cost"] == pytest.approx(mean_cost(predictions["y_true"], y_pred))


def test_different_seeds_give_different_test_sets(xyg):
    a = holdout_split(*xyg, random_state=0)[5]
    b = holdout_split(*xyg, random_state=1)[5]
    assert set(a) != set(b)


def test_summarize_repeats():
    runs = pd.DataFrame({f"{rule}_{m}": [0.6, 0.8] for rule in RULES
                         for m in KEY_METRICS + ["mean_cost"]})
    table = summarize_repeats(runs)
    assert table.loc[("default", "macro_f1"), "mean"] == pytest.approx(0.7)
    assert table.loc[("threshold", "recall_Pathological"), "min"] == pytest.approx(0.6)
    assert table.loc[("cost", "mean_cost"), "max"] == pytest.approx(0.8)
    assert len(table) == len(RULES) * (len(KEY_METRICS) + 1)


def test_summarize_repeats_skips_missing_columns():
    runs = pd.DataFrame({f"default_{m}": [0.6, 0.8] for m in KEY_METRICS})
    assert len(summarize_repeats(runs)) == len(KEY_METRICS)
