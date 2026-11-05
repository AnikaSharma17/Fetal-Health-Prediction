import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from fetal_health.experiment import KEY_METRICS, run_holdout, summarize_repeats
from fetal_health.train import holdout_split, model_grid

GRID = {
    "Majority baseline": model_grid()["Majority baseline"],
    "Random Forest": {"model": RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0),
                      "params": {}, "strategies": ["class_weight"]},
}


@pytest.fixture(scope="module")
def row(xyg):
    return run_holdout(*xyg, seed=3, grid=GRID, n_jobs=1)


def test_run_holdout_uses_the_split_for_its_seed(row, xyg):
    _, X_test, *_ = holdout_split(*xyg, random_state=3)
    assert row["seed"] == 3
    assert row["test_rows"] == len(X_test)
    assert row["model"] == "Random Forest" and row["strategy"] == "class_weight"


def test_run_holdout_reports_both_decision_rules(row):
    for rule in ("default", "threshold"):
        for metric in KEY_METRICS:
            assert 0 <= row[f"{rule}_{metric}"] <= 1
    assert 0 <= row["threshold"] <= 1


def test_different_seeds_give_different_test_sets(xyg):
    a = holdout_split(*xyg, random_state=0)[5]
    b = holdout_split(*xyg, random_state=1)[5]
    assert set(a) != set(b)


def test_summarize_repeats():
    runs = pd.DataFrame({f"{rule}_{m}": [0.6, 0.8] for rule in ("default", "threshold")
                         for m in KEY_METRICS})
    table = summarize_repeats(runs)
    assert table.loc[("default", "macro_f1"), "mean"] == pytest.approx(0.7)
    assert table.loc[("threshold", "recall_Pathological"), "min"] == pytest.approx(0.6)
    assert len(table) == 2 * len(KEY_METRICS)
