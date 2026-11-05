"""Leak-free model pipelines, holdout split and hyperparameter search."""
from contextlib import contextmanager

import pandas as pd
from joblib import parallel_config
from threadpoolctl import threadpool_limits
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
SCORING = "f1_macro"  # every class counts equally; accuracy is dominated by Normal (~78%)
# Parallel CV workers. More than this (e.g. n_jobs=-1 on a 32-core machine) intermittently
# broke the worker pool on Windows ("BrokenProcessPool", WinError 6).
N_JOBS = 8

# Ways of handling the class imbalance, compared for every model that supports them
STRATEGIES = ("smote", "class_weight", "none")


@contextmanager
def limited_threads():
    """Run model fits single-threaded inside; parallelism comes from CV workers instead.

    HistGradientBoosting uses OpenMP threads by default, which on a data set this small
    costs far more than it saves (a single fit measured ~20x slower at 32 threads than
    at 1), and oversubscribes the CPU when CV folds already run in parallel.
    """
    with parallel_config(backend="loky", inner_max_num_threads=1),             threadpool_limits(1, user_api="openmp"):
        yield


def make_pipeline(model, strategy="smote", random_state=RANDOM_STATE):
    """Scaler + (SMOTE | class weights | nothing) + model.

    Everything is fit only on the data passed to .fit(), so inside cross-validation the
    scaler and SMOTE only ever see the training folds. SMOTE is skipped at predict time.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    model = clone(model)
    steps = [("scale", MinMaxScaler())]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=random_state)))
    elif strategy == "class_weight":
        model.set_params(class_weight="balanced")
    steps.append(("clf", model))
    return Pipeline(steps)


def model_grid(random_state=RANDOM_STATE):
    """Candidate models, their hyperparameter grids and the imbalance strategies to try.

    Models without a class_weight parameter (KNN, AdaBoost) are tried with SMOTE and
    without any correction only.
    """
    return {
        "Majority baseline": {
            "model": DummyClassifier(strategy="most_frequent"),
            "params": {},
            "strategies": ["none"],  # with balanced classes "most frequent" would be arbitrary
        },
        "Gradient Boosting": {
            "model": HistGradientBoostingClassifier(random_state=random_state),
            "params": {"learning_rate": [0.05, 0.1], "max_iter": [100, 300], "max_depth": [3, None],
                       "l2_regularization": [0.0, 1.0]},
        },
        "SVM": {
            "model": SVC(),
            "params": {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=random_state),
            "params": {"n_estimators": [10, 50, 100], "max_depth": [2, 3, 4, 5, 6],
                       "max_features": ["sqrt", "log2"]},
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=5000),
            "params": {"C": [1, 5, 10]},
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7, 9]},
            "strategies": ["smote", "none"],
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "params": {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 4, 6, 8],
                       "min_samples_leaf": [1, 2, 3, 4]},
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=random_state),
            "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
            "strategies": ["smote", "none"],
        },
    }


def grouped_cv(n_splits=5, random_state=RANDOM_STATE):
    """Stratified CV that never puts rows of the same recording on both sides."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def holdout_split(X, y, groups, n_splits=5, random_state=RANDOM_STATE):
    """Group-aware, stratified train/test split; the test set is 1/n_splits of the data.

    Returns (X_train, X_test, y_train, y_test, groups_train, groups_test).
    """
    train_idx, test_idx = next(grouped_cv(n_splits, random_state).split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx], groups.iloc[test_idx])


def search_models(X, y, groups, grid=None, cv=None, scoring=SCORING, n_jobs=N_JOBS):
    """Grid-search every model under every imbalance strategy it supports, with grouped CV.

    Returns (results, best):
      results: one row per (model, strategy), sorted by CV score, best first;
      best: {model name: refit pipeline of that model's best strategy}.
    """
    grid = model_grid() if grid is None else grid
    cv = grouped_cv() if cv is None else cv
    rows, fitted = [], {}
    for name, spec in grid.items():
        params = {f"clf__{k}": v for k, v in spec["params"].items()}
        for strategy in spec.get("strategies", STRATEGIES):
            search = GridSearchCV(make_pipeline(spec["model"], strategy), params, cv=cv,
                                  scoring=scoring, n_jobs=n_jobs)
            with limited_threads():
                search.fit(X, y, groups=groups)
            fitted[name, strategy] = search.best_estimator_
            i = search.best_index_
            rows.append({
                "model": name,
                "strategy": strategy,
                f"cv_{scoring}": search.best_score_,
                "cv_std": search.cv_results_["std_test_score"][i],
                "parameters": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
            })
    results = pd.DataFrame(rows).sort_values(f"cv_{scoring}", ascending=False, ignore_index=True)
    best_rows = results.drop_duplicates("model")  # first row per model = its best strategy
    best = {r.model: fitted[r.model, r.strategy] for r in best_rows.itertuples()}
    return results, best


def strategy_table(results, scoring=SCORING):
    """Model x strategy table of CV scores (the resampling ablation)."""
    table = results.pivot(index="model", columns="strategy", values=f"cv_{scoring}")
    table = table.reindex(columns=[s for s in STRATEGIES if s in table.columns])
    return table.loc[table.max(axis=1).sort_values(ascending=False).index]
