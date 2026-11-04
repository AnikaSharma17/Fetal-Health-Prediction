"""Leak-free model pipelines, holdout split and hyperparameter search."""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
SCORING = "f1_macro"  # every class counts equally; accuracy is dominated by Normal (~78%)


def make_pipeline(model, resample=True, random_state=RANDOM_STATE):
    """Scaler (+ SMOTE) + model. Both are fit only on the data passed to .fit(),
    so inside cross-validation they only ever see the training folds."""
    steps = [("scale", MinMaxScaler())]
    if resample:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", model))
    return Pipeline(steps)


def model_grid(random_state=RANDOM_STATE):
    """Candidate models and their hyperparameter grids."""
    return {
        "Majority baseline": {
            "model": DummyClassifier(strategy="most_frequent"),
            "params": {},
            "resample": False,  # with balanced classes "most frequent" would be arbitrary
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
            "model": LogisticRegression(solver="saga", max_iter=5000),
            "params": {"C": [1, 5, 10]},
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7, 9]},
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "params": {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 4, 6, 8],
                       "min_samples_leaf": [1, 2, 3, 4]},
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=random_state),
            "params": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
        },
    }


def grouped_cv(n_splits=5, random_state=RANDOM_STATE):
    """Stratified CV that never puts rows of the same (pseudo-)recording on both sides."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def holdout_split(X, y, groups, n_splits=5, random_state=RANDOM_STATE):
    """Group-aware, stratified train/test split; the test set is 1/n_splits of the data.

    Returns (X_train, X_test, y_train, y_test, groups_train, groups_test).
    """
    train_idx, test_idx = next(grouped_cv(n_splits, random_state).split(X, y, groups))
    return (X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx],
            groups.iloc[train_idx], groups.iloc[test_idx])


def search_models(X, y, groups, grid=None, cv=None, scoring=SCORING, n_jobs=4):
    """Grid-search each model with grouped CV.

    Returns (results DataFrame sorted by CV score, {model name: refit best pipeline}).
    """
    grid = model_grid() if grid is None else grid
    cv = grouped_cv() if cv is None else cv
    rows, best = [], {}
    for name, spec in grid.items():
        params = {f"clf__{k}": v for k, v in spec["params"].items()}
        search = GridSearchCV(make_pipeline(spec["model"], resample=spec.get("resample", True)),
                              params, cv=cv, scoring=scoring, n_jobs=n_jobs)
        search.fit(X, y, groups=groups)
        best[name] = search.best_estimator_
        i = search.best_index_
        rows.append({
            "model": name,
            f"cv_{scoring}": search.best_score_,
            "cv_std": search.cv_results_["std_test_score"][i],
            "parameters": {k.removeprefix("clf__"): v for k, v in search.best_params_.items()},
        })
    results = pd.DataFrame(rows).sort_values(f"cv_{scoring}", ascending=False, ignore_index=True)
    return results, best
