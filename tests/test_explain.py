import numpy as np
from sklearn.ensemble import RandomForestClassifier

from fetal_health.explain import shap_explanation
from fetal_health.train import holdout_split, make_pipeline


def test_tree_shap_values_add_up_to_predicted_probabilities(xyg):
    X_train, X_test, y_train, *_ = holdout_split(*xyg)
    pipe = make_pipeline(RandomForestClassifier(n_estimators=20, max_depth=4, random_state=0))
    pipe.fit(X_train, y_train)
    rows = X_test.head(10)
    exp = shap_explanation(pipe, rows)
    assert exp.values.shape == (10, X_test.shape[1], 3)
    np.testing.assert_allclose(exp.data, rows.to_numpy())  # raw values kept for plots
    # for a random forest the raw output is the class probability
    np.testing.assert_allclose(exp.values.sum(axis=1) + exp.base_values, pipe.predict_proba(rows),
                               atol=1e-6)
