"""SHAP explanations for a fitted scaler (+ SMOTE) + model pipeline."""
import pandas as pd
import shap


def shap_explanation(pipeline, X: pd.DataFrame, background: pd.DataFrame | None = None,
                     max_background=100, random_state=0):
    """SHAP values of the pipeline's model for the rows of X.

    Tree models (random forest, gradient boosting, ...) use the exact TreeExplainer on
    the scaled features, in the model's raw output: class probabilities for a random
    forest, per-class log-odds for gradient boosting. MinMax scaling is monotone per
    feature, so attributions match the raw features, and raw values are put back for
    plotting. Other models fall back to a model-agnostic permutation explainer of
    predict_proba on raw features, which needs background data (e.g. the training set)
    and is much slower.

    Returns a shap.Explanation of shape (rows, features, classes).
    """
    clf = pipeline.named_steps["clf"]
    scaled = pd.DataFrame(pipeline.named_steps["scale"].transform(X), columns=X.columns,
                          index=X.index)
    try:
        explainer = shap.TreeExplainer(clf)
    except Exception:  # not a tree model shap knows
        if background is None:
            raise ValueError("this model needs background data for SHAP") from None
        bg = background.sample(min(max_background, len(background)), random_state=random_state)
        explainer = shap.Explainer(pipeline.predict_proba, shap.maskers.Independent(bg),
                                   feature_names=list(X.columns))
        return explainer(X)
    explanation = explainer(scaled)
    explanation.data = X.to_numpy()
    explanation.feature_names = list(X.columns)
    return explanation
