"""Unchanged train-only SHAP helper from shap_works."""
def mean_abs_training_shap(model, X_train):
    """One mean absolute TreeSHAP vector, using this fold's training rows only."""
    import numpy as np
    import shap
    explainer = shap.TreeExplainer(model)
    if callable(explainer):
        shap_values = explainer(X_train)
    else:
        shap_values = explainer.shap_values(X_train)
    shap_array = np.asarray(getattr(shap_values, "values", shap_values))
    if shap_array.shape != X_train.shape:
        raise ValueError(f"Expected SHAP shape {X_train.shape}, got {shap_array.shape}")
    shap_importance = np.mean(np.abs(shap_array), axis=0)
    return shap_importance
