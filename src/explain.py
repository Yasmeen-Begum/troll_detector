import numpy as np

def explain_prediction(model, X, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    explanation = [(feature_names[i], float(importances[i])) for i in sorted_idx]
    return explanation
