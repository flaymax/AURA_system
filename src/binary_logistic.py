import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def binary_loglikelihood(probabilities, target):
    # numerical stability guard
    p = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return np.sum(target * np.log(p) + (1 - target) * np.log(1 - p))


def coef_standard_errors(model, X_scaled):
    # fitted probabilities define local curvature
    p = model.predict_proba(X_scaled)[:, 1]
    w = p * (1 - p)

    # add intercept explicitly
    design_matrix = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
    hessian = (design_matrix.T * w).dot(design_matrix)

    try:
        covariance = np.linalg.inv(hessian)
        # intercept uncertainty is not used
        return np.sqrt(np.diag(covariance))[1:]
    except np.linalg.LinAlgError:
        # degenerate case: treat coefficients as unreliable
        return np.full(X_scaled.shape[1], np.inf)
