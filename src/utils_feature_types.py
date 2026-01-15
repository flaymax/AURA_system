import pandas as pd
import numpy as np


def detect_feature_type(
        df: pd.DataFrame,
        feature: str,
        cat_threshold: int = 10,
        cat_ratio_threshold: float = 0.05
):
    """
    Infers feature type using cardinality, data type and value distribution.

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Column name to analyze.
    cat_threshold : int
        Maximum number of unique values to classify a feature as categorical.
    cat_ratio_threshold : float
        Maximum ratio of unique values to total non-null observations
        for numeric features to be treated as categorical.

    Returns
    -------
    dict with keys:
        - feature
        - dtype
        - n_unique
        - n_non_null
        - unique_ratio
        - feature_type:
            binary / categorical / continuous / id_like
    """

    x = df[feature]
    dtype = str(x.dtype)

    # basic stats
    n_non_null = x.notna().sum()
    n_unique = x.nunique(dropna=True)
    unique_ratio = n_unique / max(n_non_null, 1)

    # ---- binary detection (numeric or object) ----
    if n_unique == 2:
        feature_type = "binary"

    # ---- object / string-like features ----
    elif x.dtype == "object" or pd.api.types.is_string_dtype(x):
        feature_type = "categorical"

    # ---- numeric features with small discrete support ----
    elif n_unique <= cat_threshold:
        feature_type = "categorical"

    # ---- numeric but low-cardinality relative to sample size ----
    elif unique_ratio <= cat_ratio_threshold:
        feature_type = "categorical"

    # ---- suspicious high-cardinality numeric (IDs, hashes, etc.) ----
    elif unique_ratio > 0.9:
        feature_type = "id_like"

    # ---- default case ----
    else:
        feature_type = "continuous"

    return {
        "feature": feature,
        "dtype": dtype,
        "n_unique": int(n_unique),
        "n_non_null": int(n_non_null),
        "unique_ratio": round(unique_ratio, 4),
        "feature_type": feature_type
    }
