import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score


def woe_monotonicity_diagnostics(
        data: pd.DataFrame,
        feature_name: str,
        target_name: str,
        n_bins: int = 15
):
    """
    Visual diagnostic for feature monotonicity via WoE transformation.
    Produces WoE estimates with confidence intervals, linear fit and
    summary metrics (AUC, IV, R²).
    """

    work_df = data[[feature_name, target_name]].copy()

    # discretize feature into quantile-based buckets
    discretizer = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="quantile"
    )
    work_df["_bucket"] = discretizer.fit_transform(
        work_df[[feature_name]]
    ).astype(int)

    # aggregate event statistics per bucket
    stats = (
        work_df
        .groupby("_bucket")
        .agg(
            event_cnt=(target_name, lambda x: (x == 1).sum()),
            nonevent_cnt=(target_name, lambda x: (x == 0).sum()),
            bucket_mean=(feature_name, "mean")
        )
        .reset_index(drop=True)
    )

    # stabilize extreme bins
    stats["event_cnt"] = stats["event_cnt"].replace(0, 0.5)
    stats["nonevent_cnt"] = stats["nonevent_cnt"].replace(0, 0.5)

    total_events = stats["event_cnt"].sum()
    total_nonevents = stats["nonevent_cnt"].sum()

    # normalized distributions
    stats["p_event"] = stats["event_cnt"] / total_events
    stats["p_nonevent"] = stats["nonevent_cnt"] / total_nonevents

    # WoE & IV
    stats["woe"] = np.log(stats["p_event"] / stats["p_nonevent"])
    stats["iv_component"] = (stats["p_event"] - stats["p_nonevent"]) * stats["woe"]
    iv_score = stats["iv_component"].sum()

    # uncertainty estimation for WoE
    stats["woe_std"] = np.sqrt(
        1 / stats["event_cnt"] + 1 / stats["nonevent_cnt"]
    )
    stats["ci_lower"] = stats["woe"] - 1.96 * stats["woe_std"]
    stats["ci_upper"] = stats["woe"] + 1.96 * stats["woe_std"]

    # linear approximation: feature_mean -> WoE
    X = stats["bucket_mean"].values.reshape(-1, 1)
    y = stats["woe"].values

    lin_model = LinearRegression().fit(X, y)
    woe_fitted = lin_model.predict(X)
    r_squared = lin_model.score(X, y)

    # raw feature discrimination power
    try:
        auc_value = roc_auc_score(
            work_df[target_name],
            work_df[feature_name]
        )
    except ValueError:
        auc_value = np.nan

    # visualization
    plt.figure(figsize=(10, 5))

    plt.errorbar(
        stats["bucket_mean"],
        stats["woe"],
        yerr=[
            stats["woe"] - stats["ci_lower"],
            stats["ci_upper"] - stats["woe"]
        ],
        fmt="o",
        capsize=4,
        elinewidth=2,
        label="WoE with 95% CI"
    )

    plt.plot(
        stats["bucket_mean"],
        woe_fitted,
        linestyle="--",
        label="Linear trend"
    )

    plt.title(
        f"WoE diagnostic for '{feature_name}'\n"
        f"AUC={auc_value:.4f} | IV={iv_score:.4f} | R²={r_squared:.4f}"
    )
    plt.xlabel("Average feature value per bin")
    plt.ylabel("Weight of Evidence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return stats
