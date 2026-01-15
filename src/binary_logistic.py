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


def train_logistic_block(X_tr, X_te, y_tr, variables, fallback_tr, fallback_te):
    # empty model fallback
    if not variables:
        return None, fallback_tr, fallback_te, None

    # scaling is local to the current feature subset
    scaler = StandardScaler()
    Xtr_raw = X_tr[variables].values
    Xte_raw = X_te[variables].values

    scaler.fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    model = LogisticRegression(
        penalty="none",
        solver="lbfgs",
        max_iter=3000
    )

    model.fit(Xtr, y_tr)

    # probabilities only, no hard predictions anywhere
    p_tr = model.predict_proba(Xtr)[:, 1]
    p_te = model.predict_proba(Xte)[:, 1]

    return model, p_tr, p_te, Xtr


def iterative_logistic_selection(
        X_train, y_train,
        X_test, y_test,
        alpha=0.05,
        max_steps=200
):
    # pool of all candidates
    candidates = list(X_train.columns)

    # current state of the model
    active = []
    entrance_log = []

    # diagnostics
    auc_track = []
    auc_at_entrance = {}

    # baseline = constant prediction
    base_tr = np.full(len(y_train), y_train.mean())
    base_te = np.full(len(y_test),  y_test.mean())

    step = 0
    something_changed = True

    while something_changed and step < max_steps:
        something_changed = False
        step += 1

        base_model, p_tr_base, p_te_base, _ = train_logistic_block(
            X_train, X_test, y_train,
            active, base_tr, base_te
        )

        ll_base = binary_loglikelihood(p_tr_base, y_train)

        # ---- try expanding the model ----
        chosen_var = None
        best_auc_shift = 0

        for var in (v for v in candidates if v not in active):
            trial_vars = active + [var]

            model, p_tr, p_te, Xtr = train_logistic_block(
                X_train, X_test, y_train,
                trial_vars, base_tr, base_te
            )

            # likelihood ratio against current state
            ll_full = binary_loglikelihood(p_tr, y_train)
            lr_value = 2 * (ll_full - ll_base)
            p_lr = 1 - chi2.cdf(lr_value, 1)

            # Wald check for the new coefficient only
            se = coef_standard_errors(model, Xtr)
            z = model.coef_[0][-1] / se[-1]
            p_wald = 2 * (1 - norm.cdf(abs(z)))

            auc_gain = roc_auc_score(y_test, p_te) - roc_auc_score(y_test, p_te_base)

            # conservative inclusion rule
            if p_lr < alpha and p_wald < alpha and auc_gain > best_auc_shift:
                chosen_var = var
                best_auc_shift = auc_gain
                best_tr = p_tr
                best_te = p_te

        if chosen_var is not None:
            active.append(chosen_var)
            entrance_log.append(chosen_var)
            something_changed = True
            print(f"feature added: {chosen_var}")

            auc_at_entrance[chosen_var] = (
                roc_auc_score(y_train, best_tr),
                roc_auc_score(y_test,  best_te)
            )

        # ---- try shrinking the model ----
        for var in active.copy():
            reduced = [v for v in active if v != var]

            full_model, p_tr_full, _, Xtr_full = train_logistic_block(
                X_train, X_test, y_train,
                active, base_tr, base_te
            )

            ll_full = binary_loglikelihood(p_tr_full, y_train)

            if reduced:
                _, p_tr_red, _, _ = train_logistic_block(
                    X_train, X_test, y_train,
                    reduced, base_tr, base_te
                )
                ll_red = binary_loglikelihood(p_tr_red, y_train)
            else:
                ll_red = binary_loglikelihood(base_tr, y_train)

            lr_value = 2 * (ll_full - ll_red)
            p_lr = 1 - chi2.cdf(lr_value, 1)

            se = coef_standard_errors(full_model, Xtr_full)
            idx = active.index(var)
            z = full_model.coef_[0][idx] / se[idx]
            p_wald = 2 * (1 - norm.cdf(abs(z)))

            # remove if evidence is weak
            if p_wald > alpha or p_lr > alpha:
                print(f"feature removed: {var}")
                active.remove(var)
                something_changed = True

        # ---- snapshot of current quality ----
        _, p_tr_now, p_te_now, _ = train_logistic_block(
            X_train, X_test, y_train,
            active, base_tr, base_te
        )

        auc_track.append([
            len(active),
            roc_auc_score(y_train, p_tr_now),
            roc_auc_score(y_test,  p_te_now)
        ])

        if not something_changed:
            break

    auc_df = pd.DataFrame(
        auc_track,
        columns=["num_features", "train_auc", "test_auc"]
    )

    return entrance_log, auc_df, active, auc_at_entrance


def resolve_final_order(entrance_log, final_vars):
    last_seen = {v: i for i, v in enumerate(entrance_log)}
    return sorted(final_vars, key=lambda v: last_seen[v])


def visualize_auc_path(entrance_log, final_vars, auc_at_entrance):
    ordered = resolve_final_order(entrance_log, final_vars)

    x = np.arange(1, len(ordered) + 1)
    tr_auc = [auc_at_entrance[v][0] for v in ordered]
    te_auc = [auc_at_entrance[v][1] for v in ordered]

    plt.figure(figsize=(10, 5))
    plt.plot(x, tr_auc, marker="o", label="Train AUC")
    plt.plot(x, te_auc, marker="o", label="Test AUC")
    plt.xticks(x, ordered, rotation=45)
    plt.grid(True)
    plt.legend()
    plt.title("AUC evolution along the final feature path")
    plt.tight_layout()
    plt.show()
