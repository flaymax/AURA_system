# src/

Low-level algorithms. This is where the math lives.

## Files

### binary_logistic.py
Custom logistic regression utilities that sklearn doesn't give you out of the box.

**`binary_loglikelihood(probabilities, target)`**
Log-likelihood for binary outcomes. Used for likelihood ratio tests.

**`coef_standard_errors(model, X_scaled)`**
Computes standard errors from the Hessian. Sklearn doesn't expose this, but you need it for p-values and Wald tests.

**`train_logistic_block(X_tr, X_te, y_tr, variables, ...)`**
Fits logistic regression on a subset of features. Returns model + predictions. Handles scaling internally.

**`iterative_logistic_selection(...)`**
The stepwise selection engine. Forward-backward with:
- Likelihood ratio test for entry
- Wald test for removal
- Tracks AUC at each step

This is the workhorse behind `StepwiseSelectionStage`.

### utils_feature_types.py
**`detect_feature_type(df, feature, ...)`**
Figures out if a column is binary, categorical, continuous, or ID-like. Uses:
- Cardinality (unique values count)
- Unique ratio (uniques / total)
- Data type

Rules:
- 2 unique values → binary
- Object/string → categorical
- Few unique values → categorical
- High unique ratio → probably an ID, skip it
- Everything else → continuous

## Notes

The stepwise algo uses chi2 for LR test (df=1 for single feature). P-value from `scipy.stats.chi2.sf()`.

Standard errors come from inverse of the information matrix (Hessian). Can fail if matrix is singular - returns inf in that case.
