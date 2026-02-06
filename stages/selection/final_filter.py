"""
Final filtering stage for feature validation.

Performs final checks on features before model training:
- P-value significance in logistic regression
- VIF (Variance Inflation Factor) for multicollinearity
- Coefficient sign consistency
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np
from scipy import stats

from core.base import PipelineStage, StageResult


@dataclass
class FinalFilterConfig:
    """Configuration for final filtering stage."""
    max_pvalue: float = 0.05  # maximum allowed p-value
    max_vif: float = 5.0  # maximum allowed VIF (10 is common, 5 is strict)
    check_signs: bool = True  # check if coefficients have expected signs
    expected_signs: Optional[Dict[str, int]] = None  # feature -> expected sign (+1 or -1)
    remove_on_pvalue: bool = True
    remove_on_vif: bool = True
    iterative_vif: bool = True  # remove features one at a time by VIF


@dataclass
class FeatureFilterInfo:
    """Information about filtering decisions for a single feature."""
    feature_name: str
    pvalue: float
    vif: float
    coefficient: float
    # filtering decisions
    passed_pvalue: bool
    passed_vif: bool
    passed_sign: bool
    # final status
    kept: bool
    removal_reason: Optional[str] = None


@dataclass
class FinalFilterInfo:
    """Comprehensive information about final filtering."""
    n_input_features: int
    n_output_features: int
    max_pvalue_threshold: float
    max_vif_threshold: float
    # feature details
    feature_details: List[FeatureFilterInfo] = field(default_factory=list)
    # summary counts
    removed_by_pvalue: int = 0
    removed_by_vif: int = 0
    removed_by_sign: int = 0
    # timing
    elapsed_seconds: float = 0.0


class FinalFilterStage(PipelineStage):
    """
    Stage for final feature validation before model training.

    Performs the following checks:
    1. P-value significance - removes features with p > threshold
    2. VIF check - removes features causing multicollinearity
    3. Sign check - optionally validates coefficient signs

    VIF can be checked iteratively (remove worst offender, recalculate)
    or all at once.
    """

    def __init__(self, config: Optional[FinalFilterConfig] = None):
        """
        Initialize FinalFilterStage.

        Args:
            config: FinalFilterConfig with thresholds and options.
                   If None, uses defaults (pvalue=0.05, vif=5.0)
        """
        super().__init__(config or FinalFilterConfig())
        self._filter_info: Optional[FinalFilterInfo] = None
        self._selected_features: List[str] = []
        self._feature_stats: Dict[str, Dict] = {}
        self._is_fitted = False

    def _calculate_vif(
        self,
        X: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, float]:
        """
        Calculate VIF for each feature.

        VIF measures how much the variance of a coefficient is inflated
        due to correlation with other predictors. VIF > 5-10 indicates
        problematic multicollinearity.

        Args:
            X: Feature DataFrame
            features: List of feature names to check

        Returns:
            Dictionary mapping feature name to VIF value
        """
        vif_values = {}

        for feat in features:
            if len(features) < 2:
                # cannot calculate VIF with a single feature
                vif_values[feat] = 1.0
                continue

            other_feats = [f for f in features if f != feat]

            # regress this feature against all others
            y_temp = X[feat].values
            X_temp = X[other_feats].values

            # handle case where X might be constant or near-constant
            if np.std(y_temp) < 1e-10:
                vif_values[feat] = 1.0
                continue

            # add intercept
            X_temp = np.column_stack([np.ones(len(X_temp)), X_temp])

            try:
                # OLS regression
                coeffs, residuals, rank, s = np.linalg.lstsq(X_temp, y_temp, rcond=None)

                # calculate R-squared
                ss_res = np.sum((y_temp - X_temp @ coeffs) ** 2)
                ss_tot = np.sum((y_temp - np.mean(y_temp)) ** 2)

                if ss_tot < 1e-10:
                    r_squared = 0
                else:
                    r_squared = 1 - (ss_res / ss_tot)

                # VIF = 1 / (1 - R^2)
                if r_squared >= 1:
                    vif_values[feat] = float('inf')
                else:
                    vif_values[feat] = 1 / (1 - r_squared)

            except Exception:
                # fallback if regression fails
                vif_values[feat] = float('inf')

        return vif_values

    def _calculate_logistic_stats(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate p-values and coefficients from logistic regression.

        Uses Wald test for significance of each coefficient.

        Args:
            X: Feature DataFrame
            y: Target series
            features: List of feature names

        Returns:
            Tuple of (pvalues dict, coefficients dict)
        """
        from sklearn.linear_model import LogisticRegression

        if len(features) == 0:
            return {}, {}

        X_arr = X[features].values

        # fit logistic regression
        model = LogisticRegression(
            penalty=None,  # no regularization for unbiased p-values
            solver='lbfgs',
            max_iter=3000
        )

        try:
            model.fit(X_arr, y)
        except Exception as e:
            # if fitting fails, return large p-values
            return {f: 1.0 for f in features}, {f: 0.0 for f in features}

        # get coefficients
        coeffs = model.coef_[0]
        coef_dict = {f: c for f, c in zip(features, coeffs)}

        # calculate standard errors using Hessian
        # predicted probabilities
        p = model.predict_proba(X_arr)[:, 1]
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # weights for weighted least squares interpretation
        w = p * (1 - p)

        # design matrix with intercept
        X_design = np.column_stack([np.ones(len(X_arr)), X_arr])

        # Hessian approximation
        hessian = (X_design.T * w).dot(X_design)

        try:
            cov_matrix = np.linalg.inv(hessian)
            se = np.sqrt(np.diag(cov_matrix))[1:]  # skip intercept
        except np.linalg.LinAlgError:
            # singular matrix, coefficients are unreliable
            se = np.full(len(features), np.inf)

        # Wald statistic and p-values
        z_scores = coeffs / se
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

        pvalue_dict = {f: p for f, p in zip(features, pvalues)}

        return pvalue_dict, coef_dict

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'FinalFilterStage':
        """
        Fit the final filter to identify features to keep.

        Args:
            data: DataFrame with features and target
            target: Name of target variable
            feature_columns: Features to check. If None, auto-detects
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        start_time = time.time()

        # determine feature columns
        if feature_columns is None:
            exclude_cols = [target]
            if 'sample_type' in data.columns:
                exclude_cols.append('sample_type')
            feature_columns = [
                col for col in data.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
            ]

        n_input = len(feature_columns)

        # use training data only if sample_type exists
        if 'sample_type' in data.columns:
            train_data = data[data['sample_type'] == 0]
        else:
            train_data = data

        X = train_data[feature_columns]
        y = train_data[target]

        # calculate initial stats
        pvalues, coefficients = self._calculate_logistic_stats(X, y, feature_columns)

        # track features to keep
        current_features = feature_columns.copy()
        feature_details = []

        # step 1: filter by p-value
        removed_pvalue = 0
        if self.config.remove_on_pvalue:
            features_to_remove = []
            for feat in current_features:
                if pvalues.get(feat, 1.0) > self.config.max_pvalue:
                    features_to_remove.append(feat)
                    removed_pvalue += 1

            for feat in features_to_remove:
                current_features.remove(feat)

        # recalculate stats after p-value removal (if any were removed)
        if removed_pvalue > 0 and len(current_features) > 0:
            X_filtered = train_data[current_features]
            pvalues, coefficients = self._calculate_logistic_stats(
                X_filtered, y, current_features
            )

        # step 2: iterative VIF filtering
        removed_vif = 0
        if self.config.remove_on_vif and len(current_features) > 1:
            if self.config.iterative_vif:
                # remove worst offender iteratively
                while True:
                    vif_values = self._calculate_vif(
                        train_data[current_features],
                        current_features
                    )

                    # find worst VIF
                    max_vif_feat = max(vif_values, key=vif_values.get)
                    max_vif = vif_values[max_vif_feat]

                    if max_vif <= self.config.max_vif:
                        break

                    current_features.remove(max_vif_feat)
                    removed_vif += 1

                    if len(current_features) <= 1:
                        break
            else:
                # remove all at once
                vif_values = self._calculate_vif(
                    train_data[current_features],
                    current_features
                )
                features_to_remove = [
                    f for f, v in vif_values.items()
                    if v > self.config.max_vif
                ]
                for feat in features_to_remove:
                    current_features.remove(feat)
                    removed_vif += 1

        # final stats after all filtering
        if len(current_features) > 0:
            final_pvalues, final_coeffs = self._calculate_logistic_stats(
                train_data[current_features], y, current_features
            )
            final_vif = self._calculate_vif(
                train_data[current_features], current_features
            )
        else:
            final_pvalues, final_coeffs, final_vif = {}, {}, {}

        # build detailed info for each original feature
        removed_sign = 0
        for feat in feature_columns:
            pval = pvalues.get(feat, final_pvalues.get(feat, 1.0))
            vif = final_vif.get(feat, 999.0)
            coef = coefficients.get(feat, final_coeffs.get(feat, 0.0))

            passed_pvalue = pval <= self.config.max_pvalue
            passed_vif = vif <= self.config.max_vif

            # sign check
            passed_sign = True
            if self.config.check_signs and self.config.expected_signs:
                expected = self.config.expected_signs.get(feat)
                if expected is not None:
                    actual_sign = 1 if coef > 0 else -1
                    passed_sign = (actual_sign == expected)
                    if not passed_sign and feat in current_features:
                        current_features.remove(feat)
                        removed_sign += 1

            kept = feat in current_features

            # determine removal reason
            removal_reason = None
            if not kept:
                reasons = []
                if not passed_pvalue:
                    reasons.append(f"p-value {pval:.4f} > {self.config.max_pvalue}")
                if not passed_vif:
                    reasons.append(f"VIF {vif:.2f} > {self.config.max_vif}")
                if not passed_sign:
                    reasons.append("Coefficient sign mismatch")
                removal_reason = "; ".join(reasons) if reasons else "Unknown"

            feature_details.append(FeatureFilterInfo(
                feature_name=feat,
                pvalue=pval,
                vif=vif if feat in final_vif else float('nan'),
                coefficient=coef,
                passed_pvalue=passed_pvalue,
                passed_vif=passed_vif,
                passed_sign=passed_sign,
                kept=kept,
                removal_reason=removal_reason
            ))

        elapsed = time.time() - start_time

        self._filter_info = FinalFilterInfo(
            n_input_features=n_input,
            n_output_features=len(current_features),
            max_pvalue_threshold=self.config.max_pvalue,
            max_vif_threshold=self.config.max_vif,
            feature_details=feature_details,
            removed_by_pvalue=removed_pvalue,
            removed_by_vif=removed_vif,
            removed_by_sign=removed_sign,
            elapsed_seconds=elapsed
        )

        self._selected_features = current_features
        self._feature_stats = {
            feat: {
                'pvalue': final_pvalues.get(feat),
                'vif': final_vif.get(feat),
                'coefficient': final_coeffs.get(feat)
            }
            for feat in current_features
        }
        self._is_fitted = True

        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data by keeping only validated features.

        Args:
            data: DataFrame to transform
            **kwargs: Additional arguments

        Returns:
            DataFrame with only validated features
        """
        if not self._is_fitted:
            raise RuntimeError("FinalFilterStage must be fitted before transform")

        cols_to_keep = [col for col in self._selected_features if col in data.columns]

        # preserve special columns
        target = kwargs.get('target')
        for col in data.columns:
            if col not in cols_to_keep:
                if col in ['sample_type'] or col == target:
                    cols_to_keep.append(col)

        return data[cols_to_keep]

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> StageResult:
        """
        Fit and transform in one step.

        Args:
            data: Input DataFrame
            target: Target variable name
            feature_columns: Features to check
            **kwargs: Additional arguments

        Returns:
            StageResult with validated data and logs
        """
        self.fit(data, target, feature_columns, **kwargs)
        transformed = self.transform(data, target=target, **kwargs)

        logs = self._build_logs()

        return StageResult(
            data=transformed,
            logs=logs,
            metadata={
                'selected_features': self._selected_features.copy(),
                'feature_stats': self._feature_stats.copy()
            }
        )

    def _build_logs(self) -> Dict[str, Any]:
        """Build comprehensive logs for visualization."""
        if self._filter_info is None:
            return {}

        info = self._filter_info

        # detailed feature info
        feature_details = []
        for fd in info.feature_details:
            feature_details.append({
                'feature': fd.feature_name,
                'pvalue': round(fd.pvalue, 6) if not np.isnan(fd.pvalue) else None,
                'vif': round(fd.vif, 3) if not np.isnan(fd.vif) else None,
                'coefficient': round(fd.coefficient, 6),
                'passed_pvalue': fd.passed_pvalue,
                'passed_vif': fd.passed_vif,
                'passed_sign': fd.passed_sign,
                'kept': fd.kept,
                'removal_reason': fd.removal_reason
            })

        # separate kept and removed
        kept_features = [fd for fd in feature_details if fd['kept']]
        removed_features = [fd for fd in feature_details if not fd['kept']]

        return {
            'stage_name': 'FinalFilterStage',
            'summary': {
                'input_features': info.n_input_features,
                'output_features': info.n_output_features,
                'max_pvalue_threshold': info.max_pvalue_threshold,
                'max_vif_threshold': info.max_vif_threshold,
                'removed_by_pvalue': info.removed_by_pvalue,
                'removed_by_vif': info.removed_by_vif,
                'removed_by_sign': info.removed_by_sign,
                'elapsed_seconds': round(info.elapsed_seconds, 3)
            },
            'kept_features': kept_features,
            'removed_features': removed_features,
            'all_features': feature_details
        }

    def get_filter_info(self) -> Optional[FinalFilterInfo]:
        """Get detailed filter information."""
        return self._filter_info

    def get_selected_features(self) -> List[str]:
        """Get list of features that passed all checks."""
        return self._selected_features.copy()

    def get_feature_stats(self) -> Dict[str, Dict]:
        """Get final stats for selected features."""
        return self._feature_stats.copy()

    def validate(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate that filtering produced reasonable results.

        Args:
            data: DataFrame to validate
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self._is_fitted:
            issues.append("FinalFilterStage has not been fitted")
            return False, issues

        if len(self._selected_features) == 0:
            issues.append("No features passed final filtering")
            return False, issues

        # check if we removed too many features
        if self._filter_info:
            retention_ratio = (
                self._filter_info.n_output_features /
                max(self._filter_info.n_input_features, 1)
            )
            if retention_ratio < 0.2:
                issues.append(
                    f"Final filter removed too many features: "
                    f"{self._filter_info.n_input_features} -> "
                    f"{self._filter_info.n_output_features}"
                )

        return len(issues) == 0, issues
