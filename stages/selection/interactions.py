"""
Feature interaction detection stage.

Detects meaningful interaction terms between features that can improve
model predictive power. Tests multiplicative, ratio, and difference
interactions and selects those with significant AUC improvement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import warnings

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm

from core.base import PipelineStage, StageResult, InteractionConfig


@dataclass
class InteractionCandidate:
    """Information about a candidate interaction term."""
    feature1: str
    feature2: str
    interaction_type: str  # "multiplicative", "ratio", "difference"
    interaction_name: str
    correlation_between: float  # correlation between the two features
    train_auc_improvement: float
    test_auc_improvement: float
    pvalue: float
    coefficient: float
    selected: bool
    rejection_reason: Optional[str] = None


@dataclass
class InteractionInfo:
    """Comprehensive information about interaction detection results."""
    n_features_input: int
    n_candidates_tested: int
    n_interactions_selected: int
    interaction_types_tested: List[str]
    candidates: List[InteractionCandidate] = field(default_factory=list)
    selected_interactions: List[str] = field(default_factory=list)
    baseline_train_auc: float = 0.0
    baseline_test_auc: float = 0.0
    final_train_auc: float = 0.0
    final_test_auc: float = 0.0
    elapsed_seconds: float = 0.0


class InteractionDetectorStage(PipelineStage):
    """
    Stage for detecting and generating feature interactions.

    This stage searches for meaningful interaction terms between features
    that can improve model predictive power. It supports:
    - Multiplicative interactions (X1 * X2)
    - Ratio interactions (X1 / X2)
    - Difference interactions (X1 - X2)

    Interactions are evaluated by:
    1. Testing statistical significance (p-value)
    2. Measuring AUC improvement on test set
    3. Checking correlation to avoid redundancy
    """

    name = "InteractionDetectorStage"

    def __init__(self, config: Optional[InteractionConfig] = None):
        """
        Initialize InteractionDetectorStage.

        Args:
            config: InteractionConfig with detection parameters.
                   If None, uses defaults (disabled by default)
        """
        super().__init__(config or InteractionConfig())
        self._interaction_info: Optional[InteractionInfo] = None
        self._selected_interactions: List[str] = []
        self._interaction_formulas: Dict[str, Tuple[str, str, str]] = {}
        self._is_fitted = False

    def _generate_interaction_name(
        self,
        feat1: str,
        feat2: str,
        interaction_type: str
    ) -> str:
        """Generate a descriptive name for the interaction."""
        if interaction_type == "multiplicative":
            return f"{feat1}_X_{feat2}"
        elif interaction_type == "ratio":
            return f"{feat1}_DIV_{feat2}"
        elif interaction_type == "difference":
            return f"{feat1}_MINUS_{feat2}"
        return f"{feat1}_{interaction_type}_{feat2}"

    def _compute_interaction(
        self,
        data: pd.DataFrame,
        feat1: str,
        feat2: str,
        interaction_type: str
    ) -> pd.Series:
        """Compute the interaction term values."""
        if interaction_type == "multiplicative":
            return data[feat1] * data[feat2]
        elif interaction_type == "ratio":
            # safe division to avoid inf
            denominator = data[feat2].replace(0, np.nan)
            return data[feat1] / denominator
        elif interaction_type == "difference":
            return data[feat1] - data[feat2]
        raise ValueError(f"Unknown interaction type: {interaction_type}")

    def _get_candidate_pairs(
        self,
        features: List[str],
        data: pd.DataFrame
    ) -> List[Tuple[str, str]]:
        """Get pairs of features to test for interactions."""
        config = self.config

        # filter features based on config
        if config.candidate_features:
            features = [f for f in features if f in config.candidate_features]
        features = [f for f in features if f not in config.exclude_features]

        pairs = []
        corr_threshold = config.correlation_threshold

        # compute correlation matrix
        corr_matrix = data[features].corr()

        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                # only consider pairs with low correlation (avoid redundancy)
                corr = abs(corr_matrix.loc[f1, f2])
                if corr < corr_threshold:
                    pairs.append((f1, f2))

        # if test_all_pairs is False, limit to reasonable number
        if not config.test_all_pairs and len(pairs) > 50:
            # use heuristic: prioritize pairs with moderate correlation
            pair_scores = []
            for f1, f2 in pairs:
                corr = abs(corr_matrix.loc[f1, f2])
                # prefer pairs with some correlation but not too much
                score = 1 - abs(corr - 0.3)
                pair_scores.append((f1, f2, score))
            pair_scores.sort(key=lambda x: x[2], reverse=True)
            pairs = [(f1, f2) for f1, f2, _ in pair_scores[:50]]

        return pairs

    def _evaluate_interaction(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        interaction_name: str,
        baseline_features: List[str],
        baseline_train_auc: float,
        baseline_test_auc: float
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate a single interaction term.

        Returns:
            Tuple of (train_auc_improvement, test_auc_improvement, pvalue, coefficient)
        """
        # fit model with interaction
        features_with_interaction = baseline_features + [interaction_name]
        X_train_subset = X_train[features_with_interaction].dropna()
        y_train_subset = y_train.loc[X_train_subset.index]

        if len(X_train_subset) < 100:
            return 0.0, 0.0, 1.0, 0.0

        try:
            # fit statsmodels for p-value
            X_sm = sm.add_constant(X_train_subset)
            model_sm = sm.Logit(y_train_subset, X_sm)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result_sm = model_sm.fit(disp=0, maxiter=100)

            # get p-value for interaction term
            if interaction_name in result_sm.pvalues:
                pvalue = result_sm.pvalues[interaction_name]
                coefficient = result_sm.params[interaction_name]
            else:
                pvalue = 1.0
                coefficient = 0.0

            # fit sklearn for AUC
            model_sk = LogisticRegression(
                solver='lbfgs',
                max_iter=200,
                random_state=self.config.random_state
            )
            model_sk.fit(X_train_subset, y_train_subset)

            # calculate train AUC
            train_proba = model_sk.predict_proba(X_train_subset)[:, 1]
            train_auc = roc_auc_score(y_train_subset, train_proba)
            train_improvement = train_auc - baseline_train_auc

            # calculate test AUC
            X_test_subset = X_test[features_with_interaction].dropna()
            y_test_subset = y_test.loc[X_test_subset.index]

            if len(X_test_subset) > 0:
                test_proba = model_sk.predict_proba(X_test_subset)[:, 1]
                test_auc = roc_auc_score(y_test_subset, test_proba)
                test_improvement = test_auc - baseline_test_auc
            else:
                test_improvement = 0.0

            return train_improvement, test_improvement, pvalue, coefficient

        except Exception:
            return 0.0, 0.0, 1.0, 0.0

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'InteractionDetectorStage':
        """
        Fit interaction detection to find meaningful interactions.

        Args:
            data: DataFrame with features and target
            target: Name of target variable
            feature_columns: Features to consider. If None, auto-detects
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        start_time = time.time()
        config = self.config

        if not config.enabled:
            self._interaction_info = InteractionInfo(
                n_features_input=0,
                n_candidates_tested=0,
                n_interactions_selected=0,
                interaction_types_tested=[],
                elapsed_seconds=0.0
            )
            self._is_fitted = True
            return self

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

        # split by sample_type
        if 'sample_type' in data.columns:
            train_mask = data['sample_type'] == 0
            test_mask = data['sample_type'] == 1
        else:
            n_samples = len(data)
            train_size = int(0.8 * n_samples)
            np.random.seed(config.random_state)
            train_idx = np.random.choice(n_samples, size=train_size, replace=False)
            train_mask = pd.Series(False, index=data.index)
            train_mask.iloc[train_idx] = True
            test_mask = ~train_mask

        X_train = data.loc[train_mask, feature_columns].copy()
        y_train = data.loc[train_mask, target].copy()
        X_test = data.loc[test_mask, feature_columns].copy()
        y_test = data.loc[test_mask, target].copy()

        # calculate baseline AUC
        try:
            model_baseline = LogisticRegression(
                solver='lbfgs',
                max_iter=200,
                random_state=config.random_state
            )
            X_train_clean = X_train.dropna()
            y_train_clean = y_train.loc[X_train_clean.index]
            model_baseline.fit(X_train_clean, y_train_clean)

            train_proba = model_baseline.predict_proba(X_train_clean)[:, 1]
            baseline_train_auc = roc_auc_score(y_train_clean, train_proba)

            X_test_clean = X_test.dropna()
            y_test_clean = y_test.loc[X_test_clean.index]
            test_proba = model_baseline.predict_proba(X_test_clean)[:, 1]
            baseline_test_auc = roc_auc_score(y_test_clean, test_proba)
        except Exception:
            baseline_train_auc = 0.5
            baseline_test_auc = 0.5

        # get candidate pairs
        pairs = self._get_candidate_pairs(feature_columns, X_train)

        # compute correlation matrix for reporting
        corr_matrix = X_train[feature_columns].corr()

        candidates = []
        n_tested = 0

        # test each interaction type
        for interaction_type in config.interaction_types:
            for feat1, feat2 in pairs:
                interaction_name = self._generate_interaction_name(
                    feat1, feat2, interaction_type
                )

                # compute interaction values
                X_train[interaction_name] = self._compute_interaction(
                    X_train, feat1, feat2, interaction_type
                )
                X_test[interaction_name] = self._compute_interaction(
                    X_test, feat1, feat2, interaction_type
                )

                # skip if too many NaN
                if X_train[interaction_name].isna().mean() > 0.5:
                    X_train.drop(columns=[interaction_name], inplace=True)
                    X_test.drop(columns=[interaction_name], inplace=True)
                    continue

                n_tested += 1

                # evaluate interaction
                train_imp, test_imp, pvalue, coef = self._evaluate_interaction(
                    X_train, y_train, X_test, y_test,
                    interaction_name, feature_columns,
                    baseline_train_auc, baseline_test_auc
                )

                # determine if selected
                selected = (
                    pvalue < config.pvalue_threshold and
                    test_imp >= config.min_auc_improvement
                )

                rejection_reason = None
                if not selected:
                    if pvalue >= config.pvalue_threshold:
                        rejection_reason = f"p-value {pvalue:.4f} >= {config.pvalue_threshold}"
                    elif test_imp < config.min_auc_improvement:
                        rejection_reason = f"AUC improvement {test_imp:.4f} < {config.min_auc_improvement}"

                corr_between = abs(corr_matrix.loc[feat1, feat2])

                candidate = InteractionCandidate(
                    feature1=feat1,
                    feature2=feat2,
                    interaction_type=interaction_type,
                    interaction_name=interaction_name,
                    correlation_between=corr_between,
                    train_auc_improvement=train_imp,
                    test_auc_improvement=test_imp,
                    pvalue=pvalue,
                    coefficient=coef,
                    selected=selected,
                    rejection_reason=rejection_reason
                )
                candidates.append(candidate)

                # clean up if not selected
                if not selected:
                    X_train.drop(columns=[interaction_name], inplace=True)
                    X_test.drop(columns=[interaction_name], inplace=True)

        # select top interactions
        selected_candidates = [c for c in candidates if c.selected]
        selected_candidates.sort(key=lambda c: c.test_auc_improvement, reverse=True)
        selected_candidates = selected_candidates[:config.max_interactions]

        # update selection flags
        selected_names = {c.interaction_name for c in selected_candidates}
        for c in candidates:
            if c.selected and c.interaction_name not in selected_names:
                c.selected = False
                c.rejection_reason = "Exceeded max_interactions limit"

        self._selected_interactions = [c.interaction_name for c in selected_candidates]
        self._interaction_formulas = {
            c.interaction_name: (c.feature1, c.feature2, c.interaction_type)
            for c in selected_candidates
        }

        # calculate final AUC with selected interactions
        final_train_auc = baseline_train_auc
        final_test_auc = baseline_test_auc

        if self._selected_interactions:
            try:
                all_features = feature_columns + self._selected_interactions
                X_train_final = X_train[all_features].dropna()
                y_train_final = y_train.loc[X_train_final.index]

                model_final = LogisticRegression(
                    solver='lbfgs',
                    max_iter=200,
                    random_state=config.random_state
                )
                model_final.fit(X_train_final, y_train_final)

                train_proba = model_final.predict_proba(X_train_final)[:, 1]
                final_train_auc = roc_auc_score(y_train_final, train_proba)

                X_test_final = X_test[all_features].dropna()
                y_test_final = y_test.loc[X_test_final.index]
                test_proba = model_final.predict_proba(X_test_final)[:, 1]
                final_test_auc = roc_auc_score(y_test_final, test_proba)
            except Exception:
                pass

        elapsed = time.time() - start_time

        self._interaction_info = InteractionInfo(
            n_features_input=n_input,
            n_candidates_tested=n_tested,
            n_interactions_selected=len(self._selected_interactions),
            interaction_types_tested=config.interaction_types.copy(),
            candidates=candidates,
            selected_interactions=self._selected_interactions.copy(),
            baseline_train_auc=baseline_train_auc,
            baseline_test_auc=baseline_test_auc,
            final_train_auc=final_train_auc,
            final_test_auc=final_test_auc,
            elapsed_seconds=elapsed
        )

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data by adding selected interaction features.

        Args:
            data: DataFrame to transform
            **kwargs: Additional arguments

        Returns:
            DataFrame with interaction features added
        """
        if not self._is_fitted:
            raise RuntimeError("InteractionDetectorStage must be fitted before transform")

        if not self._selected_interactions:
            return data

        result = data.copy()

        for interaction_name, (feat1, feat2, interaction_type) in self._interaction_formulas.items():
            if feat1 in result.columns and feat2 in result.columns:
                result[interaction_name] = self._compute_interaction(
                    result, feat1, feat2, interaction_type
                )

        return result

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
            feature_columns: Features to consider
            **kwargs: Additional arguments

        Returns:
            StageResult with transformed data and logging
        """
        self.fit(data, target, feature_columns, **kwargs)
        transformed = self.transform(data, **kwargs)

        logs = self._build_logs()

        return StageResult(
            data=transformed,
            logs=logs,
            metadata={
                'selected_interactions': self._selected_interactions.copy(),
                'interaction_formulas': self._interaction_formulas.copy(),
                'n_interactions_added': len(self._selected_interactions)
            }
        )

    def _build_logs(self) -> Dict[str, Any]:
        """Build comprehensive logs for visualization."""
        if self._interaction_info is None:
            return {}

        info = self._interaction_info

        # candidate details
        candidate_details = []
        for c in info.candidates:
            candidate_details.append({
                'interaction_name': c.interaction_name,
                'feature1': c.feature1,
                'feature2': c.feature2,
                'interaction_type': c.interaction_type,
                'correlation_between': round(c.correlation_between, 4),
                'train_auc_improvement': round(c.train_auc_improvement, 4),
                'test_auc_improvement': round(c.test_auc_improvement, 4),
                'pvalue': round(c.pvalue, 4),
                'coefficient': round(c.coefficient, 4),
                'selected': c.selected,
                'rejection_reason': c.rejection_reason
            })

        return {
            'stage_name': 'InteractionDetectorStage',
            'summary': {
                'input_features': info.n_features_input,
                'candidates_tested': info.n_candidates_tested,
                'interactions_selected': info.n_interactions_selected,
                'interaction_types': info.interaction_types_tested,
                'baseline_train_auc': round(info.baseline_train_auc, 4),
                'baseline_test_auc': round(info.baseline_test_auc, 4),
                'final_train_auc': round(info.final_train_auc, 4),
                'final_test_auc': round(info.final_test_auc, 4),
                'auc_improvement': round(info.final_test_auc - info.baseline_test_auc, 4),
                'elapsed_seconds': round(info.elapsed_seconds, 3)
            },
            'candidates': candidate_details,
            'selected_interactions': info.selected_interactions
        }

    def get_interaction_info(self) -> Optional[InteractionInfo]:
        """Get detailed interaction detection information."""
        return self._interaction_info

    def get_selected_interactions(self) -> List[str]:
        """Get list of selected interaction feature names."""
        return self._selected_interactions.copy()

    def get_interaction_formulas(self) -> Dict[str, Tuple[str, str, str]]:
        """Get formulas for selected interactions.

        Returns:
            Dict mapping interaction name to (feature1, feature2, type)
        """
        return self._interaction_formulas.copy()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return stage-specific diagnostic information."""
        return self._build_logs()

    def validate(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate interaction detection results.

        Args:
            data: DataFrame to validate
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self._is_fitted:
            issues.append("InteractionDetectorStage has not been fitted")
            return False, issues

        # check that interaction columns exist in transformed data
        for interaction_name in self._selected_interactions:
            if interaction_name not in data.columns:
                issues.append(f"Missing interaction column: {interaction_name}")

        # check for reasonable AUC improvement
        if self._interaction_info and self._selected_interactions:
            improvement = (
                self._interaction_info.final_test_auc -
                self._interaction_info.baseline_test_auc
            )
            if improvement < 0:
                issues.append(
                    f"Interactions decreased test AUC by {abs(improvement):.4f}"
                )

        return len(issues) == 0, issues
