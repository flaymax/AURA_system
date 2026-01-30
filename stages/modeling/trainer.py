"""
Model training stage for logistic regression scorecard.

Trains the final logistic regression model and converts it to
a scorecard format with points-based scoring.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

from core.base import PipelineStage, StageResult, ModelConfig


@dataclass
class FeatureCoefficient:
    """Information about a single feature's coefficient."""
    feature_name: str
    coefficient: float
    scaled_coefficient: float  # coefficient after scaling
    std_error: Optional[float] = None
    z_score: Optional[float] = None
    pvalue: Optional[float] = None


@dataclass
class ScorecardFeature:
    """Scorecard points for a single feature."""
    feature_name: str
    base_points: float  # points contribution at mean value
    coefficient: float  # for calculating points from WoE
    # if binning info is available, points per bin
    bin_points: Optional[Dict[str, float]] = None


@dataclass
class ModelInfo:
    """Comprehensive information about the trained model."""
    n_features: int
    intercept: float
    # performance metrics
    train_auc: float
    test_auc: float
    train_gini: float
    test_gini: float
    train_ks: float  # Kolmogorov-Smirnov statistic
    test_ks: float
    # coefficients
    feature_coefficients: List[FeatureCoefficient] = field(default_factory=list)
    # scorecard params
    base_score: float = 600.0
    pdo: float = 20.0  # points to double odds
    base_odds: float = 50.0  # odds at base score
    # timing
    elapsed_seconds: float = 0.0


@dataclass
class ScorecardInfo:
    """Full scorecard definition."""
    base_score: float
    pdo: float
    base_odds: float
    # feature contributions
    features: List[ScorecardFeature] = field(default_factory=list)
    # score range
    min_possible_score: float = 0.0
    max_possible_score: float = 1000.0


class ModelTrainerStage(PipelineStage):
    """
    Stage for training logistic regression and building scorecard.

    Performs the following:
    1. Fits logistic regression on WoE-transformed features
    2. Calculates performance metrics (AUC, Gini, KS)
    3. Converts model to scorecard format (points-based)

    The scorecard uses standard industry formulas:
    Score = Base_Score + (Coefficient * WoE * Factor)
    where Factor = PDO / ln(2)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize ModelTrainerStage.

        Args:
            config: ModelConfig with scorecard parameters.
                   If None, uses defaults (base_score=600, pdo=20)
        """
        super().__init__(config or ModelConfig())
        self._model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._model_info: Optional[ModelInfo] = None
        self._scorecard_info: Optional[ScorecardInfo] = None
        self._feature_names: List[str] = []
        self._is_fitted = False

    def _calculate_ks(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.

        KS measures maximum separation between cumulative distributions
        of goods and bads. Higher is better.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities

        Returns:
            KS statistic (0 to 1)
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks = max(tpr - fpr)
        return ks

    def _calculate_standard_errors(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: LogisticRegression
    ) -> np.ndarray:
        """
        Calculate standard errors for coefficients using Hessian.

        Args:
            X: Feature matrix (scaled)
            y: Target vector
            model: Fitted logistic regression

        Returns:
            Array of standard errors for each coefficient
        """
        # predicted probabilities
        p = model.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # weights
        w = p * (1 - p)

        # design matrix with intercept
        X_design = np.column_stack([np.ones(len(X)), X])

        # Hessian
        hessian = (X_design.T * w).dot(X_design)

        try:
            cov = np.linalg.inv(hessian)
            se = np.sqrt(np.diag(cov))
            return se[1:]  # skip intercept
        except np.linalg.LinAlgError:
            return np.full(X.shape[1], np.inf)

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'ModelTrainerStage':
        """
        Fit logistic regression model.

        Args:
            data: DataFrame with WoE-transformed features and target
            target: Name of target variable
            feature_columns: Features to use. If None, auto-detects
            **kwargs: Additional arguments (binning_info for scorecard)

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

        self._feature_names = feature_columns

        # split by sample_type if available
        if 'sample_type' in data.columns:
            train_mask = data['sample_type'] == 0
            test_mask = data['sample_type'] == 1

            X_train = data.loc[train_mask, feature_columns].values
            y_train = data.loc[train_mask, target].values
            X_test = data.loc[test_mask, feature_columns].values
            y_test = data.loc[test_mask, target].values
        else:
            # fallback: random split
            n = len(data)
            idx = np.random.permutation(n)
            split = int(0.8 * n)

            train_idx = idx[:split]
            test_idx = idx[split:]

            X_train = data.iloc[train_idx][feature_columns].values
            y_train = data.iloc[train_idx][target].values
            X_test = data.iloc[test_idx][feature_columns].values
            y_test = data.iloc[test_idx][target].values

        # scale features for numerical stability
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        # fit logistic regression without regularization
        self._model = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=3000,
            random_state=42
        )
        self._model.fit(X_train_scaled, y_train)

        # predictions
        p_train = self._model.predict_proba(X_train_scaled)[:, 1]
        p_test = self._model.predict_proba(X_test_scaled)[:, 1]

        # metrics
        train_auc = roc_auc_score(y_train, p_train)
        test_auc = roc_auc_score(y_test, p_test)
        train_gini = 2 * train_auc - 1
        test_gini = 2 * test_auc - 1
        train_ks = self._calculate_ks(y_train, p_train)
        test_ks = self._calculate_ks(y_test, p_test)

        # standard errors
        se = self._calculate_standard_errors(X_train_scaled, y_train, self._model)

        # build coefficient info
        coefficients = self._model.coef_[0]
        feature_coefs = []
        from scipy import stats as scipy_stats

        for i, feat in enumerate(feature_columns):
            coef = coefficients[i]
            std_err = se[i] if i < len(se) else np.inf
            z = coef / std_err if std_err != np.inf else 0
            pval = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

            feature_coefs.append(FeatureCoefficient(
                feature_name=feat,
                coefficient=coef / self._scaler.scale_[i],  # unscaled
                scaled_coefficient=coef,
                std_error=std_err,
                z_score=z,
                pvalue=pval
            ))

        elapsed = time.time() - start_time

        # get scorecard params from config
        base_score = getattr(self.config, 'base_score', 600.0)
        pdo = getattr(self.config, 'pdo', 20.0)
        base_odds = getattr(self.config, 'base_odds', 50.0)

        self._model_info = ModelInfo(
            n_features=len(feature_columns),
            intercept=self._model.intercept_[0],
            train_auc=train_auc,
            test_auc=test_auc,
            train_gini=train_gini,
            test_gini=test_gini,
            train_ks=train_ks,
            test_ks=test_ks,
            feature_coefficients=feature_coefs,
            base_score=base_score,
            pdo=pdo,
            base_odds=base_odds,
            elapsed_seconds=elapsed
        )

        # build scorecard
        self._build_scorecard(kwargs.get('binning_info'))

        self._is_fitted = True
        return self

    def _build_scorecard(self, binning_info: Optional[Dict] = None):
        """
        Build scorecard from fitted model.

        Converts logistic regression coefficients to points using:
        Score = Base_Score - PDO/ln(2) * (Intercept + sum(Coef_i * WoE_i))

        Args:
            binning_info: Optional binning information for feature bins
        """
        if self._model is None or self._model_info is None:
            return

        info = self._model_info
        factor = info.pdo / np.log(2)
        offset = info.base_score - factor * np.log(info.base_odds)

        scorecard_features = []

        for fc in info.feature_coefficients:
            # unscaled coefficient for WoE transformation
            coef = fc.coefficient

            # points formula: -coef * factor * WoE
            # base points (at WoE=0) is 0, points vary with WoE
            sc_feat = ScorecardFeature(
                feature_name=fc.feature_name,
                base_points=0.0,  # contribution at WoE=0
                coefficient=-coef * factor
            )

            # if binning info provided, calculate points per bin
            if binning_info and fc.feature_name in binning_info:
                bin_info = binning_info[fc.feature_name]
                bin_points = {}
                for bin_name, woe in bin_info.get('woe_values', {}).items():
                    bin_points[bin_name] = round(-coef * factor * woe, 2)
                sc_feat.bin_points = bin_points

            scorecard_features.append(sc_feat)

        # calculate score contribution from intercept
        intercept_points = offset - factor * info.intercept

        self._scorecard_info = ScorecardInfo(
            base_score=intercept_points,
            pdo=info.pdo,
            base_odds=info.base_odds,
            features=scorecard_features
        )

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data by adding predicted probabilities and scores.

        Args:
            data: DataFrame with features
            **kwargs: Additional arguments

        Returns:
            DataFrame with predictions added
        """
        if not self._is_fitted:
            raise RuntimeError("ModelTrainerStage must be fitted before transform")

        result = data.copy()

        # get features
        X = data[self._feature_names].values
        X_scaled = self._scaler.transform(X)

        # predict probability
        prob = self._model.predict_proba(X_scaled)[:, 1]
        result['predicted_probability'] = prob

        # calculate score
        scores = self.predict_score(data)
        result['score'] = scores

        return result

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default.

        Args:
            data: DataFrame with features

        Returns:
            Array of probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")

        X = data[self._feature_names].values
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]

    def predict_score(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict scorecard score.

        Higher score = lower risk (convention in credit scoring).

        Args:
            data: DataFrame with features

        Returns:
            Array of scores
        """
        if not self._is_fitted or self._scorecard_info is None:
            raise RuntimeError("Model not fitted")

        info = self._model_info
        factor = info.pdo / np.log(2)

        # start with base score
        scores = np.full(len(data), self._scorecard_info.base_score)

        # add contribution from each feature
        for sc_feat in self._scorecard_info.features:
            woe_values = data[sc_feat.feature_name].values
            scores += sc_feat.coefficient * woe_values

        return scores

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
            feature_columns: Features to use
            **kwargs: Additional arguments

        Returns:
            StageResult with predictions and model info
        """
        self.fit(data, target, feature_columns, **kwargs)
        transformed = self.transform(data, **kwargs)

        logs = self._build_logs()

        return StageResult(
            data=transformed,
            logs=logs,
            metadata={
                'feature_names': self._feature_names.copy(),
                'train_auc': self._model_info.train_auc,
                'test_auc': self._model_info.test_auc,
                'train_gini': self._model_info.train_gini,
                'test_gini': self._model_info.test_gini
            }
        )

    def _build_logs(self) -> Dict[str, Any]:
        """Build comprehensive logs for visualization."""
        if self._model_info is None:
            return {}

        info = self._model_info

        # coefficient details
        coef_details = []
        for fc in info.feature_coefficients:
            coef_details.append({
                'feature': fc.feature_name,
                'coefficient': round(fc.coefficient, 6),
                'scaled_coefficient': round(fc.scaled_coefficient, 6),
                'std_error': round(fc.std_error, 6) if fc.std_error != np.inf else None,
                'z_score': round(fc.z_score, 4) if fc.z_score else None,
                'pvalue': round(fc.pvalue, 6) if fc.pvalue else None
            })

        # scorecard details
        scorecard_details = []
        if self._scorecard_info:
            for sf in self._scorecard_info.features:
                scorecard_details.append({
                    'feature': sf.feature_name,
                    'coefficient_factor': round(sf.coefficient, 4),
                    'bin_points': sf.bin_points
                })

        return {
            'stage_name': 'ModelTrainerStage',
            'summary': {
                'n_features': info.n_features,
                'intercept': round(info.intercept, 6),
                'train_auc': round(info.train_auc, 4),
                'test_auc': round(info.test_auc, 4),
                'train_gini': round(info.train_gini, 4),
                'test_gini': round(info.test_gini, 4),
                'train_ks': round(info.train_ks, 4),
                'test_ks': round(info.test_ks, 4),
                'elapsed_seconds': round(info.elapsed_seconds, 3)
            },
            'scorecard_params': {
                'base_score': info.base_score,
                'pdo': info.pdo,
                'base_odds': info.base_odds
            },
            'coefficients': coef_details,
            'scorecard': scorecard_details
        }

    def get_model(self) -> Optional[LogisticRegression]:
        """Get the fitted sklearn model."""
        return self._model

    def get_model_info(self) -> Optional[ModelInfo]:
        """Get detailed model information."""
        return self._model_info

    def get_scorecard_info(self) -> Optional[ScorecardInfo]:
        """Get scorecard definition."""
        return self._scorecard_info

    def get_feature_names(self) -> List[str]:
        """Get list of features used in model."""
        return self._feature_names.copy()

    def to_sql(self) -> str:
        """
        Generate SQL for scoring.

        Returns:
            SQL CASE statement for calculating scores
        """
        if not self._is_fitted or self._scorecard_info is None:
            raise RuntimeError("Model not fitted")

        lines = [f"-- Scorecard SQL"]
        lines.append(f"-- Base score: {self._scorecard_info.base_score:.2f}")
        lines.append(f"SELECT")
        lines.append(f"  {self._scorecard_info.base_score:.4f}")

        for sf in self._scorecard_info.features:
            lines.append(f"  + ({sf.coefficient:.6f} * {sf.feature_name})")

        lines.append("AS score")

        return "\n".join(lines)

    def validate(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate model quality.

        Args:
            data: DataFrame to validate
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self._is_fitted:
            issues.append("ModelTrainerStage has not been fitted")
            return False, issues

        if self._model_info is None:
            issues.append("Model info not available")
            return False, issues

        info = self._model_info

        # check AUC
        if info.test_auc < 0.5:
            issues.append(f"Test AUC below 0.5: {info.test_auc:.4f}")

        if info.test_auc < 0.6:
            issues.append(f"Test AUC below 0.6 (weak model): {info.test_auc:.4f}")

        # check overfitting
        auc_gap = info.train_auc - info.test_auc
        if auc_gap > 0.05:
            issues.append(
                f"Potential overfitting: train AUC {info.train_auc:.4f}, "
                f"test AUC {info.test_auc:.4f} (gap: {auc_gap:.4f})"
            )

        # check KS
        if info.test_ks < 0.2:
            issues.append(f"Test KS below 0.2 (weak separation): {info.test_ks:.4f}")

        return len(issues) == 0, issues
