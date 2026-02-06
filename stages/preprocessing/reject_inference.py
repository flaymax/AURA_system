"""
Reject Inference Stage for handling selection bias in predictive modeling.

When building predictive models, we only observe outcomes for
samples that were approved/selected. This creates selection bias since rejected
samples may have different characteristics. Reject inference techniques attempt
to infer the likely outcomes for rejected samples to reduce this bias.

Supported methods:
- hard_cutoff: Assign bad label if application score is below threshold
- fuzzy: Use predicted probabilities as sample weights
- parceling: Distribute rejects across good/bad based on score distribution
- augmentation: Simple augmentation with assumed bad rate
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from core.base import PipelineStage, StageResult, RejectInferenceConfig


logger = logging.getLogger(__name__)


@dataclass
class RejectInferenceInfo:
    """Information about a single reject inference operation."""
    n_accepted: int
    n_rejected: int
    n_accepted_good: int
    n_accepted_bad: int
    n_inferred_good: int
    n_inferred_bad: int
    original_bad_rate: float  # among accepted
    inferred_bad_rate: float  # among rejected
    combined_bad_rate: float  # after augmentation
    method_used: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_accepted": self.n_accepted,
            "n_rejected": self.n_rejected,
            "n_accepted_good": self.n_accepted_good,
            "n_accepted_bad": self.n_accepted_bad,
            "n_inferred_good": self.n_inferred_good,
            "n_inferred_bad": self.n_inferred_bad,
            "original_bad_rate": round(self.original_bad_rate, 4),
            "inferred_bad_rate": round(self.inferred_bad_rate, 4),
            "combined_bad_rate": round(self.combined_bad_rate, 4),
            "method_used": self.method_used,
        }


@dataclass
class RejectInferenceLog:
    """Comprehensive log for reject inference stage."""
    stage_name: str = "reject_inference"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    config_used: Dict[str, Any] = field(default_factory=dict)
    inference_info: Optional[RejectInferenceInfo] = None
    score_distribution: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "config_used": self.config_used,
            "inference_info": self.inference_info.to_dict() if self.inference_info else None,
            "score_distribution": self.score_distribution,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class RejectInferenceStage(PipelineStage):
    """Stage for performing reject inference on application data.

    This stage handles the selection bias problem by inferring
    likely outcomes for rejected samples. It supports multiple inference
    methods and produces augmented training data.

    The stage requires:
    - A decision column indicating accept/reject
    - An application score column (original score used for decision)
    - For accepted applications: the actual target outcome

    Example:
        >>> config = RejectInferenceConfig(
        ...     enabled=True,
        ...     method="fuzzy",
        ...     decision_column="approved",
        ...     score_column="application_score"
        ... )
        >>> stage = RejectInferenceStage(config)
        >>> result = stage.fit_transform(data, target="default_flag")
        >>> augmented_data = result.data
    """

    name = "reject_inference"

    def __init__(self, config: Optional[RejectInferenceConfig] = None):
        """Initialize reject inference stage.

        Args:
            config: RejectInferenceConfig instance, or None for defaults
        """
        super().__init__(config)
        self.config: RejectInferenceConfig = config or RejectInferenceConfig()
        self._inference_info: Optional[RejectInferenceInfo] = None
        self._stage_log: Optional[RejectInferenceLog] = None
        self._inference_model: Optional[LogisticRegression] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> "RejectInferenceStage":
        """Fit the reject inference model on accepted applications.

        Trains a logistic regression model on accepted applications to
        predict the probability of default. This model is then used to
        infer outcomes for rejected applications.

        Args:
            X: DataFrame containing features, decision column, and score column
            y: Target series (only available for accepted applications)
            **kwargs: Additional arguments

        Returns:
            self: fitted stage
        """
        if not self.config.enabled:
            self._is_fitted = True
            return self

        self._stage_log = RejectInferenceLog(
            config_used={
                "method": self.config.method,
                "decision_column": self.config.decision_column,
                "score_column": self.config.score_column,
                "weight_rejects": self.config.weight_rejects,
            }
        )

        # Validate required columns
        if self.config.decision_column not in X.columns:
            raise ValueError(
                f"Decision column '{self.config.decision_column}' not found in data"
            )

        if self.config.score_column not in X.columns:
            raise ValueError(
                f"Score column '{self.config.score_column}' not found in data"
            )

        # Split accepted and rejected
        decision_col = X[self.config.decision_column]
        accepted_mask = decision_col == self.config.accept_value

        if accepted_mask.sum() == 0:
            raise ValueError("No accepted applications found in data")

        # For fuzzy and parceling methods, we need to train a model
        if self.config.method in ("fuzzy", "parceling"):
            # Get features for training (exclude decision and score columns)
            feature_cols = [
                col for col in X.columns
                if col not in [self.config.decision_column, self.config.score_column]
                and pd.api.types.is_numeric_dtype(X[col])
            ]

            if y is not None:
                X_accepted = X.loc[accepted_mask, feature_cols].fillna(0)
                y_accepted = y[accepted_mask]

                # Train simple logistic regression for inference
                self._inference_model = LogisticRegression(
                    penalty=None,
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=self.config.random_state
                )
                self._inference_model.fit(X_accepted, y_accepted)

        self._feature_names_in = list(X.columns)
        self._is_fitted = True
        return self

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Transform data by inferring outcomes for rejected applications.

        Args:
            X: DataFrame containing all applications
            y: Target series (with NaN for rejected applications)
            **kwargs: Additional arguments

        Returns:
            Tuple of (augmented_X, augmented_y, sample_weights)
        """
        if not self.config.enabled:
            return X, y, None

        self.check_is_fitted()

        decision_col = X[self.config.decision_column]
        accepted_mask = decision_col == self.config.accept_value
        rejected_mask = decision_col == self.config.reject_value

        n_accepted = accepted_mask.sum()
        n_rejected = rejected_mask.sum()

        if n_rejected == 0:
            self._stage_log.warnings.append("No rejected applications found")
            return X, y, None

        # Get accepted outcomes
        if y is not None:
            y_accepted = y[accepted_mask]
            n_accepted_bad = (y_accepted == 1).sum()
            n_accepted_good = (y_accepted == 0).sum()
            original_bad_rate = n_accepted_bad / n_accepted if n_accepted > 0 else 0
        else:
            n_accepted_bad = 0
            n_accepted_good = n_accepted
            original_bad_rate = 0

        # Perform inference based on method
        if self.config.method == "hard_cutoff":
            y_inferred, weights = self._infer_hard_cutoff(X, rejected_mask)
        elif self.config.method == "fuzzy":
            y_inferred, weights = self._infer_fuzzy(X, rejected_mask)
        elif self.config.method == "parceling":
            y_inferred, weights = self._infer_parceling(X, accepted_mask, rejected_mask, y)
        elif self.config.method == "augmentation":
            y_inferred, weights = self._infer_augmentation(X, rejected_mask)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        # Calculate inferred statistics
        n_inferred_bad = (y_inferred == 1).sum()
        n_inferred_good = (y_inferred == 0).sum()
        inferred_bad_rate = n_inferred_bad / n_rejected if n_rejected > 0 else 0

        # Create augmented target
        y_augmented = y.copy() if y is not None else pd.Series(index=X.index, dtype=float)
        y_augmented.loc[rejected_mask] = y_inferred

        # Create sample weights
        sample_weights = pd.Series(1.0, index=X.index)
        sample_weights.loc[rejected_mask] = self.config.weight_rejects

        # Apply fuzzy weights if applicable
        if weights is not None:
            sample_weights.loc[rejected_mask] *= weights

        # Calculate combined statistics
        total = n_accepted + n_rejected
        combined_bad_rate = (n_accepted_bad + n_inferred_bad) / total if total > 0 else 0

        # Store inference info
        self._inference_info = RejectInferenceInfo(
            n_accepted=n_accepted,
            n_rejected=n_rejected,
            n_accepted_good=n_accepted_good,
            n_accepted_bad=n_accepted_bad,
            n_inferred_good=n_inferred_good,
            n_inferred_bad=n_inferred_bad,
            original_bad_rate=original_bad_rate,
            inferred_bad_rate=inferred_bad_rate,
            combined_bad_rate=combined_bad_rate,
            method_used=self.config.method,
        )

        self._stage_log.inference_info = self._inference_info
        self._stage_log.completed_at = datetime.now()

        # Log summary
        self._log_info(
            f"Reject inference complete: {n_rejected} rejects inferred "
            f"({n_inferred_bad} bad, {n_inferred_good} good)"
        )
        self._log_info(
            f"Bad rates: accepted={original_bad_rate:.2%}, "
            f"inferred={inferred_bad_rate:.2%}, combined={combined_bad_rate:.2%}"
        )

        return X, y_augmented, sample_weights

    def _infer_hard_cutoff(
        self,
        X: pd.DataFrame,
        rejected_mask: pd.Series
    ) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Infer outcomes using hard cutoff on application score.

        Applications with score below threshold are labeled as bad.

        Args:
            X: DataFrame with all applications
            rejected_mask: Boolean mask for rejected applications

        Returns:
            Tuple of (inferred_labels, None)
        """
        scores = X.loc[rejected_mask, self.config.score_column]
        threshold = self.config.hard_cutoff_threshold

        # Lower score = higher risk = bad
        y_inferred = (scores < threshold).astype(int)

        return y_inferred, None

    def _infer_fuzzy(
        self,
        X: pd.DataFrame,
        rejected_mask: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Infer outcomes using fuzzy augmentation with predicted probabilities.

        Uses the fitted model to predict P(bad) for rejected applications.
        Labels are assigned probabilistically, and weights reflect confidence.

        Args:
            X: DataFrame with all applications
            rejected_mask: Boolean mask for rejected applications

        Returns:
            Tuple of (inferred_labels, confidence_weights)
        """
        if self._inference_model is None:
            raise RuntimeError("Inference model not fitted")

        # Get features for rejected applications
        feature_cols = [
            col for col in X.columns
            if col not in [self.config.decision_column, self.config.score_column]
            and pd.api.types.is_numeric_dtype(X[col])
        ]

        X_rejected = X.loc[rejected_mask, feature_cols].fillna(0)

        # Predict probabilities
        prob_bad = self._inference_model.predict_proba(X_rejected)[:, 1]

        # Assign labels based on probability
        rng = np.random.RandomState(self.config.random_state)
        random_vals = rng.random(len(prob_bad))
        y_inferred = (random_vals < prob_bad).astype(int)

        # Weights based on confidence (how far from 0.5)
        confidence = np.abs(prob_bad - 0.5) * 2  # Scale to 0-1
        weights = pd.Series(confidence, index=X_rejected.index)

        return pd.Series(y_inferred, index=X_rejected.index), weights

    def _infer_parceling(
        self,
        X: pd.DataFrame,
        accepted_mask: pd.Series,
        rejected_mask: pd.Series,
        y: pd.Series
    ) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Infer outcomes using parceling method.

        Divides applications into score bins and assigns bad rates based on
        the observed bad rate in each bin among accepted applications.

        Args:
            X: DataFrame with all applications
            accepted_mask: Boolean mask for accepted applications
            rejected_mask: Boolean mask for rejected applications
            y: Target series

        Returns:
            Tuple of (inferred_labels, None)
        """
        scores = X[self.config.score_column]
        n_bins = self.config.parceling_n_bins

        # Create bins based on accepted applications
        accepted_scores = scores[accepted_mask]
        bins = pd.qcut(accepted_scores, q=n_bins, duplicates='drop')
        bin_edges = bins.cat.categories

        # Calculate bad rate per bin among accepted
        y_accepted = y[accepted_mask]
        bin_bad_rates = {}

        for i, interval in enumerate(bin_edges):
            mask = (accepted_scores >= interval.left) & (accepted_scores <= interval.right)
            if mask.sum() > 0:
                bin_bad_rates[i] = y_accepted[mask].mean()
            else:
                bin_bad_rates[i] = 0.5  # Default if no data

        # Assign labels to rejected based on their bin's bad rate
        rejected_scores = scores[rejected_mask]
        y_inferred = pd.Series(0, index=rejected_mask[rejected_mask].index)

        rng = np.random.RandomState(self.config.random_state)

        for i, interval in enumerate(bin_edges):
            mask = (rejected_scores >= interval.left) & (rejected_scores <= interval.right)
            if mask.sum() > 0:
                bad_rate = bin_bad_rates.get(i, 0.5)
                random_vals = rng.random(mask.sum())
                y_inferred.loc[mask[mask].index] = (random_vals < bad_rate).astype(int)

        return y_inferred, None

    def _infer_augmentation(
        self,
        X: pd.DataFrame,
        rejected_mask: pd.Series
    ) -> Tuple[pd.Series, Optional[pd.Series]]:
        """Infer outcomes using simple augmentation with fixed bad rate.

        Assigns bad label to a fixed percentage of rejected applications.

        Args:
            X: DataFrame with all applications
            rejected_mask: Boolean mask for rejected applications

        Returns:
            Tuple of (inferred_labels, None)
        """
        n_rejected = rejected_mask.sum()
        bad_rate = self.config.augmentation_bad_rate

        rng = np.random.RandomState(self.config.random_state)
        random_vals = rng.random(n_rejected)
        y_inferred = (random_vals < bad_rate).astype(int)

        return pd.Series(y_inferred, index=rejected_mask[rejected_mask].index), None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **kwargs
    ) -> StageResult:
        """Fit and transform in one step.

        Args:
            X: Input DataFrame
            y: Target series
            **kwargs: Additional arguments

        Returns:
            StageResult with augmented data
        """
        self.fit(X, y, **kwargs)
        X_aug, y_aug, weights = self.transform(X, y, **kwargs)

        # Add weights to dataframe if present
        result_data = X_aug.copy()
        if weights is not None:
            result_data['_sample_weight'] = weights

        # Add augmented target
        result_data['_target_augmented'] = y_aug

        logs = self._build_logs()

        return StageResult(
            data=result_data,
            logs=logs,
            metadata={
                'inference_info': self._inference_info.to_dict() if self._inference_info else None,
                'method': self.config.method,
                'weight_column': '_sample_weight' if weights is not None else None,
                'target_column': '_target_augmented',
            }
        )

    def _build_logs(self) -> Dict[str, Any]:
        """Build comprehensive logs for visualization."""
        if self._stage_log is None:
            return {}

        return self._stage_log.to_dict()

    def get_inference_info(self) -> Optional[RejectInferenceInfo]:
        """Get detailed inference information."""
        return self._inference_info

    def get_stage_log(self) -> Optional[RejectInferenceLog]:
        """Get the comprehensive stage log."""
        return self._stage_log

    def validate(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, List[str]]:
        """Validate that data is suitable for reject inference.

        Args:
            data: DataFrame to validate
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if self.config.decision_column not in data.columns:
            issues.append(
                f"Missing decision column: '{self.config.decision_column}'"
            )

        if self.config.score_column not in data.columns:
            issues.append(
                f"Missing score column: '{self.config.score_column}'"
            )

        if self.config.decision_column in data.columns:
            decision_values = data[self.config.decision_column].unique()
            if self.config.accept_value not in decision_values:
                issues.append(
                    f"Accept value '{self.config.accept_value}' not found in decision column"
                )
            if self.config.reject_value not in decision_values:
                issues.append(
                    f"Reject value '{self.config.reject_value}' not found in decision column"
                )

        return len(issues) == 0, issues
