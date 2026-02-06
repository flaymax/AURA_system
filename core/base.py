"""
Base classes for the reasoning pipeline.

This module provides abstract base class for all pipeline stages,
configuration dataclasses and result containers. All stages should
inherit from PipelineStage and implement required methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging
import pickle
import json

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class StageStatus(Enum):
    """Status of pipeline stage execution.

    Used to track whether stage was fitted successfully or failed.
    """
    NOT_FITTED = "not_fitted"
    FITTED = "fitted"
    FAILED = "failed"
    SKIPPED = "skipped"  # when stage is disabled in config


class BinnerOptimization(Enum):
    """Optimization mode for WoE binner.

    IV - maximizes Information Value (better for predictive power)
    R2 - maximizes R-squared (better for linear relationship)
    """
    IV = "IV"
    R2 = "R2"


class SelectionMethod(Enum):
    """Method for stepwise feature selection."""
    FORWARD = "forward"
    BACKWARD = "backward"
    HYBRID = "hybrid"  # forward with backward elimination steps


class ImputationStrategy(Enum):
    """Strategy for handling missing values.

    Note: for WoE-based models we usually keep missings as seperate bin,
    but sometimes client wants to impute them beforehand.
    """
    NONE = "none"  # keep as is (WoE binner handles NaN)
    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    CONSTANT = "constant"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing stage.

    Attributes:
        remove_duplicates: whether to drop duplicate rows
        remove_constant_cols: drop columns with single unique value
        min_non_null_ratio: minimum ratio of non-null values to keep column
        id_columns: list of columns to exclude from modeling (like customer_id)
        target_col: name of target variable
        sample_type_col: column indicating train/test/validation split
        date_col: optional date column for temporal analysis
    """
    remove_duplicates: bool = True
    remove_constant_cols: bool = True
    min_non_null_ratio: float = 0.05  # drop cols with >95% missing
    id_columns: List[str] = field(default_factory=list)
    target_col: str = "target"
    sample_type_col: str = "sample_type"
    date_col: Optional[str] = None

    def __post_init__(self):
        # validate ratio is between 0 and 1
        if not 0.0 <= self.min_non_null_ratio <= 1.0:
            raise ValueError(
                f"min_non_null_ratio must be between 0 and 1, got {self.min_non_null_ratio}"
            )


@dataclass
class TypeDetectionConfig:
    """Configuration for feature type detection.

    Distinguishes between numeric and categorical features
    because they require different binning strategies.
    """
    cardinality_threshold: int = 10  # below this -> categorical
    unique_ratio_threshold: float = 0.05  # above this -> likely ID column
    force_numeric: List[str] = field(default_factory=list)
    force_categorical: List[str] = field(default_factory=list)


@dataclass
class ImputationConfig:
    """Configuration for missing value imputation.

    Usually not needed for WoE models since the binner handles NaN values,
    but this option is here for flexibility.
    """
    strategy: ImputationStrategy = ImputationStrategy.NONE
    constant_value: Optional[float] = None  # used when strategy is CONSTANT
    per_feature_strategy: Dict[str, ImputationStrategy] = field(default_factory=dict)


@dataclass
class BinningConfig:
    """Configuration for WoE binning stage.

    Attributes:
        optimization_mode: IV or R2 optimization
        power: tree depth for decision tree based binning (higher = more bins)
        min_bin_size: minimum fraction of samples in each bin
        max_bins: maximum number of bins per feature
        monotonic: whether to enforce monotonic WoE relationship
        min_iv: minimum IV to keep feature after binning
        handle_missing: create separate bin for missing values
    """
    optimization_mode: BinnerOptimization = BinnerOptimization.IV
    power: int = 3
    min_bin_size: float = 0.05
    max_bins: int = 10
    monotonic: bool = True
    min_iv: float = 0.02  # features with IV < 0.02 are usually weak
    handle_missing: bool = True

    def __post_init__(self):
        if self.power < 1:
            raise ValueError(f"power must be >= 1, got {self.power}")
        if self.min_iv < 0:
            raise ValueError(f"min_iv cannot be negative")


@dataclass
class PostprocessingConfig:
    """Configuration for post-binning validation.

    After WoE transformation we need to check that everything
    looks reasonable before proceeding.
    """
    check_woe_range: bool = True
    max_woe_abs: float = 5.0  # WoE values outside [-5, 5] are suspicious
    check_monotonicity: bool = True
    min_iv_threshold: float = 0.02
    drop_low_iv: bool = True  # automatically remove features with low IV


@dataclass
class ClusteringConfig:
    """Configuration for feature clustering stage.

    Groups correlated features and selects best representative
    from each cluster to reduce multicolinearity. Optionally filters
    selected features by statistical significance using logistic regression.

    Attributes:
        correlation_threshold: features with |corr| > threshold are clustered
        linkage_method: hierarchical clustering linkage method
        selection_type: how to pick representative from cluster
        distance_metric: how to compute feature distance (usually 1 - |corr|)
        min_cluster_size: minimum number of features to form a cluster
        pvalue_filter_enabled: whether to filter features by p-value after clustering
        pvalue_threshold: maximum p-value to keep a feature (default 0.05)
    """
    correlation_threshold: float = 0.7
    linkage_method: str = "average"  # average, complete, ward
    selection_type: str = "max_train"  # max_train, max_test, closest_train_test, center_cluster
    distance_metric: str = "correlation"
    min_cluster_size: int = 1
    pvalue_filter_enabled: bool = True
    pvalue_threshold: float = 0.05


@dataclass
class SelectionConfig:
    """Configuration for stepwise feature selection.

    Controls how forward/backward selection works.
    Alpha thresholds determine entry and exit criteria.

    Early stopping helps prevent overfitting by monitoring test AUC:
    - patience: stop if no improvement for N consecutive steps
    - min_improvement: minimum AUC gain to count as improvement
    - restore_best: restore model to best test AUC point

    Attributes:
        method: selection method (forward, backward, both)
        alpha_enter: p-value threshold to enter model
        alpha_exit: p-value threshold to exit (for backward steps)
        max_features: limit number of features (None = no limit)
        min_features: always keep at least this many
        use_wald_test: use Wald test for coefficient significance
        use_lr_test: use likelihood ratio test for feature addition
        early_stopping_enabled: whether to use early stopping
        patience: number of steps without improvement before stopping
        min_improvement: minimum AUC improvement to count as progress
        restore_best: restore to best test AUC point when stopping
        monitor_metric: which metric to monitor ("test_auc", "train_auc", "gap")
    """
    method: SelectionMethod = SelectionMethod.FORWARD
    alpha_enter: float = 0.05  # p-value threshold to enter model
    alpha_exit: float = 0.10  # p-value threshold to exit (for backward steps)
    max_features: Optional[int] = None  # limit number of features
    min_features: int = 3  # always keep at least this many
    use_wald_test: bool = True
    use_lr_test: bool = True
    # Early stopping options
    early_stopping_enabled: bool = True
    patience: int = 5  # stop after N steps without improvement
    min_improvement: float = 0.001  # minimum AUC improvement threshold
    restore_best: bool = True  # restore to best test AUC when stopping early
    monitor_metric: str = "test_auc"  # "test_auc", "train_auc", or "gap"

    def __post_init__(self):
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {self.min_improvement}")
        valid_metrics = {"test_auc", "train_auc", "gap"}
        if self.monitor_metric not in valid_metrics:
            raise ValueError(
                f"monitor_metric must be one of {valid_metrics}, got '{self.monitor_metric}'"
            )


@dataclass
class FinalFilterConfig:
    """Configuration for final feature filtering.

    After stepwise selection we do additional checks:
    - p-value significance
    - VIF for multicolinearity
    - coefficient sign validation
    """
    max_pvalue: float = 0.05
    max_vif: float = 5.0  # VIF > 5 indicates multicolinearity issue
    check_coefficient_sign: bool = True  # coefficients should be positive for WoE
    remove_insignificant: bool = True


@dataclass
class ModelConfig:
    """Configuration for final model training.

    We use logistic regression for interpretability.
    Regularization is optional but can help with stability.
    """
    regularization: Optional[str] = None  # None, "l1", "l2"
    C: float = 1.0  # inverse regularization strength
    fit_intercept: bool = True
    max_iter: int = 1000
    solver: str = "lbfgs"  # lbfgs works well for l2 or no regularization

    def __post_init__(self):
        if self.regularization == "l1" and self.solver == "lbfgs":
            # lbfgs does not support L1, switching to saga
            logger.warning("Switching solver to 'saga' for L1 regularization")
            self.solver = "saga"


@dataclass
class DiagnosticsConfig:
    """Configuration for model diagnostics and reporting."""
    calculate_psi: bool = True
    calculate_csi: bool = True
    psi_threshold: float = 0.25  # PSI > 0.25 indicates significant shift
    generate_lift_charts: bool = True
    generate_score_distribution: bool = True
    n_score_bins: int = 10


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap confidence intervals.

    Bootstrap resampling provides confidence intervals for model metrics
    (AUC, Gini, KS) without assuming a specific distribution. This helps
    quantify uncertainty in performance estimates.

    Attributes:
        enabled: whether to calculate bootstrap CIs
        n_iterations: number of bootstrap resamples (1000 is standard)
        confidence_level: confidence level for intervals (0.95 = 95% CI)
        random_state: seed for reproducibility
        metrics: which metrics to bootstrap ("auc", "gini", "ks")
        stratified: whether to use stratified resampling (preserves class ratio)
    """
    enabled: bool = True
    n_iterations: int = 1000
    confidence_level: float = 0.95
    random_state: int = 42
    metrics: List[str] = field(default_factory=lambda: ["auc", "gini", "ks"])
    stratified: bool = True

    def __post_init__(self):
        if self.n_iterations < 100:
            raise ValueError(f"n_iterations should be >= 100, got {self.n_iterations}")
        if not 0 < self.confidence_level < 1:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {self.confidence_level}"
            )
        valid_metrics = {"auc", "gini", "ks"}
        for m in self.metrics:
            if m not in valid_metrics:
                raise ValueError(f"Invalid metric '{m}'. Must be one of {valid_metrics}")


@dataclass
class RejectInferenceConfig:
    """Configuration for reject inference.

    Reject inference handles the bias introduced by only observing outcomes
    for approved applicants. This stage infers likely outcomes for rejected
    applications and augments the training data.

    Attributes:
        enabled: whether to perform reject inference
        method: inference method to use:
            - "hard_cutoff": assign bad if score below threshold
            - "fuzzy": use predicted probabilities as weights
            - "parceling": distribute rejects based on score distribution
            - "augmentation": simple augmentation with fixed bad rate
        score_column: column containing application scores (for accepted and rejected)
        decision_column: column indicating accept/reject decision
        accept_value: value in decision_column indicating acceptance
        reject_value: value in decision_column indicating rejection
        hard_cutoff_threshold: score threshold for hard cutoff method
        parceling_n_bins: number of bins for parceling method
        augmentation_bad_rate: assumed bad rate for simple augmentation
        weight_rejects: weight to apply to inferred reject samples (0-1)
        random_state: seed for reproducibility
    """
    enabled: bool = False
    method: str = "fuzzy"  # "hard_cutoff", "fuzzy", "parceling", "augmentation"
    score_column: str = "application_score"
    decision_column: str = "decision"
    accept_value: Any = 1
    reject_value: Any = 0
    hard_cutoff_threshold: float = 0.5
    parceling_n_bins: int = 10
    augmentation_bad_rate: float = 0.5
    weight_rejects: float = 1.0
    random_state: int = 42

    def __post_init__(self):
        valid_methods = {"hard_cutoff", "fuzzy", "parceling", "augmentation"}
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{self.method}'"
            )
        if not 0 <= self.weight_rejects <= 1:
            raise ValueError(
                f"weight_rejects must be between 0 and 1, got {self.weight_rejects}"
            )
        if not 0 <= self.augmentation_bad_rate <= 1:
            raise ValueError(
                f"augmentation_bad_rate must be between 0 and 1, got {self.augmentation_bad_rate}"
            )


@dataclass
class InteractionConfig:
    """Configuration for feature interaction detection.

    Detects and generates meaningful interaction terms between features.
    Interactions can improve model performance by capturing non-additive effects
    (e.g., high income with low debt ratio is more predictive than either alone).

    Attributes:
        enabled: whether to search for interaction terms
        interaction_types: types of interactions to consider
            - "multiplicative": X1 * X2
            - "ratio": X1 / X2 (with safe division)
            - "difference": X1 - X2
        max_interactions: maximum number of interactions to add
        min_auc_improvement: minimum AUC improvement to keep interaction
        candidate_features: specific features to consider (None = all)
        exclude_features: features to exclude from interaction search
        test_all_pairs: if True, test all pairs; if False, use heuristic
        correlation_threshold: only try interactions between features with
                               abs correlation < threshold (avoid redundancy)
        pvalue_threshold: p-value threshold for interaction significance
        use_woe_values: if True, create interactions on WoE-transformed values
        random_state: seed for reproducibility
    """
    enabled: bool = False
    interaction_types: List[str] = field(default_factory=lambda: ["multiplicative"])
    max_interactions: int = 5
    min_auc_improvement: float = 0.005
    candidate_features: Optional[List[str]] = None
    exclude_features: List[str] = field(default_factory=list)
    test_all_pairs: bool = False
    correlation_threshold: float = 0.5
    pvalue_threshold: float = 0.05
    use_woe_values: bool = True
    random_state: int = 42

    def __post_init__(self):
        valid_types = {"multiplicative", "ratio", "difference"}
        for it in self.interaction_types:
            if it not in valid_types:
                raise ValueError(
                    f"Invalid interaction type '{it}'. Must be one of {valid_types}"
                )
        if self.max_interactions < 1:
            raise ValueError(f"max_interactions must be >= 1, got {self.max_interactions}")
        if not 0 <= self.min_auc_improvement <= 1:
            raise ValueError(
                f"min_auc_improvement must be between 0 and 1, got {self.min_auc_improvement}"
            )


@dataclass
class PipelineConfig:
    """Master configuration for entire pipeline.

    Agregates all stage-specific configs into single object.
    Use this to configure pipeline behaviour.
    """
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    type_detection: TypeDetectionConfig = field(default_factory=TypeDetectionConfig)
    imputation: ImputationConfig = field(default_factory=ImputationConfig)
    binning: BinningConfig = field(default_factory=BinningConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    final_filter: FinalFilterConfig = field(default_factory=FinalFilterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    reject_inference: RejectInferenceConfig = field(default_factory=RejectInferenceConfig)
    interactions: InteractionConfig = field(default_factory=InteractionConfig)

    # general settings
    random_state: int = 42
    verbose: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        import dataclasses

        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        return _convert(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create config from dictionary.

        Note: This is a simplified version that does not handle all edge cases,
        but works for basic usage.
        """
        # TODO: implement proper deserialization with enum handling
        return cls(**data)

    def save(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PipelineConfig":
        """Load config from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class StageResult:
    """Container for stage execution results.

    Each stage returns this object containing transformed data,
    list of selected/remaining features, and diagnostic info.

    Attributes:
        data: transformed DataFrame
        selected_features: features that passed this stage
        dropped_features: features removed by this stage (with reasons)
        diagnostics: stage-specific metrics and stats
        stage_name: identifier of the stage
        execution_time: how long stage took to run (seconds)
        status: whether stage succeeded or failed
    """
    data: pd.DataFrame
    selected_features: List[str]
    dropped_features: Dict[str, str] = field(default_factory=dict)  # feature -> reason
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    stage_name: str = ""
    execution_time: float = 0.0
    status: StageStatus = StageStatus.FITTED
    timestamp: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        """Generate human-readable summary of stage results."""
        lines = [
            f"=== {self.stage_name} ===",
            f"Status: {self.status.value}",
            f"Features in: {len(self.selected_features) + len(self.dropped_features)}",
            f"Features out: {len(self.selected_features)}",
            f"Dropped: {len(self.dropped_features)}",
            f"Execution time: {self.execution_time:.2f}s",
        ]
        if self.dropped_features:
            lines.append("Dropped features:")
            for feat, reason in list(self.dropped_features.items())[:5]:
                lines.append(f"  - {feat}: {reason}")
            if len(self.dropped_features) > 5:
                lines.append(f"  ... and {len(self.dropped_features) - 5} more")
        return "\n".join(lines)


# =============================================================================
# Abstract Base Class for Pipeline Stages
# =============================================================================

class PipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    Every stage in the reasoning pipeline should inherit from this class
    and implement the required abstract methods. Stages follow sklearn-like
    fit/transform pattern for consistency.

    The stage lifecycle is:
    1. __init__: receive configuration
    2. fit(): learn parameters from training data
    3. transform(): apply learned transformation to new data
    4. get_diagnostics(): return metrics and stats

    Attributes:
        name: unique identifier for this stage
        config: stage-specific configuration object
        _is_fitted: flag indicating if fit() was called
        _feature_names_in: input feature names (set during fit)
        _feature_names_out: output feature names (set during fit)

    Example:
        >>> stage = MyCustomStage(config)
        >>> stage.fit(X_train, y_train)
        >>> X_transformed = stage.transform(X_test)
        >>> diagnostics = stage.get_diagnostics()
    """

    # Subclasses should override this
    name: str = "base_stage"

    def __init__(self, config: Any = None):
        """Initialize stage with configuration.

        Args:
            config: stage-specific configuration dataclass
        """
        self.config = config
        self._is_fitted = False
        self._feature_names_in: List[str] = []
        self._feature_names_out: List[str] = []
        self._fit_timestamp: Optional[datetime] = None
        self._dropped_features: Dict[str, str] = {}  # feature -> drop reason
        self._diagnostics: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> "PipelineStage":
        """Fit stage to training data.

        Learn any parameters needed for transformation from the training set.
        This method should set _is_fitted = True upon successful completion.

        Args:
            X: feature DataFrame
            y: target Series (optional, some stages do not need it)
            **kwargs: additional arguments (e.g. sample_weights)

        Returns:
            self: fitted stage instance

        Raises:
            ValueError: if input data is invalid
        """
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformation to data.

        Transform input features using parameters learned during fit().
        Must be called after fit().

        Args:
            X: feature DataFrame to transform

        Returns:
            Transformed DataFrame

        Raises:
            RuntimeError: if called before fit()
            ValueError: if X has different columns than training data
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> pd.DataFrame:
        """Fit and transform in single step.

        Convenience method that calls fit() then transform().
        More efficient than calling them separately in some cases.

        Args:
            X: feature DataFrame
            y: target Series (optional)
            **kwargs: passed to fit()

        Returns:
            Transformed DataFrame
        """
        self.fit(X, y, **kwargs)
        return self.transform(X)

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return stage-specific diagnostic information.

        Called after fit() to retrieve metrics, statistics and
        other diagnostic data for reporting.

        Returns:
            Dictionary with diagnostic metrics
        """
        pass

    def get_result(self, X_transformed: pd.DataFrame, execution_time: float = 0.0) -> StageResult:
        """Package transformation results into StageResult object.

        Helper method to create consistent result objects.

        Args:
            X_transformed: the transformed DataFrame
            execution_time: how long fit_transform took

        Returns:
            StageResult with all metadata
        """
        return StageResult(
            data=X_transformed,
            selected_features=self._feature_names_out.copy(),
            dropped_features=self._dropped_features.copy(),
            diagnostics=self.get_diagnostics(),
            stage_name=self.name,
            execution_time=execution_time,
            status=StageStatus.FITTED if self._is_fitted else StageStatus.FAILED,
        )

    def check_is_fitted(self) -> None:
        """Verify that stage has been fitted.

        Raises:
            RuntimeError: if stage has not been fitted yet
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Stage '{self.name}' is not fitted yet. Call fit() first."
            )

    def _validate_input(self, X: pd.DataFrame, check_features: bool = True) -> None:
        """Validate input DataFrame.

        Common validation logic used by fit() and transform().

        Args:
            X: input DataFrame to validate
            check_features: if True, verify columns match training data

        Raises:
            ValueError: if validation fails
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                f"Expected pandas DataFrame, got {type(X).__name__}"
            )

        if X.empty:
            raise ValueError("Input DataFrame is empty")

        if check_features and self._is_fitted and self._feature_names_in:
            missing = set(self._feature_names_in) - set(X.columns)
            if missing:
                raise ValueError(
                    f"Missing features in input: {missing}. "
                    f"Model was trained on: {self._feature_names_in}"
                )

    def _log_info(self, message: str) -> None:
        """Log info message with stage name prefix."""
        logger.info(f"[{self.name}] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message with stage name prefix."""
        logger.warning(f"[{self.name}] {message}")

    def _log_error(self, message: str) -> None:
        """Log error message with stage name prefix."""
        logger.error(f"[{self.name}] {message}")

    # =========================================================================
    # Serialization methods
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage state to dictionary.

        Override in subclasses to include stage-specific state.
        Used for saving pipeline to disk.

        Returns:
            Dictionary with stage state
        """
        return {
            "name": self.name,
            "is_fitted": self._is_fitted,
            "feature_names_in": self._feature_names_in,
            "feature_names_out": self._feature_names_out,
            "dropped_features": self._dropped_features,
            "diagnostics": self._diagnostics,
            "fit_timestamp": self._fit_timestamp.isoformat() if self._fit_timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Any = None) -> "PipelineStage":
        """Reconstruct stage from serialized state.

        Override in subclasses for proper deserialization.

        Args:
            data: dictionary from to_dict()
            config: configuration object

        Returns:
            Reconstructed stage instance
        """
        instance = cls(config=config)
        instance._is_fitted = data.get("is_fitted", False)
        instance._feature_names_in = data.get("feature_names_in", [])
        instance._feature_names_out = data.get("feature_names_out", [])
        instance._dropped_features = data.get("dropped_features", {})
        instance._diagnostics = data.get("diagnostics", {})

        ts = data.get("fit_timestamp")
        if ts:
            instance._fit_timestamp = datetime.fromisoformat(ts)

        return instance

    def save(self, filepath: str) -> None:
        """Save stage to pickle file.

        Args:
            filepath: path to save file
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        self._log_info(f"Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "PipelineStage":
        """Load stage from pickle file.

        Args:
            filepath: path to pickle file

        Returns:
            Loaded stage instance
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', status={status})"

    def __str__(self) -> str:
        return self.__repr__()


# =============================================================================
# Utility functions
# =============================================================================

def validate_target(y: pd.Series, expected_values: set = {0, 1}) -> None:
    """Validate target variable for binary classification.

    Args:
        y: target Series
        expected_values: set of valid values (default {0, 1})

    Raises:
        ValueError: if target is invalid
    """
    if y is None:
        raise ValueError("Target variable y cannot be None")

    unique_vals = set(y.dropna().unique())

    if not unique_vals.issubset(expected_values):
        raise ValueError(
            f"Target must contain only {expected_values}, got {unique_vals}"
        )

    if len(unique_vals) < 2:
        raise ValueError(
            f"Target must have at least 2 classes, got {len(unique_vals)}"
        )

    # check for class imbalance (just warning, not error)
    class_counts = y.value_counts()
    min_ratio = class_counts.min() / class_counts.sum()
    if min_ratio < 0.01:
        logger.warning(
            f"Severe class imbalance detected: minority class is {min_ratio:.2%} of data"
        )


def validate_sample_type(df: pd.DataFrame, col: str = "sample_type") -> None:
    """Validate sample_type column exists and has expected values.

    Expected values:
        0 = training
        1 = test
        2 = validation (optional)

    Args:
        df: DataFrame to validate
        col: name of sample_type column

    Raises:
        ValueError: if validation fails
    """
    if col not in df.columns:
        raise ValueError(
            f"DataFrame must have '{col}' column. "
            f"Available columns: {list(df.columns)}"
        )

    valid_values = {0, 1, 2}
    actual_values = set(df[col].dropna().unique())

    if not actual_values.issubset(valid_values):
        raise ValueError(
            f"sample_type must contain values from {valid_values}, got {actual_values}"
        )

    if 0 not in actual_values:
        raise ValueError("sample_type must include 0 (training data)")
