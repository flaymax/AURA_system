"""
Data Cleaner stage for initial data preprocessing.

This stage performs basic data cleaning operations:
1. Remove features with too many nulls (>97% by default)
2. Remove low-variance features (>97% same value)
3. Remove non-numeric features (strings, objects, etc.)

If no features remain after cleaning, a trigger is raised
and pipeline execution stops with detailed error info.

All dropped features are logged with comprehensive details for
later visualization and debugging.

Author: AURA Team
"""

import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
import json

import pandas as pd
import numpy as np

from core.base import PipelineStage, StageResult, StageStatus
from core.exceptions import (
    TriggerManager,
    TriggerDetails,
    TriggerSeverity,
    TriggerCategory,
    TriggerCodes,
    create_no_features_trigger,
    get_trigger_manager,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Feature Details Container
# =============================================================================

@dataclass
class FeatureDetails:
    """Comprehensive details about a single feature.

    This is used for logging and later visualization. Contains all
    the information needed to understand why feature was kept or dropped.
    """
    name: str
    dtype: str
    null_count: int
    null_ratio: float
    unique_count: int
    unique_ratio: float
    most_frequent_value: Any
    most_frequent_count: int
    most_frequent_ratio: float
    sample_values: List[Any]  # few example values
    min_value: Optional[float] = None  # for numeric only
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    is_numeric: bool = False
    status: str = "pending"  # pending, kept, dropped
    drop_reason: str = ""
    drop_reason_code: str = ""  # short code like "NULL", "LOW_VAR", "NON_NUM"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "null_count": self.null_count,
            "null_ratio": round(self.null_ratio, 4),
            "unique_count": self.unique_count,
            "unique_ratio": round(self.unique_ratio, 4),
            "most_frequent_value": self._serialize_value(self.most_frequent_value),
            "most_frequent_count": self.most_frequent_count,
            "most_frequent_ratio": round(self.most_frequent_ratio, 4),
            "sample_values": [self._serialize_value(v) for v in self.sample_values],
            "min_value": self._serialize_value(self.min_value),
            "max_value": self._serialize_value(self.max_value),
            "mean_value": round(self.mean_value, 4) if self.mean_value is not None else None,
            "std_value": round(self.std_value, 4) if self.std_value is not None else None,
            "is_numeric": self.is_numeric,
            "status": self.status,
            "drop_reason": self.drop_reason,
            "drop_reason_code": self.drop_reason_code,
        }

    @staticmethod
    def _serialize_value(val: Any) -> Any:
        """Convert value to JSON-serializable format."""
        if val is None or pd.isna(val):
            return None
        if isinstance(val, (np.integer, np.floating)):
            return float(val) if np.isfinite(val) else None
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, (np.ndarray, pd.Series)):
            return val.tolist()
        return val


@dataclass
class StageLog:
    """Structured log for pipeline stage execution.

    This is the main container for all logging information.
    Can be serialized to JSON for visualization layer.
    """
    stage_name: str
    stage_type: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    input_feature_count: int = 0
    output_feature_count: int = 0
    input_sample_count: int = 0
    features_kept: List[str] = field(default_factory=list)
    features_dropped: List[str] = field(default_factory=list)
    feature_details: Dict[str, FeatureDetails] = field(default_factory=dict)
    config_used: Dict[str, Any] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (self.completed_at - self.started_at).total_seconds() if self.completed_at else None,
            "input_feature_count": self.input_feature_count,
            "output_feature_count": self.output_feature_count,
            "input_sample_count": self.input_sample_count,
            "features_kept": self.features_kept,
            "features_dropped": self.features_dropped,
            "feature_details": {k: v.to_dict() for k, v in self.feature_details.items()},
            "config_used": self.config_used,
            "summary_stats": self.summary_stats,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def save(self, filepath: str) -> None:
        """Save log to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def add_warning(self, message: str) -> None:
        """Add warning message to log."""
        self.warnings.append(message)
        logger.warning(f"[{self.stage_name}] {message}")

    def add_error(self, message: str) -> None:
        """Add error message to log."""
        self.errors.append(message)
        logger.error(f"[{self.stage_name}] {message}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataCleanerConfig:
    """Configuration for DataCleaner stage.

    Attributes:
        null_threshold: max allowed ratio of nulls (features with more are dropped)
        single_value_threshold: max allowed ratio of most frequent value
        remove_non_numeric: whether to drop non-numeric columns
        numeric_types: pandas dtypes considered numeric
        keep_columns: columns to never drop (like sample_type)
        verbose: whether to log detailed information
    """
    null_threshold: float = 0.97  # drop if >97% null
    single_value_threshold: float = 0.97  # drop if >97% same value
    remove_non_numeric: bool = True
    numeric_types: tuple = (
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    )
    keep_columns: List[str] = field(default_factory=list)
    verbose: bool = True

    def __post_init__(self):
        # Validation
        if not 0 < self.null_threshold <= 1:
            raise ValueError(
                f"null_threshold must be between 0 and 1, got {self.null_threshold}"
            )
        if not 0 < self.single_value_threshold <= 1:
            raise ValueError(
                f"single_value_threshold must be between 0 and 1, "
                f"got {self.single_value_threshold}"
            )


# =============================================================================
# Data Cleaner Stage
# =============================================================================

class DataCleaner(PipelineStage):
    """Data cleaning stage that removes unusable features.

    This is typically the first stage in the pipeline. It identifies
    and removes features that cant be used for modeling:

    1. High-null features: columns where most values are missing
    2. Low-variance features: columns where almost all values are the same
    3. Non-numeric features: strings, objects, categories (for now)

    The stage tracks which features were dropped and why, which is
    useful for debugging data quality issues.

    Example:
        >>> cleaner = DataCleaner(DataCleanerConfig(null_threshold=0.95))
        >>> cleaner.fit(X, y)
        >>> X_clean = cleaner.transform(X)
        >>> print(cleaner.get_dropped_features())

    Attributes:
        name: stage identifier
        config: cleaning configuration
        _null_ratios: computed null ratios per feature
        _single_value_ratios: computed single-value ratios per feature
        _feature_dtypes: original dtypes of features
    """

    name = "data_cleaner"

    def __init__(self, config: Optional[DataCleanerConfig] = None):
        """Initialize DataCleaner with configuration.

        Args:
            config: DataCleanerConfig instance, or None for defaults
        """
        super().__init__(config)
        self.config: DataCleanerConfig = config or DataCleanerConfig()

        # These are computed during fit()
        self._null_ratios: Dict[str, float] = {}
        self._single_value_ratios: Dict[str, float] = {}
        self._feature_dtypes: Dict[str, str] = {}
        self._most_frequent_values: Dict[str, Any] = {}

        # Detailed feature information for logging
        self._feature_details: Dict[str, FeatureDetails] = {}

        # Stage log - comprehensive logging for visualization
        self._stage_log: Optional[StageLog] = None

        # Trigger manager for error handling
        self._trigger_manager: Optional[TriggerManager] = None

    def set_trigger_manager(self, manager: TriggerManager) -> "DataCleaner":
        """Set trigger manager for this stage.

        Args:
            manager: TriggerManager instance

        Returns:
            self for chaining
        """
        self._trigger_manager = manager
        return self

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> "DataCleaner":
        """Analyze data and determine which features to drop.

        Computes null ratios, single-value ratios, and dtypes for all
        features. Determines which features should be kept based on
        the configured thresholds.

        Args:
            X: feature DataFrame
            y: target Series (not used in this stage)
            **kwargs: ignored

        Returns:
            self: fitted stage

        Raises:
            DataQualityError: if no features remain after cleaning
        """
        self._validate_input(X, check_features=False)

        self._feature_names_in = list(X.columns)
        n_features = len(self._feature_names_in)
        n_samples = len(X)

        # Initialize stage log
        self._stage_log = StageLog(
            stage_name=self.name,
            stage_type="preprocessing",
            input_feature_count=n_features,
            input_sample_count=n_samples,
            config_used={
                "null_threshold": self.config.null_threshold,
                "single_value_threshold": self.config.single_value_threshold,
                "remove_non_numeric": self.config.remove_non_numeric,
                "keep_columns": self.config.keep_columns,
            }
        )

        self._log_info(f"Fitting on {n_samples} samples, {n_features} features")

        # Step 1: Compute detailed statistics for all features
        self._compute_feature_stats(X)

        # Step 2: Identify features to drop (also updates feature details)
        features_to_drop = self._identify_features_to_drop(X)

        # Step 3: Determine final feature set
        features_to_keep = [
            f for f in self._feature_names_in
            if f not in features_to_drop
        ]

        # Step 4: Update feature statuses and log
        for feat in features_to_keep:
            self._feature_details[feat].status = "kept"
        self._stage_log.features_kept = features_to_keep
        self._stage_log.features_dropped = list(features_to_drop)
        self._stage_log.output_feature_count = len(features_to_keep)
        self._stage_log.feature_details = self._feature_details

        # Step 5: Log summary statistics
        self._compute_summary_stats()

        # Step 6: Check if we have any features left
        if len(features_to_keep) == 0:
            self._stage_log.add_error("No features remaining after cleaning")
            self._handle_no_features_remaining(n_features, self._dropped_features)

        self._feature_names_out = features_to_keep
        self._is_fitted = True

        # Finalize log
        self._stage_log.completed_at = datetime.now()

        self._log_info(
            f"Fit complete: keeping {len(features_to_keep)}/{n_features} features, "
            f"dropped {len(self._dropped_features)}"
        )

        # Log dropped features details
        self._log_dropped_features_summary()

        return self

    def _compute_feature_stats(self, X: pd.DataFrame) -> None:
        """Compute detailed statistics for all features.

        Calculates comprehensive information for each feature:
        - null ratio
        - unique value count and ratio
        - most frequent value and its frequency
        - sample values for inspection
        - numeric statistics (min, max, mean, std) if applicable

        All information is stored in FeatureDetails objects for logging.
        """
        n_samples = len(X)

        for col in X.columns:
            col_data = X[col]
            dtype_str = str(col_data.dtype)

            # Basic counts
            null_count = int(col_data.isna().sum())
            null_ratio = null_count / n_samples
            non_null = col_data.dropna()
            non_null_count = len(non_null)

            # Unique values
            unique_count = int(col_data.nunique(dropna=True))
            unique_ratio = unique_count / n_samples if n_samples > 0 else 0

            # Most frequent value
            if non_null_count > 0:
                value_counts = non_null.value_counts()
                most_frequent_val = value_counts.index[0]
                most_frequent_cnt = int(value_counts.iloc[0])
                most_frequent_ratio = most_frequent_cnt / n_samples
            else:
                most_frequent_val = None
                most_frequent_cnt = 0
                most_frequent_ratio = 1.0  # all nulls considered as "same value"

            # Sample values (up to 5 unique non-null values)
            if non_null_count > 0:
                sample_vals = non_null.drop_duplicates().head(5).tolist()
            else:
                sample_vals = []

            # Check if numeric
            is_numeric = self._is_numeric_dtype(dtype_str)

            # Numeric statistics
            min_val = max_val = mean_val = std_val = None
            if is_numeric and non_null_count > 0:
                try:
                    numeric_data = pd.to_numeric(non_null, errors="coerce").dropna()
                    if len(numeric_data) > 0:
                        min_val = float(numeric_data.min())
                        max_val = float(numeric_data.max())
                        mean_val = float(numeric_data.mean())
                        std_val = float(numeric_data.std()) if len(numeric_data) > 1 else 0.0
                except (ValueError, TypeError):
                    # Cant compute numeric stats, leave as None
                    pass

            # Store in legacy dicts for backward compatibility
            self._null_ratios[col] = null_ratio
            self._single_value_ratios[col] = most_frequent_ratio
            self._most_frequent_values[col] = most_frequent_val
            self._feature_dtypes[col] = dtype_str

            # Store detailed feature info
            self._feature_details[col] = FeatureDetails(
                name=col,
                dtype=dtype_str,
                null_count=null_count,
                null_ratio=null_ratio,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                most_frequent_value=most_frequent_val,
                most_frequent_count=most_frequent_cnt,
                most_frequent_ratio=most_frequent_ratio,
                sample_values=sample_vals,
                min_value=min_val,
                max_value=max_val,
                mean_value=mean_val,
                std_value=std_val,
                is_numeric=is_numeric,
                status="pending",
            )

    def _identify_features_to_drop(self, X: pd.DataFrame) -> Set[str]:
        """Identify which features should be dropped based on criteria.

        Also updates FeatureDetails with drop reason and status.

        Returns:
            Set of feature names to drop
        """
        features_to_drop = set()
        self._dropped_features = {}

        for col in X.columns:
            # Skip protected columns
            if col in self.config.keep_columns:
                self._feature_details[col].status = "kept"
                self._feature_details[col].drop_reason = "protected (in keep_columns)"
                continue

            drop_info = self._check_feature_validity(col, X[col])
            if drop_info:
                reason_code, reason_text = drop_info
                features_to_drop.add(col)
                self._dropped_features[col] = reason_text

                # Update feature details
                self._feature_details[col].status = "dropped"
                self._feature_details[col].drop_reason = reason_text
                self._feature_details[col].drop_reason_code = reason_code

        return features_to_drop

    def _check_feature_validity(self, col_name: str, col_data: pd.Series) -> Optional[tuple]:
        """Check if a single feature should be dropped.

        Args:
            col_name: feature name
            col_data: feature values

        Returns:
            Tuple of (reason_code, reason_text) or None if feature is valid
        """
        # Check 1: Too many nulls
        null_ratio = self._null_ratios[col_name]
        if null_ratio > self.config.null_threshold:
            return (
                "NULL",
                f"too_many_nulls ({null_ratio:.1%} null, threshold {self.config.null_threshold:.1%})"
            )

        # Check 2: Low variance (most values are the same)
        single_val_ratio = self._single_value_ratios[col_name]
        if single_val_ratio > self.config.single_value_threshold:
            most_common = self._most_frequent_values[col_name]
            return (
                "LOW_VAR",
                f"low_variance ({single_val_ratio:.1%} are '{most_common}', "
                f"threshold {self.config.single_value_threshold:.1%})"
            )

        # Check 3: Non-numeric type
        if self.config.remove_non_numeric:
            dtype = self._feature_dtypes[col_name]
            if not self._is_numeric_dtype(dtype):
                return (
                    "NON_NUM",
                    f"non_numeric_type (dtype={dtype})"
                )

        return None

    def _is_numeric_dtype(self, dtype: str) -> bool:
        """Check if dtype is considered numeric.

        Args:
            dtype: string representation of dtype

        Returns:
            True if numeric, False otherwise
        """
        # Check against known numeric types
        for numeric_type in self.config.numeric_types:
            if numeric_type in dtype.lower():
                return True

        # Also check for numpy number types
        if "int" in dtype.lower() or "float" in dtype.lower():
            return True

        return False

    def _compute_summary_stats(self) -> None:
        """Compute summary statistics for the stage log.

        Aggregates information about dropped features by reason,
        data type distribution, and other useful metrics.
        """
        if self._stage_log is None:
            return

        # Count features by drop reason
        drop_reason_counts = {"NULL": 0, "LOW_VAR": 0, "NON_NUM": 0}
        for details in self._feature_details.values():
            if details.drop_reason_code in drop_reason_counts:
                drop_reason_counts[details.drop_reason_code] += 1

        # Count features by dtype
        dtype_counts = {}
        for details in self._feature_details.values():
            dtype = details.dtype
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        # Count numeric vs non-numeric
        numeric_count = sum(1 for d in self._feature_details.values() if d.is_numeric)
        non_numeric_count = len(self._feature_details) - numeric_count

        # Average null ratio
        avg_null_ratio = np.mean([d.null_ratio for d in self._feature_details.values()])

        # Features with high null ratio (but below threshold)
        high_null_but_kept = [
            d.name for d in self._feature_details.values()
            if d.status == "kept" and d.null_ratio > 0.5
        ]

        self._stage_log.summary_stats = {
            "drop_reason_counts": drop_reason_counts,
            "dtype_distribution": dtype_counts,
            "numeric_features_count": numeric_count,
            "non_numeric_features_count": non_numeric_count,
            "average_null_ratio": round(avg_null_ratio, 4),
            "high_null_features_kept": high_null_but_kept,
            "thresholds": {
                "null_threshold": self.config.null_threshold,
                "single_value_threshold": self.config.single_value_threshold,
            }
        }

        # Add warnings for concerning patterns
        if len(high_null_but_kept) > 0:
            self._stage_log.add_warning(
                f"{len(high_null_but_kept)} features have >50% nulls but were kept: "
                f"{high_null_but_kept[:3]}{'...' if len(high_null_but_kept) > 3 else ''}"
            )

        if drop_reason_counts["NON_NUM"] > 10:
            self._stage_log.add_warning(
                f"{drop_reason_counts['NON_NUM']} non-numeric features were dropped. "
                f"Consider preprocessing categorical features separately."
            )

    def _log_dropped_features_summary(self) -> None:
        """Log detailed summary of dropped features.

        Outputs structured information about each dropped feature
        for debugging and later visualization.
        """
        if not self._dropped_features:
            self._log_info("No features were dropped")
            return

        # Group by reason code
        by_reason = {"NULL": [], "LOW_VAR": [], "NON_NUM": []}
        for details in self._feature_details.values():
            if details.status == "dropped" and details.drop_reason_code in by_reason:
                by_reason[details.drop_reason_code].append(details)

        # Log summary by reason
        self._log_info("=" * 60)
        self._log_info("DROPPED FEATURES SUMMARY")
        self._log_info("=" * 60)

        # High null features
        if by_reason["NULL"]:
            self._log_info(f"\n[NULL] Too many missing values ({len(by_reason['NULL'])} features):")
            for details in sorted(by_reason["NULL"], key=lambda x: -x.null_ratio)[:10]:
                self._log_info(
                    f"  - {details.name}: {details.null_ratio:.1%} null, "
                    f"dtype={details.dtype}"
                )
            if len(by_reason["NULL"]) > 10:
                self._log_info(f"  ... and {len(by_reason['NULL']) - 10} more")

        # Low variance features
        if by_reason["LOW_VAR"]:
            self._log_info(f"\n[LOW_VAR] Low variance ({len(by_reason['LOW_VAR'])} features):")
            for details in sorted(by_reason["LOW_VAR"], key=lambda x: -x.most_frequent_ratio)[:10]:
                self._log_info(
                    f"  - {details.name}: {details.most_frequent_ratio:.1%} are '{details.most_frequent_value}', "
                    f"unique_count={details.unique_count}"
                )
            if len(by_reason["LOW_VAR"]) > 10:
                self._log_info(f"  ... and {len(by_reason['LOW_VAR']) - 10} more")

        # Non-numeric features
        if by_reason["NON_NUM"]:
            self._log_info(f"\n[NON_NUM] Non-numeric types ({len(by_reason['NON_NUM'])} features):")
            for details in by_reason["NON_NUM"][:10]:
                sample_str = str(details.sample_values[:3]) if details.sample_values else "[]"
                self._log_info(
                    f"  - {details.name}: dtype={details.dtype}, "
                    f"unique={details.unique_count}, samples={sample_str}"
                )
            if len(by_reason["NON_NUM"]) > 10:
                self._log_info(f"  ... and {len(by_reason['NON_NUM']) - 10} more")

        self._log_info("=" * 60)

    def _handle_no_features_remaining(
        self,
        original_count: int,
        dropped_features: Dict[str, str]
    ) -> None:
        """Handle the case when no features remain after cleaning.

        Creates a trigger and raises an exception to stop pipeline.

        Args:
            original_count: how many features we started with
            dropped_features: dict of dropped feature -> reason
        """
        trigger = create_no_features_trigger(
            stage_name=self.name,
            original_count=original_count,
            dropped_features=dropped_features,
        )

        # Add more context to the trigger
        trigger.details["null_threshold"] = self.config.null_threshold
        trigger.details["single_value_threshold"] = self.config.single_value_threshold
        trigger.details["remove_non_numeric"] = self.config.remove_non_numeric

        # Breakdown of drop reasons
        reason_counts = {}
        for reason in dropped_features.values():
            reason_type = reason.split(" ")[0]  # first word is reason type
            reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1
        trigger.details["drop_reason_breakdown"] = reason_counts

        # Get trigger manager
        manager = self._trigger_manager or get_trigger_manager()
        manager.raise_trigger(trigger)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning transformation to data.

        Removes features that were identified during fit().

        Args:
            X: feature DataFrame to transform

        Returns:
            Cleaned DataFrame with only valid features
        """
        self.check_is_fitted()
        self._validate_input(X)

        # Select only features that passed cleaning
        X_clean = X[self._feature_names_out].copy()

        self._log_info(
            f"Transform: {len(X.columns)} -> {len(X_clean.columns)} features"
        )

        return X_clean

    def get_stage_log(self) -> Optional[StageLog]:
        """Get the comprehensive stage log.

        Returns the StageLog object containing all information about
        what happened during fit(). Can be serialized to JSON for
        visualization layer.

        Returns:
            StageLog instance or None if not fitted
        """
        return self._stage_log

    def save_stage_log(self, filepath: str) -> None:
        """Save stage log to JSON file.

        Args:
            filepath: path to save JSON file
        """
        if self._stage_log is None:
            raise RuntimeError("No stage log available. Call fit() first.")
        self._stage_log.save(filepath)
        self._log_info(f"Stage log saved to {filepath}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about cleaning process.

        Returns:
            Dictionary with cleaning statistics and details
        """
        if not self._is_fitted:
            return {}

        # Group dropped features by reason
        drop_reasons = {}
        for feat, reason in self._dropped_features.items():
            reason_type = reason.split(" ")[0]
            if reason_type not in drop_reasons:
                drop_reasons[reason_type] = []
            drop_reasons[reason_type].append(feat)

        diagnostics = {
            "original_feature_count": len(self._feature_names_in),
            "remaining_feature_count": len(self._feature_names_out),
            "dropped_feature_count": len(self._dropped_features),
            "drop_reasons_summary": {k: len(v) for k, v in drop_reasons.items()},
            "dropped_features_by_reason": drop_reasons,
            "null_threshold": self.config.null_threshold,
            "single_value_threshold": self.config.single_value_threshold,
            "null_ratios": {
                f: r for f, r in self._null_ratios.items()
                if r > 0.5  # only report features with >50% nulls
            },
            "low_variance_features": {
                f: r for f, r in self._single_value_ratios.items()
                if r > 0.9  # only report features with >90% same value
            },
        }

        # Add summary stats from stage log if available
        if self._stage_log and self._stage_log.summary_stats:
            diagnostics["summary_stats"] = self._stage_log.summary_stats

        return diagnostics

    def get_feature_report(self) -> pd.DataFrame:
        """Generate detailed report on all features.

        Returns:
            DataFrame with one row per feature showing comprehensive stats
        """
        self.check_is_fitted()

        records = []
        for feat in self._feature_names_in:
            details = self._feature_details.get(feat)
            if details:
                records.append({
                    "feature": details.name,
                    "dtype": details.dtype,
                    "is_numeric": details.is_numeric,
                    "null_count": details.null_count,
                    "null_ratio": round(details.null_ratio, 4),
                    "unique_count": details.unique_count,
                    "unique_ratio": round(details.unique_ratio, 4),
                    "most_frequent_value": details.most_frequent_value,
                    "most_frequent_ratio": round(details.most_frequent_ratio, 4),
                    "min_value": details.min_value,
                    "max_value": details.max_value,
                    "mean_value": round(details.mean_value, 4) if details.mean_value else None,
                    "std_value": round(details.std_value, 4) if details.std_value else None,
                    "sample_values": str(details.sample_values[:3]),
                    "status": details.status,
                    "drop_reason_code": details.drop_reason_code,
                    "drop_reason": details.drop_reason,
                })
            else:
                # Fallback for features without details (shouldnt happen)
                records.append({
                    "feature": feat,
                    "dtype": self._feature_dtypes.get(feat, "unknown"),
                    "is_numeric": False,
                    "null_count": 0,
                    "null_ratio": self._null_ratios.get(feat, 0),
                    "unique_count": 0,
                    "unique_ratio": 0,
                    "most_frequent_value": self._most_frequent_values.get(feat),
                    "most_frequent_ratio": self._single_value_ratios.get(feat, 0),
                    "min_value": None,
                    "max_value": None,
                    "mean_value": None,
                    "std_value": None,
                    "sample_values": "[]",
                    "status": "kept" if feat in self._feature_names_out else "dropped",
                    "drop_reason_code": "",
                    "drop_reason": self._dropped_features.get(feat, ""),
                })

        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage state for persistence."""
        base_dict = super().to_dict()
        base_dict.update({
            "config": {
                "null_threshold": self.config.null_threshold,
                "single_value_threshold": self.config.single_value_threshold,
                "remove_non_numeric": self.config.remove_non_numeric,
                "keep_columns": self.config.keep_columns,
            },
            "null_ratios": self._null_ratios,
            "single_value_ratios": self._single_value_ratios,
            "feature_dtypes": self._feature_dtypes,
            "feature_details": {
                k: v.to_dict() for k, v in self._feature_details.items()
            },
            "stage_log": self._stage_log.to_dict() if self._stage_log else None,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[DataCleanerConfig] = None) -> "DataCleaner":
        """Reconstruct stage from serialized state."""
        if config is None and "config" in data:
            config = DataCleanerConfig(**data["config"])

        instance = cls(config=config)

        # Restore base state
        instance._is_fitted = data.get("is_fitted", False)
        instance._feature_names_in = data.get("feature_names_in", [])
        instance._feature_names_out = data.get("feature_names_out", [])
        instance._dropped_features = data.get("dropped_features", {})

        # Restore stage-specific state
        instance._null_ratios = data.get("null_ratios", {})
        instance._single_value_ratios = data.get("single_value_ratios", {})
        instance._feature_dtypes = data.get("feature_dtypes", {})

        # Note: feature_details and stage_log are not fully restored
        # as they contain complex objects. Use save/load for full persistence.

        return instance
