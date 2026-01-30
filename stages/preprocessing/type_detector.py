"""
Type Detector stage for feature type classification.

This stage analyzes features and classifies them into types:
1. binary - exactly 2 unique values (0/1, true/false, yes/no)
2. categorical - limited values (3-5 unique values)
3. discrete_numeric - numeric with finite set of values (6-30 unique)
4. continuous_numeric - regular numeric with many unique values (>30)

The stage wraps and extends the detect_feature_type function from
src/utils_feature_types.py with more granular classification.

Author: AURA Team
"""

import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

import pandas as pd
import numpy as np

from core.base import PipelineStage, StageResult, StageStatus
from src.utils_feature_types import detect_feature_type


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class FeatureType(Enum):
    """Classification of feature types.

    These types determine how features are treated in later pipeline stages.
    For WoE binning, binary and categorical may use different strategies.
    """
    BINARY = "binary"                    # exactly 2 unique values
    CATEGORICAL = "categorical"          # 3-5 unique values (narrow range)
    DISCRETE_NUMERIC = "discrete_numeric"  # 6-30 unique values
    CONTINUOUS_NUMERIC = "continuous_numeric"  # >30 unique values
    ID_LIKE = "id_like"                  # likely an ID column (very high cardinality)
    UNKNOWN = "unknown"                  # couldn't determine type


# =============================================================================
# Type Info Container
# =============================================================================

@dataclass
class FeatureTypeInfo:
    """Detailed type information for a single feature.

    Contains classification result and supporting statistics.
    """
    name: str
    dtype: str
    inferred_type: FeatureType
    n_unique: int
    n_non_null: int
    n_total: int
    null_ratio: float
    unique_ratio: float
    unique_values: List[Any]  # actual unique values (for binary/categorical)
    value_counts: Dict[Any, int]  # frequency of each value
    is_numeric_dtype: bool
    original_type: str  # type from base detect_feature_type function
    confidence: str  # high, medium, low - how confident we are in classification

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type.value,
            "n_unique": self.n_unique,
            "n_non_null": self.n_non_null,
            "n_total": self.n_total,
            "null_ratio": round(self.null_ratio, 4),
            "unique_ratio": round(self.unique_ratio, 4),
            "unique_values": self._serialize_values(self.unique_values),
            "value_counts": {str(k): v for k, v in self.value_counts.items()},
            "is_numeric_dtype": self.is_numeric_dtype,
            "original_type": self.original_type,
            "confidence": self.confidence,
        }

    @staticmethod
    def _serialize_values(values: List[Any]) -> List[Any]:
        """Convert values to JSON-serializable format."""
        result = []
        for val in values:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                result.append(None)
            elif isinstance(val, (np.integer, np.floating)):
                result.append(float(val) if np.isfinite(val) else None)
            elif isinstance(val, np.bool_):
                result.append(bool(val))
            else:
                result.append(val)
        return result


@dataclass
class TypeDetectorLog:
    """Structured log for TypeDetector stage execution."""
    stage_name: str
    stage_type: str = "preprocessing"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    input_feature_count: int = 0
    input_sample_count: int = 0
    feature_type_info: Dict[str, FeatureTypeInfo] = field(default_factory=dict)
    type_distribution: Dict[str, int] = field(default_factory=dict)
    config_used: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "stage_name": self.stage_name,
            "stage_type": self.stage_type,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": (
                (self.completed_at - self.started_at).total_seconds()
                if self.completed_at else None
            ),
            "input_feature_count": self.input_feature_count,
            "input_sample_count": self.input_sample_count,
            "feature_type_info": {k: v.to_dict() for k, v in self.feature_type_info.items()},
            "type_distribution": self.type_distribution,
            "config_used": self.config_used,
            "warnings": self.warnings,
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


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TypeDetectorConfig:
    """Configuration for TypeDetector stage.

    Thresholds define boundaries between feature types based on
    number of unique values.

    Attributes:
        binary_max: max unique values for binary (always 2)
        categorical_max: max unique values for categorical (default 5)
        discrete_max: max unique values for discrete numeric (default 30)
        id_like_ratio: unique ratio above which feature is considered ID-like
        detect_binary_strings: whether to detect string binary like 'yes'/'no'
        binary_string_patterns: known binary value pairs
    """
    binary_max: int = 2  # exactly 2 unique = binary
    categorical_min: int = 3  # at least 3 for categorical (not binary)
    categorical_max: int = 5  # up to 5 unique = categorical
    discrete_min: int = 6  # 6+ = discrete numeric
    discrete_max: int = 30  # up to 30 = discrete numeric
    continuous_min: int = 31  # 31+ = continuous
    id_like_ratio: float = 0.9  # if unique_ratio > 0.9, likely an ID
    detect_binary_strings: bool = True
    binary_string_patterns: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("yes", "no"),
        ("true", "false"),
        ("y", "n"),
        ("t", "f"),
        ("1", "0"),
        ("male", "female"),
        ("m", "f"),
    ])
    verbose: bool = True

    def __post_init__(self):
        # Validate thresholds
        if self.binary_max != 2:
            logger.warning("binary_max should be 2, setting to 2")
            self.binary_max = 2
        if self.categorical_max < self.categorical_min:
            raise ValueError(
                f"categorical_max ({self.categorical_max}) must be >= "
                f"categorical_min ({self.categorical_min})"
            )
        if self.discrete_max < self.discrete_min:
            raise ValueError(
                f"discrete_max ({self.discrete_max}) must be >= "
                f"discrete_min ({self.discrete_min})"
            )


# =============================================================================
# Type Detector Stage
# =============================================================================

class TypeDetector(PipelineStage):
    """Feature type detection and classification stage.

    Analyzes each feature and classifies it into one of the types:
    - binary: exactly 2 unique values
    - categorical: 3-5 unique values
    - discrete_numeric: 6-30 unique values
    - continuous_numeric: >30 unique values
    - id_like: very high cardinality, likely an ID

    This classification helps downstream stages (like WoE binning) choose
    appropriate strategies for each feature type.

    Note: This stage does NOT drop features. It only adds type metadata.
    The DataCleaner stage should run before this one to remove unusable features.

    Example:
        >>> detector = TypeDetector(TypeDetectorConfig(categorical_max=5))
        >>> detector.fit(X)
        >>> type_info = detector.get_feature_types()
        >>> binary_features = detector.get_features_by_type(FeatureType.BINARY)
    """

    name = "type_detector"

    def __init__(self, config: Optional[TypeDetectorConfig] = None):
        """Initialize TypeDetector with configuration.

        Args:
            config: TypeDetectorConfig instance, or None for defaults
        """
        super().__init__(config)
        self.config: TypeDetectorConfig = config or TypeDetectorConfig()

        # Results storage
        self._feature_type_info: Dict[str, FeatureTypeInfo] = {}
        self._features_by_type: Dict[FeatureType, List[str]] = {
            t: [] for t in FeatureType
        }

        # Stage log
        self._stage_log: Optional[TypeDetectorLog] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **kwargs) -> "TypeDetector":
        """Analyze features and classify their types.

        Args:
            X: feature DataFrame (should already be cleaned by DataCleaner)
            y: target Series (not used in this stage)
            **kwargs: ignored

        Returns:
            self: fitted stage
        """
        self._validate_input(X, check_features=False)

        self._feature_names_in = list(X.columns)
        n_features = len(self._feature_names_in)
        n_samples = len(X)

        # Initialize stage log
        self._stage_log = TypeDetectorLog(
            stage_name=self.name,
            input_feature_count=n_features,
            input_sample_count=n_samples,
            config_used={
                "binary_max": self.config.binary_max,
                "categorical_max": self.config.categorical_max,
                "discrete_max": self.config.discrete_max,
                "id_like_ratio": self.config.id_like_ratio,
            }
        )

        self._log_info(f"Analyzing types for {n_features} features")

        # Reset storage
        self._feature_type_info = {}
        self._features_by_type = {t: [] for t in FeatureType}

        # Analyze each feature
        for col in X.columns:
            type_info = self._analyze_feature(X, col)
            self._feature_type_info[col] = type_info
            self._features_by_type[type_info.inferred_type].append(col)

        # Update log with results
        self._stage_log.feature_type_info = self._feature_type_info
        self._stage_log.type_distribution = {
            t.value: len(features)
            for t, features in self._features_by_type.items()
            if len(features) > 0
        }

        # Check for potential issues
        self._check_for_warnings()

        self._feature_names_out = self._feature_names_in.copy()
        self._is_fitted = True
        self._stage_log.completed_at = datetime.now()

        # Log summary
        self._log_type_summary()

        return self

    def _analyze_feature(self, df: pd.DataFrame, feature: str) -> FeatureTypeInfo:
        """Analyze a single feature and determine its type.

        Uses the base detect_feature_type function and extends it with
        more granular classification.
        """
        col_data = df[feature]
        n_total = len(col_data)
        n_non_null = int(col_data.notna().sum())
        null_ratio = 1 - (n_non_null / n_total) if n_total > 0 else 0

        # Use base function for initial analysis
        base_result = detect_feature_type(df, feature)

        n_unique = base_result["n_unique"]
        unique_ratio = base_result["unique_ratio"]
        dtype_str = base_result["dtype"]
        original_type = base_result["feature_type"]

        # Check if dtype is numeric
        is_numeric = self._is_numeric_dtype(dtype_str)

        # Get unique values and their counts (for small cardinality)
        unique_values = []
        value_counts = {}
        if n_unique <= self.config.discrete_max:
            try:
                non_null = col_data.dropna()
                unique_values = non_null.unique().tolist()
                vc = non_null.value_counts()
                value_counts = {k: int(v) for k, v in vc.items()}
            except Exception:
                pass

        # Determine type with our extended classification
        inferred_type, confidence = self._classify_feature(
            n_unique=n_unique,
            unique_ratio=unique_ratio,
            is_numeric=is_numeric,
            unique_values=unique_values,
            original_type=original_type,
        )

        return FeatureTypeInfo(
            name=feature,
            dtype=dtype_str,
            inferred_type=inferred_type,
            n_unique=n_unique,
            n_non_null=n_non_null,
            n_total=n_total,
            null_ratio=null_ratio,
            unique_ratio=unique_ratio,
            unique_values=unique_values[:20],  # limit for storage
            value_counts=dict(list(value_counts.items())[:20]),  # limit
            is_numeric_dtype=is_numeric,
            original_type=original_type,
            confidence=confidence,
        )

    def _classify_feature(
        self,
        n_unique: int,
        unique_ratio: float,
        is_numeric: bool,
        unique_values: List[Any],
        original_type: str,
    ) -> Tuple[FeatureType, str]:
        """Classify feature into one of our types.

        Returns:
            Tuple of (FeatureType, confidence)
        """
        # Check for ID-like first (very high cardinality)
        if unique_ratio > self.config.id_like_ratio:
            return FeatureType.ID_LIKE, "high"

        # Binary: exactly 2 unique values
        if n_unique == 2:
            # Check if it's a known binary pattern
            if self._is_known_binary_pattern(unique_values):
                return FeatureType.BINARY, "high"
            return FeatureType.BINARY, "high"

        # Categorical: 3-5 unique values
        if self.config.categorical_min <= n_unique <= self.config.categorical_max:
            return FeatureType.CATEGORICAL, "high"

        # For numeric types with more values
        if is_numeric:
            # Discrete numeric: 6-30 unique values
            if self.config.discrete_min <= n_unique <= self.config.discrete_max:
                return FeatureType.DISCRETE_NUMERIC, "high"

            # Continuous: >30 unique values
            if n_unique > self.config.discrete_max:
                return FeatureType.CONTINUOUS_NUMERIC, "high"

        # Non-numeric with many values - treat as categorical but low confidence
        if not is_numeric and n_unique > self.config.categorical_max:
            # Could be categorical with many levels or text field
            if n_unique <= self.config.discrete_max:
                return FeatureType.CATEGORICAL, "medium"
            else:
                return FeatureType.ID_LIKE, "medium"

        # Edge cases
        if n_unique == 1:
            # Single value - should have been removed by DataCleaner
            return FeatureType.CATEGORICAL, "low"

        if n_unique == 0:
            # All nulls - should have been removed
            return FeatureType.UNKNOWN, "low"

        return FeatureType.UNKNOWN, "low"

    def _is_known_binary_pattern(self, values: List[Any]) -> bool:
        """Check if values match a known binary pattern."""
        if len(values) != 2:
            return False

        if not self.config.detect_binary_strings:
            return False

        # Normalize values to lowercase strings for comparison
        try:
            str_values = set(str(v).lower().strip() for v in values)
        except Exception:
            return False

        for pattern in self.config.binary_string_patterns:
            if str_values == set(p.lower() for p in pattern):
                return True

        return False

    def _is_numeric_dtype(self, dtype: str) -> bool:
        """Check if dtype string represents a numeric type."""
        dtype_lower = dtype.lower()
        numeric_indicators = ["int", "float", "double", "decimal", "numeric"]
        return any(ind in dtype_lower for ind in numeric_indicators)

    def _check_for_warnings(self) -> None:
        """Check for potential issues and add warnings to log."""
        # Warn if many ID-like features
        id_like_count = len(self._features_by_type[FeatureType.ID_LIKE])
        if id_like_count > 5:
            self._stage_log.add_warning(
                f"{id_like_count} features detected as ID-like. "
                f"Consider removing them before modeling."
            )

        # Warn if no continuous features
        continuous_count = len(self._features_by_type[FeatureType.CONTINUOUS_NUMERIC])
        if continuous_count == 0:
            self._stage_log.add_warning(
                "No continuous numeric features detected. "
                "WoE binning may have limited effectiveness."
            )

        # Warn about low confidence classifications
        low_confidence = [
            name for name, info in self._feature_type_info.items()
            if info.confidence == "low"
        ]
        if low_confidence:
            self._stage_log.add_warning(
                f"{len(low_confidence)} features have low confidence type detection: "
                f"{low_confidence[:5]}{'...' if len(low_confidence) > 5 else ''}"
            )

    def _log_type_summary(self) -> None:
        """Log summary of type detection results."""
        self._log_info("=" * 60)
        self._log_info("FEATURE TYPE DETECTION SUMMARY")
        self._log_info("=" * 60)

        # Summary by type
        for ftype in FeatureType:
            features = self._features_by_type[ftype]
            if features:
                self._log_info(f"\n[{ftype.value.upper()}] {len(features)} features:")
                for feat in features[:10]:
                    info = self._feature_type_info[feat]
                    if ftype == FeatureType.BINARY:
                        values_str = str(info.unique_values)
                        self._log_info(f"  - {feat}: values={values_str}")
                    elif ftype in (FeatureType.CATEGORICAL, FeatureType.DISCRETE_NUMERIC):
                        self._log_info(
                            f"  - {feat}: {info.n_unique} unique values, "
                            f"dtype={info.dtype}"
                        )
                    else:
                        self._log_info(
                            f"  - {feat}: {info.n_unique} unique, "
                            f"unique_ratio={info.unique_ratio:.2%}"
                        )
                if len(features) > 10:
                    self._log_info(f"  ... and {len(features) - 10} more")

        self._log_info("=" * 60)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pass through data unchanged.

        TypeDetector doesn't modify data, only adds metadata.
        Use get_feature_types() to access type information.

        Args:
            X: feature DataFrame

        Returns:
            Same DataFrame unchanged
        """
        self.check_is_fitted()
        self._validate_input(X)

        # Just pass through - this stage only adds metadata
        return X.copy()

    # =========================================================================
    # Accessor Methods
    # =========================================================================

    def get_feature_types(self) -> Dict[str, FeatureType]:
        """Get mapping of feature name to inferred type.

        Returns:
            Dict mapping feature name to FeatureType
        """
        self.check_is_fitted()
        return {
            name: info.inferred_type
            for name, info in self._feature_type_info.items()
        }

    def get_features_by_type(self, feature_type: FeatureType) -> List[str]:
        """Get list of features with specified type.

        Args:
            feature_type: FeatureType to filter by

        Returns:
            List of feature names
        """
        self.check_is_fitted()
        return self._features_by_type.get(feature_type, []).copy()

    def get_binary_features(self) -> List[str]:
        """Get features classified as binary."""
        return self.get_features_by_type(FeatureType.BINARY)

    def get_categorical_features(self) -> List[str]:
        """Get features classified as categorical."""
        return self.get_features_by_type(FeatureType.CATEGORICAL)

    def get_discrete_features(self) -> List[str]:
        """Get features classified as discrete numeric."""
        return self.get_features_by_type(FeatureType.DISCRETE_NUMERIC)

    def get_continuous_features(self) -> List[str]:
        """Get features classified as continuous numeric."""
        return self.get_features_by_type(FeatureType.CONTINUOUS_NUMERIC)

    def get_numeric_features(self) -> List[str]:
        """Get all numeric features (discrete + continuous)."""
        return (
            self.get_features_by_type(FeatureType.DISCRETE_NUMERIC) +
            self.get_features_by_type(FeatureType.CONTINUOUS_NUMERIC)
        )

    def get_feature_type_info(self, feature: str) -> Optional[FeatureTypeInfo]:
        """Get detailed type info for specific feature.

        Args:
            feature: feature name

        Returns:
            FeatureTypeInfo or None if not found
        """
        return self._feature_type_info.get(feature)

    def get_stage_log(self) -> Optional[TypeDetectorLog]:
        """Get the comprehensive stage log."""
        return self._stage_log

    def save_stage_log(self, filepath: str) -> None:
        """Save stage log to JSON file."""
        if self._stage_log is None:
            raise RuntimeError("No stage log available. Call fit() first.")
        self._stage_log.save(filepath)
        self._log_info(f"Stage log saved to {filepath}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about type detection."""
        if not self._is_fitted:
            return {}

        return {
            "feature_count": len(self._feature_names_in),
            "type_distribution": {
                t.value: len(feats)
                for t, feats in self._features_by_type.items()
            },
            "binary_features": self._features_by_type[FeatureType.BINARY],
            "categorical_features": self._features_by_type[FeatureType.CATEGORICAL],
            "discrete_features": self._features_by_type[FeatureType.DISCRETE_NUMERIC],
            "continuous_features": self._features_by_type[FeatureType.CONTINUOUS_NUMERIC],
            "id_like_features": self._features_by_type[FeatureType.ID_LIKE],
            "config": {
                "categorical_max": self.config.categorical_max,
                "discrete_max": self.config.discrete_max,
            },
        }

    def get_feature_report(self) -> pd.DataFrame:
        """Generate detailed report on all features with their types.

        Returns:
            DataFrame with one row per feature
        """
        self.check_is_fitted()

        records = []
        for feat, info in self._feature_type_info.items():
            records.append({
                "feature": info.name,
                "inferred_type": info.inferred_type.value,
                "dtype": info.dtype,
                "n_unique": info.n_unique,
                "n_non_null": info.n_non_null,
                "null_ratio": round(info.null_ratio, 4),
                "unique_ratio": round(info.unique_ratio, 4),
                "is_numeric": info.is_numeric_dtype,
                "confidence": info.confidence,
                "unique_values": str(info.unique_values[:5]) if info.unique_values else "[]",
                "original_type": info.original_type,
            })

        return pd.DataFrame(records)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage state for persistence."""
        base_dict = super().to_dict()
        base_dict.update({
            "config": {
                "binary_max": self.config.binary_max,
                "categorical_max": self.config.categorical_max,
                "discrete_max": self.config.discrete_max,
                "id_like_ratio": self.config.id_like_ratio,
            },
            "feature_types": {
                name: info.inferred_type.value
                for name, info in self._feature_type_info.items()
            },
            "features_by_type": {
                t.value: feats
                for t, feats in self._features_by_type.items()
            },
            "stage_log": self._stage_log.to_dict() if self._stage_log else None,
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: Optional[TypeDetectorConfig] = None) -> "TypeDetector":
        """Reconstruct stage from serialized state."""
        if config is None and "config" in data:
            config = TypeDetectorConfig(**data["config"])

        instance = cls(config=config)

        # Restore base state
        instance._is_fitted = data.get("is_fitted", False)
        instance._feature_names_in = data.get("feature_names_in", [])
        instance._feature_names_out = data.get("feature_names_out", [])

        # Restore features by type
        if "features_by_type" in data:
            for type_str, feats in data["features_by_type"].items():
                ftype = FeatureType(type_str)
                instance._features_by_type[ftype] = feats

        return instance
