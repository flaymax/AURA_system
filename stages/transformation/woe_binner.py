"""
WoE Binner Stage - wraps existing Binner class as pipeline stage.

This stage applies Weight of Evidence (WoE) transformation to features
using decision tree-based optimal binning. The underlying Binner uses
trees of varying depths to find optimal splits, not simple percentiles.

Key features:
- Decision tree-based binning (smarter than percentile splits)
- Automatic NaN handling (separate bin for missing values)
- Monotonicity enforcement (optional)
- Optimization by IV or R²
- Comprehensive logging of all binning results
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
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
    get_trigger_manager,
    FeatureSelectionError,
)

# Import existing Binner classes
from pipeline.binner import Binner, BinnerType
from pipeline.fit_transform import Binning, BinningSettings


logger = logging.getLogger(__name__)


class MonotonicityMode(Enum):
    """How to handle non-monotonic WoE patterns.

    ENFORCE: Drop features that are not monotonic (strictest)
    WARN: Log warning but keep the feature (default)
    IGNORE: Do not check monotonicity at all
    """
    ENFORCE = "enforce"
    WARN = "warn"
    IGNORE = "ignore"


class MonotonicityDirection(Enum):
    """Direction of monotonic WoE relationship.

    INCREASING: WoE increases as feature value increases (e.g., income vs default)
    DECREASING: WoE decreases as feature value increases (e.g., debt vs default)
    NONE: Not monotonic or not checked
    """
    INCREASING = "increasing"
    DECREASING = "decreasing"
    NONE = "none"


class PSIMode(Enum):
    """How to handle features with high PSI (unstable distribution).

    ENFORCE: Drop features with PSI above threshold (strictest)
    WARN: Log warning but keep the feature (default)
    IGNORE: Do not calculate or check PSI
    """
    ENFORCE = "enforce"
    WARN = "warn"
    IGNORE = "ignore"


class PSILevel(Enum):
    """Classification of PSI values.

    Based on industry standard thresholds:
    - STABLE: PSI < 0.1 (no significant population shift)
    - MODERATE: 0.1 <= PSI < 0.25 (some shift, monitor closely)
    - UNSTABLE: PSI >= 0.25 (significant shift, feature may be unreliable)
    """
    STABLE = "stable"
    MODERATE = "moderate"
    UNSTABLE = "unstable"


# =============================================================================
# Binning Info Container
# =============================================================================

@dataclass
class BinInfo:
    """Information about a single bin (interval)."""
    bin_index: int
    lower_bound: Optional[float]  # None for NaN bin
    upper_bound: Optional[float]  # None for NaN bin
    is_nan_bin: bool
    woe: float
    n_goods: int
    n_bads: int
    n_total: int
    goods_share: float
    bads_share: float
    event_rate: float  # bads / total in this bin

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bin_index": self.bin_index,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "is_nan_bin": self.is_nan_bin,
            "woe": round(self.woe, 4),
            "n_goods": self.n_goods,
            "n_bads": self.n_bads,
            "n_total": self.n_total,
            "goods_share": round(self.goods_share, 4),
            "bads_share": round(self.bads_share, 4),
            "event_rate": round(self.event_rate, 4),
        }


@dataclass
class MonotonicityInfo:
    """Detailed information about WoE monotonicity for a feature."""
    is_monotonic: bool
    direction: str  # "increasing", "decreasing", or "none"
    n_violations: int  # number of bin transitions that violate monotonicity
    violation_indices: List[int]  # indices where violations occur
    woe_trend: List[float]  # WoE values in bin order (excluding NaN)
    severity: str  # "none", "minor" (1-2 violations), "major" (3+ violations)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_monotonic": self.is_monotonic,
            "direction": self.direction,
            "n_violations": self.n_violations,
            "violation_indices": self.violation_indices,
            "woe_trend": [round(w, 4) for w in self.woe_trend],
            "severity": self.severity,
        }


@dataclass
class PSIInfo:
    """Population Stability Index information for a feature.

    PSI measures the shift in distribution between train and test populations.
    Calculated at the bin level using the WoE binning structure.

    Standard interpretation:
    - PSI < 0.1: Stable, no significant shift
    - 0.1 <= PSI < 0.25: Moderate shift, needs monitoring
    - PSI >= 0.25: Significant shift, feature may be unstable
    """
    psi_value: float
    level: str  # "stable", "moderate", "unstable"
    train_distribution: List[float]  # % in each bin for train
    test_distribution: List[float]  # % in each bin for test
    bin_contributions: List[float]  # PSI contribution from each bin
    is_stable: bool  # True if PSI < threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psi_value": round(self.psi_value, 4),
            "level": self.level,
            "train_distribution": [round(x, 4) for x in self.train_distribution],
            "test_distribution": [round(x, 4) for x in self.test_distribution],
            "bin_contributions": [round(x, 4) for x in self.bin_contributions],
            "is_stable": self.is_stable,
        }


@dataclass
class BinningInfo:
    """Comprehensive binning information for a single feature.

    Contains all details about how a feature was binned,
    including split points, WoE values, and metrics.
    """
    feature_name: str
    woe_feature_name: str  # name after transformation (e.g., "income_woe")
    n_bins: int
    bins: List[BinInfo]
    iv: float  # Information Value
    gini: float
    hhi: float  # Herfindahl-Hirschman Index
    r2: float
    is_monotonic: bool
    monotonicity_info: Optional[MonotonicityInfo] = None
    psi_info: Optional[PSIInfo] = None  # PSI between train/test
    tree_depth_used: int = 0  # which depth gave best result
    has_nan_bin: bool = False
    status: str = "success"  # success, failed, skipped
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "woe_feature_name": self.woe_feature_name,
            "n_bins": self.n_bins,
            "bins": [b.to_dict() for b in self.bins],
            "iv": round(self.iv, 4),
            "gini": round(self.gini, 4) if self.gini else None,
            "hhi": round(self.hhi, 4) if self.hhi else None,
            "r2": round(self.r2, 4) if self.r2 else None,
            "is_monotonic": self.is_monotonic,
            "monotonicity_info": self.monotonicity_info.to_dict() if self.monotonicity_info else None,
            "psi_info": self.psi_info.to_dict() if self.psi_info else None,
            "tree_depth_used": self.tree_depth_used,
            "has_nan_bin": self.has_nan_bin,
            "status": self.status,
            "error_message": self.error_message,
        }

    def get_split_points(self) -> List[float]:
        """Get list of split points (bin boundaries)."""
        points = []
        for b in self.bins:
            if not b.is_nan_bin:
                if b.lower_bound is not None:
                    points.append(b.lower_bound)
                if b.upper_bound is not None:
                    points.append(b.upper_bound)
        return sorted(set(points))


@dataclass
class WoEBinnerLog:
    """Structured log for WoE Binner stage execution."""
    stage_name: str
    stage_type: str = "transformation"
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    input_feature_count: int = 0
    output_feature_count: int = 0
    input_sample_count: int = 0
    binning_info: Dict[str, BinningInfo] = field(default_factory=dict)
    features_binned: List[str] = field(default_factory=list)
    features_failed: List[str] = field(default_factory=list)
    features_skipped: List[str] = field(default_factory=list)
    config_used: Dict[str, Any] = field(default_factory=dict)
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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
            "output_feature_count": self.output_feature_count,
            "input_sample_count": self.input_sample_count,
            "binning_info": {k: v.to_dict() for k, v in self.binning_info.items()},
            "features_binned": self.features_binned,
            "features_failed": self.features_failed,
            "features_skipped": self.features_skipped,
            "config_used": self.config_used,
            "summary_stats": self.summary_stats,
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(f"[{self.stage_name}] {message}")

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        logger.error(f"[{self.stage_name}] {message}")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class WoEBinnerConfig:
    """Configuration for WoE Binner stage.

    Attributes:
        optimization_mode: IV or R2 optimization
        power: maximum tree depth for splitting (higher = more bins possible)
        min_leaf_ratio: minimum fraction of samples in each bin
        enforce_monotonicity: whether to request monotonic binning from the binner
        monotonicity_mode: how to handle non-monotonic results:
            - "enforce": drop features that are not monotonic
            - "warn": log warning but keep the feature (default)
            - "ignore": do not check monotonicity
        psi_mode: how to handle features with high PSI (train vs test shift):
            - "enforce": drop features with PSI above threshold
            - "warn": log warning but keep the feature (default)
            - "ignore": do not calculate or check PSI
        psi_threshold: PSI value above which a feature is considered unstable (default 0.25)
        min_iv: minimum IV to keep feature after binning
        good_mark: target value for "good" class (default 0)
        bad_mark: target value for "bad" class (default 1)
        exclude_features: features to skip binning
        verbose: whether to log detailed progress
    """
    optimization_mode: str = "IV"  # "IV" or "R2"
    power: int = 3  # max tree depth
    min_leaf_ratio: float = 0.1  # min samples in leaf
    enforce_monotonicity: bool = True
    monotonicity_mode: str = "warn"  # "enforce", "warn", or "ignore"
    psi_mode: str = "warn"  # "enforce", "warn", or "ignore"
    psi_threshold: float = 0.25  # PSI above this is considered unstable
    min_iv: float = 0.02  # drop features with IV below this
    good_mark: int = 0
    bad_mark: int = 1
    exclude_features: List[str] = field(default_factory=list)
    verbose: bool = True

    def __post_init__(self):
        if self.optimization_mode not in ("IV", "R2"):
            raise ValueError(
                f"optimization_mode must be 'IV' or 'R2', got {self.optimization_mode}"
            )
        if self.monotonicity_mode not in ("enforce", "warn", "ignore"):
            raise ValueError(
                f"monotonicity_mode must be 'enforce', 'warn', or 'ignore', "
                f"got {self.monotonicity_mode}"
            )
        if self.psi_mode not in ("enforce", "warn", "ignore"):
            raise ValueError(
                f"psi_mode must be 'enforce', 'warn', or 'ignore', "
                f"got {self.psi_mode}"
            )
        if self.power < 1:
            raise ValueError(f"power must be >= 1, got {self.power}")
        if self.good_mark == self.bad_mark:
            raise ValueError("good_mark and bad_mark cannot be equal")
        if self.psi_threshold <= 0:
            raise ValueError(f"psi_threshold must be positive, got {self.psi_threshold}")

    def get_binner_type(self) -> BinnerType:
        """Convert to BinnerType enum."""
        return BinnerType.IV if self.optimization_mode == "IV" else BinnerType.R2

    def get_monotonicity_mode(self) -> MonotonicityMode:
        """Convert string to MonotonicityMode enum."""
        return MonotonicityMode(self.monotonicity_mode)

    def get_psi_mode(self) -> PSIMode:
        """Convert string to PSIMode enum."""
        return PSIMode(self.psi_mode)


# =============================================================================
# WoE Binner Stage
# =============================================================================

class WoEBinnerStage(PipelineStage):
    """WoE binning stage using decision tree-based optimal splits.

    This stage wraps the existing Binner class and provides:
    - Integration with pipeline infrastructure
    - Comprehensive logging of all binning results
    - Feature filtering based on IV threshold
    - Consistent fit/transform interface

    The underlying algorithm uses decision trees to find optimal split points,
    not simple percentile-based binning. This results in more predictive bins.

    Example:
        >>> config = WoEBinnerConfig(power=3, min_iv=0.02)
        >>> binner_stage = WoEBinnerStage(config)
        >>> binner_stage.fit(X, y)
        >>> X_woe = binner_stage.transform(X)
        >>> binning_info = binner_stage.get_binning_info("income")
    """

    name = "woe_binner"

    def __init__(self, config: Optional[WoEBinnerConfig] = None):
        """Initialize WoE Binner stage.

        Args:
            config: WoEBinnerConfig instance, or None for defaults
        """
        super().__init__(config)
        self.config: WoEBinnerConfig = config or WoEBinnerConfig()

        # The underlying binner object (from existing code)
        self._binner: Optional[Binner] = None

        # Detailed binning information for each feature
        self._binning_info: Dict[str, BinningInfo] = {}

        # Mapping of original feature names to WoE feature names
        self._feature_name_mapping: Dict[str, str] = {}

        # Features that passed IV threshold
        self._selected_features: List[str] = []

        # Stage log
        self._stage_log: Optional[WoEBinnerLog] = None

        # Store target column name for transform
        self._target_col: Optional[str] = None

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        target_col: str = "target",
        sample_type: Optional[pd.Series] = None,
        **kwargs
    ) -> "WoEBinnerStage":
        """Fit WoE binning to training data.

        Args:
            X: feature DataFrame
            y: target Series (binary 0/1)
            target_col: name for target column (used internally)
            sample_type: Series indicating sample type (0=train, 1=test, 2=valid)
                        Used for PSI calculation. If None, PSI is not calculated.
            **kwargs: additional arguments

        Returns:
            self: fitted stage
        """
        self._validate_input(X, check_features=False)

        if y is None:
            raise ValueError("Target y is required for WoE binning")

        self._feature_names_in = list(X.columns)
        n_features = len(self._feature_names_in)
        n_samples = len(X)
        self._target_col = target_col

        # Initialize stage log
        self._stage_log = WoEBinnerLog(
            stage_name=self.name,
            input_feature_count=n_features,
            input_sample_count=n_samples,
            config_used={
                "optimization_mode": self.config.optimization_mode,
                "power": self.config.power,
                "min_leaf_ratio": self.config.min_leaf_ratio,
                "enforce_monotonicity": self.config.enforce_monotonicity,
                "monotonicity_mode": self.config.monotonicity_mode,
                "psi_mode": self.config.psi_mode,
                "psi_threshold": self.config.psi_threshold,
                "min_iv": self.config.min_iv,
            }
        )

        self._log_info(f"Fitting WoE binner on {n_features} features, {n_samples} samples")
        self._log_info(f"Config: power={self.config.power}, mode={self.config.optimization_mode}")

        # Prepare data for Binner (needs target as column)
        df_with_target = X.copy()
        df_with_target[target_col] = y.values

        # Create binner instance
        self._binner = Binner(
            binner_type=self.config.get_binner_type(),
            good_mark=self.config.good_mark,
            bad_mark=self.config.bad_mark,
        )

        # Create binning settings for each feature
        binning_settings = []
        for feat in self._feature_names_in:
            if feat not in self.config.exclude_features:
                settings = BinningSettings(
                    variable_name=feat,
                    monotone=self.config.enforce_monotonicity,
                    min_leaf_ratio=self.config.min_leaf_ratio,
                )
                binning_settings.append(settings)

        # Fit the binner
        try:
            self._binner.fit(
                data=df_with_target,
                target=target_col,
                power=self.config.power,
                binning_settings=binning_settings,
                exclude=self.config.exclude_features,
                verbose=self.config.verbose,
            )
        except Exception as e:
            self._stage_log.add_error(f"Binner fitting failed: {str(e)}")
            raise

        # Extract and log binning information
        self._extract_binning_info()

        # Filter features by IV threshold
        self._filter_by_iv()

        # Filter or warn about non-monotonic features
        self._filter_by_monotonicity()

        # Calculate and filter by PSI if sample_type is provided
        if sample_type is not None and self.config.get_psi_mode() != PSIMode.IGNORE:
            train_mask = sample_type == 0
            test_mask = sample_type == 1
            if train_mask.sum() > 0 and test_mask.sum() > 0:
                self._calculate_psi(X, train_mask, test_mask)
                self._filter_by_psi()
            else:
                self._stage_log.add_warning(
                    "PSI calculation skipped: insufficient train or test samples"
                )
        elif self.config.get_psi_mode() != PSIMode.IGNORE:
            self._stage_log.add_warning(
                "PSI calculation skipped: sample_type not provided"
            )

        # Update log
        self._stage_log.features_binned = list(self._binning_info.keys())
        self._stage_log.output_feature_count = len(self._selected_features)
        self._stage_log.binning_info = self._binning_info
        self._compute_summary_stats()

        self._feature_names_out = self._selected_features
        self._is_fitted = True
        self._stage_log.completed_at = datetime.now()

        # Log summary
        self._log_binning_summary()

        return self

    def _extract_binning_info(self) -> None:
        """Extract detailed binning information from fitted Binner."""
        fitted_bins = self._binner.get_fitted_bins()

        for binning_obj in fitted_bins:
            try:
                info = self._extract_single_binning_info(binning_obj)
                self._binning_info[info.feature_name] = info
                self._feature_name_mapping[info.feature_name] = info.woe_feature_name
            except Exception as e:
                self._stage_log.add_warning(
                    f"Failed to extract info for {binning_obj._name}: {str(e)}"
                )

    def _extract_single_binning_info(self, binning_obj: Binning) -> BinningInfo:
        """Extract binning info from a single Binning object."""
        feature_name = binning_obj._name
        woe_name = f"{feature_name}_woe"

        gaps = binning_obj._gaps
        woes = binning_obj._woes
        counts = binning_obj._counts
        shares = binning_obj._shares

        bins = []
        has_nan_bin = False

        for i, (gap, woe, count, share) in enumerate(zip(gaps, woes, counts, shares)):
            is_nan = gap[0] is None and gap[1] is None
            if is_nan:
                has_nan_bin = True

            n_goods = count[0] if len(count) > 0 else 0
            n_bads = count[1] if len(count) > 1 else 0
            n_total = n_goods + n_bads
            event_rate = n_bads / n_total if n_total > 0 else 0

            goods_share = share[0] if len(share) > 0 else 0
            bads_share = share[1] if len(share) > 1 else 0

            bin_info = BinInfo(
                bin_index=i,
                lower_bound=gap[0] if not is_nan else None,
                upper_bound=gap[1] if not is_nan else None,
                is_nan_bin=is_nan,
                woe=woe,
                n_goods=n_goods,
                n_bads=n_bads,
                n_total=n_total,
                goods_share=goods_share,
                bads_share=bads_share,
                event_rate=event_rate,
            )
            bins.append(bin_info)

        # Analyze monotonicity of WoE values (excluding NaN bin)
        non_nan_woes = [b.woe for b in bins if not b.is_nan_bin]
        monotonicity_info = self._analyze_monotonicity(non_nan_woes)

        return BinningInfo(
            feature_name=feature_name,
            woe_feature_name=woe_name,
            n_bins=len(bins),
            bins=bins,
            iv=binning_obj._iv if hasattr(binning_obj, "_iv") else 0,
            gini=binning_obj._gini if hasattr(binning_obj, "_gini") else 0,
            hhi=binning_obj._hhi if hasattr(binning_obj, "_hhi") else 0,
            r2=binning_obj._r2 if hasattr(binning_obj, "_r2") else 0,
            is_monotonic=monotonicity_info.is_monotonic,
            monotonicity_info=monotonicity_info,
            tree_depth_used=self.config.power,  # actual depth is not exposed
            has_nan_bin=has_nan_bin,
            status="success",
        )

    def _check_monotonicity(self, values: List[float]) -> bool:
        """Check if values are monotonically increasing or decreasing."""
        if len(values) <= 1:
            return True

        increasing = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        decreasing = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

        return increasing or decreasing

    def _analyze_monotonicity(self, woe_values: List[float]) -> MonotonicityInfo:
        """Analyze monotonicity of WoE values in detail.

        Returns comprehensive information about the monotonicity pattern,
        including direction, violations, and severity.

        Args:
            woe_values: List of WoE values in bin order (excluding NaN bin)

        Returns:
            MonotonicityInfo with detailed analysis
        """
        if len(woe_values) <= 1:
            return MonotonicityInfo(
                is_monotonic=True,
                direction=MonotonicityDirection.NONE.value,
                n_violations=0,
                violation_indices=[],
                woe_trend=woe_values,
                severity="none"
            )

        # Count increasing and decreasing transitions
        increasing_count = 0
        decreasing_count = 0
        violation_indices = []

        for i in range(len(woe_values) - 1):
            if woe_values[i + 1] > woe_values[i]:
                increasing_count += 1
            elif woe_values[i + 1] < woe_values[i]:
                decreasing_count += 1
            # Equal values do not count as violations

        # Determine expected direction based on majority of transitions
        if increasing_count > decreasing_count:
            expected_direction = MonotonicityDirection.INCREASING
            # Violations are decreasing transitions
            for i in range(len(woe_values) - 1):
                if woe_values[i + 1] < woe_values[i]:
                    violation_indices.append(i)
        elif decreasing_count > increasing_count:
            expected_direction = MonotonicityDirection.DECREASING
            # Violations are increasing transitions
            for i in range(len(woe_values) - 1):
                if woe_values[i + 1] > woe_values[i]:
                    violation_indices.append(i)
        else:
            # No clear direction
            expected_direction = MonotonicityDirection.NONE
            violation_indices = []

        n_violations = len(violation_indices)
        is_monotonic = n_violations == 0

        # Determine severity
        if n_violations == 0:
            severity = "none"
        elif n_violations <= 2:
            severity = "minor"
        else:
            severity = "major"

        return MonotonicityInfo(
            is_monotonic=is_monotonic,
            direction=expected_direction.value if is_monotonic else MonotonicityDirection.NONE.value,
            n_violations=n_violations,
            violation_indices=violation_indices,
            woe_trend=woe_values,
            severity=severity
        )

    def _filter_by_iv(self) -> None:
        """Filter features by minimum IV threshold."""
        self._selected_features = []
        dropped_low_iv = []

        for feat_name, info in self._binning_info.items():
            if info.iv >= self.config.min_iv:
                self._selected_features.append(info.woe_feature_name)
            else:
                dropped_low_iv.append((feat_name, info.iv))
                self._stage_log.features_skipped.append(feat_name)

        if dropped_low_iv:
            self._stage_log.add_warning(
                f"{len(dropped_low_iv)} features dropped due to low IV (<{self.config.min_iv}): "
                f"{[f'{n} (IV={iv:.4f})' for n, iv in dropped_low_iv[:5]]}"
                f"{'...' if len(dropped_low_iv) > 5 else ''}"
            )

    def _filter_by_monotonicity(self) -> None:
        """Filter or warn about non-monotonic features based on config.

        Depending on monotonicity_mode:
        - "enforce": Remove non-monotonic features from selected list
        - "warn": Log warnings but keep features
        - "ignore": Do nothing
        """
        mode = self.config.get_monotonicity_mode()

        if mode == MonotonicityMode.IGNORE:
            return

        non_monotonic_features = []
        for feat_name, info in self._binning_info.items():
            if not info.is_monotonic and info.woe_feature_name in self._selected_features:
                mono_info = info.monotonicity_info
                non_monotonic_features.append({
                    "name": feat_name,
                    "woe_name": info.woe_feature_name,
                    "iv": info.iv,
                    "n_violations": mono_info.n_violations if mono_info else 0,
                    "severity": mono_info.severity if mono_info else "unknown",
                    "direction": mono_info.direction if mono_info else "none",
                })

        if not non_monotonic_features:
            return

        # Build warning message
        warning_details = []
        for feat in non_monotonic_features[:10]:
            warning_details.append(
                f"{feat['name']} (IV={feat['iv']:.4f}, violations={feat['n_violations']}, "
                f"severity={feat['severity']})"
            )

        if mode == MonotonicityMode.ENFORCE:
            # Remove non-monotonic features
            dropped_count = 0
            for feat in non_monotonic_features:
                if feat["woe_name"] in self._selected_features:
                    self._selected_features.remove(feat["woe_name"])
                    self._stage_log.features_skipped.append(feat["name"])
                    dropped_count += 1

            self._stage_log.add_warning(
                f"MONOTONICITY ENFORCED: {dropped_count} non-monotonic features dropped: "
                f"{warning_details}"
                f"{'...' if len(non_monotonic_features) > 10 else ''}"
            )

        elif mode == MonotonicityMode.WARN:
            # Just log warning, keep features
            self._stage_log.add_warning(
                f"MONOTONICITY WARNING: {len(non_monotonic_features)} features have "
                f"non-monotonic WoE patterns (kept in model): {warning_details}"
                f"{'...' if len(non_monotonic_features) > 10 else ''}"
            )

    def _calculate_psi(
        self,
        X: pd.DataFrame,
        train_mask: pd.Series,
        test_mask: pd.Series
    ) -> None:
        """Calculate PSI (Population Stability Index) for each feature.

        PSI measures distribution shift between train and test populations
        using the WoE bin structure. High PSI indicates the feature distribution
        has shifted significantly, which may affect model stability.

        Formula: PSI = Σ (Test% - Train%) * ln(Test% / Train%)

        Args:
            X: Feature DataFrame
            train_mask: Boolean mask for training samples
            test_mask: Boolean mask for test samples
        """
        mode = self.config.get_psi_mode()

        if mode == PSIMode.IGNORE:
            return

        # Minimum percentage to avoid log(0) issues
        MIN_PCT = 0.0001

        for feat_name, info in self._binning_info.items():
            if info.status != "success":
                continue

            try:
                feature_values = X[feat_name]
                train_values = feature_values[train_mask]
                test_values = feature_values[test_mask]

                # Get bin boundaries from binning info
                bins = info.bins
                train_distribution = []
                test_distribution = []
                bin_contributions = []

                for bin_info in bins:
                    if bin_info.is_nan_bin:
                        # Count NaN values
                        train_count = train_values.isna().sum()
                        test_count = test_values.isna().sum()
                    else:
                        # Count values in this bin range
                        lower = bin_info.lower_bound
                        upper = bin_info.upper_bound

                        if lower == float('-inf') or lower is None:
                            train_count = (train_values <= upper).sum()
                            test_count = (test_values <= upper).sum()
                        elif upper == float('inf') or upper is None:
                            train_count = (train_values > lower).sum()
                            test_count = (test_values > lower).sum()
                        else:
                            train_count = ((train_values > lower) & (train_values <= upper)).sum()
                            test_count = ((test_values > lower) & (test_values <= upper)).sum()

                    # Calculate percentages
                    train_pct = max(train_count / len(train_values), MIN_PCT) if len(train_values) > 0 else MIN_PCT
                    test_pct = max(test_count / len(test_values), MIN_PCT) if len(test_values) > 0 else MIN_PCT

                    train_distribution.append(train_pct)
                    test_distribution.append(test_pct)

                    # PSI contribution from this bin
                    contribution = (test_pct - train_pct) * np.log(test_pct / train_pct)
                    bin_contributions.append(contribution)

                # Total PSI
                psi_value = sum(bin_contributions)

                # Classify PSI level
                if psi_value < 0.1:
                    level = PSILevel.STABLE.value
                    is_stable = True
                elif psi_value < 0.25:
                    level = PSILevel.MODERATE.value
                    is_stable = psi_value < self.config.psi_threshold
                else:
                    level = PSILevel.UNSTABLE.value
                    is_stable = psi_value < self.config.psi_threshold

                # Store PSI info
                info.psi_info = PSIInfo(
                    psi_value=psi_value,
                    level=level,
                    train_distribution=train_distribution,
                    test_distribution=test_distribution,
                    bin_contributions=bin_contributions,
                    is_stable=is_stable,
                )

            except Exception as e:
                self._stage_log.add_warning(
                    f"Failed to calculate PSI for {feat_name}: {str(e)}"
                )

    def _filter_by_psi(self) -> None:
        """Filter or warn about features with high PSI based on config.

        Depending on psi_mode:
        - "enforce": Remove features with PSI above threshold
        - "warn": Log warnings but keep features
        - "ignore": Do nothing (PSI not even calculated)
        """
        mode = self.config.get_psi_mode()

        if mode == PSIMode.IGNORE:
            return

        unstable_features = []
        for feat_name, info in self._binning_info.items():
            if info.psi_info and not info.psi_info.is_stable:
                if info.woe_feature_name in self._selected_features:
                    unstable_features.append({
                        "name": feat_name,
                        "woe_name": info.woe_feature_name,
                        "iv": info.iv,
                        "psi": info.psi_info.psi_value,
                        "level": info.psi_info.level,
                    })

        if not unstable_features:
            return

        # Build warning message
        warning_details = []
        for feat in unstable_features[:10]:
            warning_details.append(
                f"{feat['name']} (IV={feat['iv']:.4f}, PSI={feat['psi']:.4f}, "
                f"level={feat['level']})"
            )

        if mode == PSIMode.ENFORCE:
            # Remove unstable features
            dropped_count = 0
            for feat in unstable_features:
                if feat["woe_name"] in self._selected_features:
                    self._selected_features.remove(feat["woe_name"])
                    self._stage_log.features_skipped.append(feat["name"])
                    dropped_count += 1

            self._stage_log.add_warning(
                f"PSI ENFORCED: {dropped_count} features dropped due to high PSI "
                f"(>{self.config.psi_threshold}): {warning_details}"
                f"{'...' if len(unstable_features) > 10 else ''}"
            )

        elif mode == PSIMode.WARN:
            # Just log warning, keep features
            self._stage_log.add_warning(
                f"PSI WARNING: {len(unstable_features)} features have high PSI "
                f"(>{self.config.psi_threshold}, kept in model): {warning_details}"
                f"{'...' if len(unstable_features) > 10 else ''}"
            )

    def _compute_summary_stats(self) -> None:
        """Compute summary statistics for the log."""
        if not self._binning_info:
            return

        iv_values = [info.iv for info in self._binning_info.values()]
        n_bins_list = [info.n_bins for info in self._binning_info.values()]

        # Count monotonicity by severity
        mono_by_severity = {"none": 0, "minor": 0, "major": 0}
        for info in self._binning_info.values():
            if info.monotonicity_info:
                mono_by_severity[info.monotonicity_info.severity] = \
                    mono_by_severity.get(info.monotonicity_info.severity, 0) + 1

        # Count monotonicity by direction
        mono_by_direction = {"increasing": 0, "decreasing": 0, "none": 0}
        for info in self._binning_info.values():
            if info.monotonicity_info and info.is_monotonic:
                direction = info.monotonicity_info.direction
                mono_by_direction[direction] = mono_by_direction.get(direction, 0) + 1

        # Count PSI by level
        psi_by_level = {"stable": 0, "moderate": 0, "unstable": 0}
        psi_values = []
        for info in self._binning_info.values():
            if info.psi_info:
                psi_by_level[info.psi_info.level] = psi_by_level.get(info.psi_info.level, 0) + 1
                psi_values.append(info.psi_info.psi_value)

        self._stage_log.summary_stats = {
            "total_features_processed": len(self._binning_info),
            "features_selected": len(self._selected_features),
            "features_dropped_low_iv": len(self._stage_log.features_skipped),
            "iv_stats": {
                "min": round(min(iv_values), 4),
                "max": round(max(iv_values), 4),
                "mean": round(np.mean(iv_values), 4),
                "median": round(np.median(iv_values), 4),
            },
            "bins_stats": {
                "min": min(n_bins_list),
                "max": max(n_bins_list),
                "mean": round(np.mean(n_bins_list), 2),
            },
            "features_with_nan_bin": sum(
                1 for info in self._binning_info.values() if info.has_nan_bin
            ),
            "monotonicity": {
                "n_monotonic": sum(
                    1 for info in self._binning_info.values() if info.is_monotonic
                ),
                "n_non_monotonic": sum(
                    1 for info in self._binning_info.values() if not info.is_monotonic
                ),
                "by_severity": mono_by_severity,
                "by_direction": mono_by_direction,
                "mode": self.config.monotonicity_mode,
            },
            "psi": {
                "n_calculated": len(psi_values),
                "by_level": psi_by_level,
                "psi_stats": {
                    "min": round(min(psi_values), 4) if psi_values else None,
                    "max": round(max(psi_values), 4) if psi_values else None,
                    "mean": round(np.mean(psi_values), 4) if psi_values else None,
                    "median": round(np.median(psi_values), 4) if psi_values else None,
                },
                "mode": self.config.psi_mode,
                "threshold": self.config.psi_threshold,
            },
        }

    def _log_binning_summary(self) -> None:
        """Log detailed summary of binning results."""
        self._log_info("=" * 60)
        self._log_info("WOE BINNING SUMMARY")
        self._log_info("=" * 60)

        # Sort features by IV
        sorted_features = sorted(
            self._binning_info.items(),
            key=lambda x: x[1].iv,
            reverse=True
        )

        # Top features by IV
        self._log_info(f"\nTop features by IV (showing up to 15):")
        for feat_name, info in sorted_features[:15]:
            status = "✓" if info.woe_feature_name in self._selected_features else "✗"
            mono = "mono" if info.is_monotonic else "non-mono"
            nan_str = "+NaN" if info.has_nan_bin else ""
            self._log_info(
                f"  {status} {feat_name}: IV={info.iv:.4f}, "
                f"bins={info.n_bins}{nan_str}, {mono}"
            )

        # Detailed bin info for top 5 features
        self._log_info(f"\nDetailed binning for top 5 features:")
        for feat_name, info in sorted_features[:5]:
            self._log_info(f"\n  [{feat_name}] IV={info.iv:.4f}, Gini={info.gini:.4f}")
            for b in info.bins:
                if b.is_nan_bin:
                    bounds = "NaN"
                else:
                    bounds = f"({b.lower_bound:.2f}, {b.upper_bound:.2f}]"
                self._log_info(
                    f"    Bin {b.bin_index}: {bounds} -> WoE={b.woe:.4f}, "
                    f"n={b.n_total}, rate={b.event_rate:.2%}"
                )

        self._log_info("=" * 60)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to WoE values.

        Args:
            X: feature DataFrame

        Returns:
            DataFrame with WoE-transformed features
        """
        self.check_is_fitted()

        if self._binner is None:
            raise RuntimeError("Binner not fitted")

        # Need to add a dummy target column for the transform method
        df_with_dummy = X.copy()
        df_with_dummy[self._target_col] = 0  # dummy value, not used in transform

        # Use the binner's transform method
        transformed = self._binner.transform(df_with_dummy)

        # Select only WoE columns that passed IV threshold
        woe_cols = [col for col in transformed.columns if col.endswith("_woe")]
        selected_woe_cols = [col for col in woe_cols if col in self._selected_features]

        result = transformed[selected_woe_cols].copy()

        self._log_info(f"Transform: {len(X.columns)} features -> {len(result.columns)} WoE features")

        return result

    # =========================================================================
    # Accessor Methods
    # =========================================================================

    def get_binning_info(self, feature_name: str) -> Optional[BinningInfo]:
        """Get detailed binning info for a specific feature."""
        return self._binning_info.get(feature_name)

    def get_all_binning_info(self) -> Dict[str, BinningInfo]:
        """Get binning info for all features."""
        return self._binning_info.copy()

    def get_iv_ranking(self) -> pd.DataFrame:
        """Get features ranked by IV."""
        self.check_is_fitted()

        records = []
        for feat_name, info in self._binning_info.items():
            records.append({
                "feature": feat_name,
                "woe_feature": info.woe_feature_name,
                "iv": info.iv,
                "gini": info.gini,
                "n_bins": info.n_bins,
                "is_monotonic": info.is_monotonic,
                "has_nan_bin": info.has_nan_bin,
                "selected": info.woe_feature_name in self._selected_features,
            })

        df = pd.DataFrame(records)
        return df.sort_values("iv", ascending=False).reset_index(drop=True)

    def get_woe_mapping(self, feature_name: str) -> Optional[Dict[str, float]]:
        """Get WoE mapping for a feature (bin description -> WoE value)."""
        info = self._binning_info.get(feature_name)
        if not info:
            return None

        mapping = {}
        for b in info.bins:
            if b.is_nan_bin:
                key = "NaN"
            else:
                key = f"({b.lower_bound}, {b.upper_bound}]"
            mapping[key] = b.woe

        return mapping

    def get_stage_log(self) -> Optional[WoEBinnerLog]:
        """Get the comprehensive stage log."""
        return self._stage_log

    def save_stage_log(self, filepath: str) -> None:
        """Save stage log to JSON file."""
        if self._stage_log is None:
            raise RuntimeError("No stage log available. Call fit() first.")
        self._stage_log.save(filepath)
        self._log_info(f"Stage log saved to {filepath}")

    def get_binner(self) -> Optional[Binner]:
        """Get the underlying Binner object for advanced usage."""
        return self._binner

    def get_sql(self) -> str:
        """Generate SQL for WoE transformation."""
        if self._binner is None:
            raise RuntimeError("Binner not fitted")
        return self._binner.to_sql()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information."""
        if not self._is_fitted:
            return {}

        return {
            "input_feature_count": len(self._feature_names_in),
            "output_feature_count": len(self._selected_features),
            "features_binned": len(self._binning_info),
            "features_dropped_low_iv": len(self._stage_log.features_skipped) if self._stage_log else 0,
            "config": {
                "power": self.config.power,
                "optimization_mode": self.config.optimization_mode,
                "min_iv": self.config.min_iv,
            },
            "summary_stats": self._stage_log.summary_stats if self._stage_log else {},
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize stage state."""
        base_dict = super().to_dict()
        base_dict.update({
            "config": {
                "optimization_mode": self.config.optimization_mode,
                "power": self.config.power,
                "min_iv": self.config.min_iv,
                "enforce_monotonicity": self.config.enforce_monotonicity,
            },
            "feature_name_mapping": self._feature_name_mapping,
            "selected_features": self._selected_features,
            "binning_info": {k: v.to_dict() for k, v in self._binning_info.items()},
            "stage_log": self._stage_log.to_dict() if self._stage_log else None,
        })
        return base_dict
