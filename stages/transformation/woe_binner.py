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

Author: AURA Team
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
    tree_depth_used: int  # which depth gave best result
    has_nan_bin: bool
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
        enforce_monotonicity: whether WoE must be monotonic
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
        if self.power < 1:
            raise ValueError(f"power must be >= 1, got {self.power}")
        if self.good_mark == self.bad_mark:
            raise ValueError("good_mark and bad_mark cannot be equal")

    def get_binner_type(self) -> BinnerType:
        """Convert to BinnerType enum."""
        return BinnerType.IV if self.optimization_mode == "IV" else BinnerType.R2


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
        **kwargs
    ) -> "WoEBinnerStage":
        """Fit WoE binning to training data.

        Args:
            X: feature DataFrame
            y: target Series (binary 0/1)
            target_col: name for target column (used internally)
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

        # Check monotonicity of WoE values (excluding NaN bin)
        non_nan_woes = [b.woe for b in bins if not b.is_nan_bin]
        is_monotonic = self._check_monotonicity(non_nan_woes)

        return BinningInfo(
            feature_name=feature_name,
            woe_feature_name=woe_name,
            n_bins=len(bins),
            bins=bins,
            iv=binning_obj._iv if hasattr(binning_obj, "_iv") else 0,
            gini=binning_obj._gini if hasattr(binning_obj, "_gini") else 0,
            hhi=binning_obj._hhi if hasattr(binning_obj, "_hhi") else 0,
            r2=binning_obj._r2 if hasattr(binning_obj, "_r2") else 0,
            is_monotonic=is_monotonic,
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

    def _compute_summary_stats(self) -> None:
        """Compute summary statistics for the log."""
        if not self._binning_info:
            return

        iv_values = [info.iv for info in self._binning_info.values()]
        n_bins_list = [info.n_bins for info in self._binning_info.values()]

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
            "non_monotonic_features": sum(
                1 for info in self._binning_info.values() if not info.is_monotonic
            ),
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
