"""
Scorecard Pipeline Orchestrator.

This module provides the main ScorecardPipeline class that chains together
all processing stages and manages the end-to-end model building workflow.

The pipeline follows this sequence:
1. Preprocessing (cleaning, type detection, imputation)
2. Transformation (WoE binning, postprocessing)
3. Feature Selection (clustering, stepwise, final filtering)
4. Model Training (logistic regression)
5. Diagnostics (optional, run after training)

Author: AURA Team
"""

import time
import logging
from typing import Dict, Any, Optional, List, Type, Callable
from datetime import datetime
from dataclasses import dataclass, field
import json
import pickle

import pandas as pd
import numpy as np

from core.base import (
    PipelineStage,
    PipelineConfig,
    StageResult,
    StageStatus,
    validate_target,
    validate_sample_type,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Result Container
# =============================================================================

@dataclass
class PipelineResult:
    """Container for complete pipeline execution results.

    Stores results from all stages, final model, and metadata.
    Use this to inspect what happend during pipeline run.

    Attributes:
        stage_results: dict mapping stage name to StageResult
        final_features: list of features in final model
        final_model: trained sklearn model object
        coefficients: model coefficients with feature names
        scorecard: points-based scorecard (if generated)
        total_execution_time: how long entire pipeline took
        config: configuration used for this run
    """
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    final_features: List[str] = field(default_factory=list)
    final_model: Any = None
    coefficients: Dict[str, float] = field(default_factory=dict)
    scorecard: Optional[pd.DataFrame] = None
    total_execution_time: float = 0.0
    config: Optional[PipelineConfig] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "not_started"  # not_started, running, completed, failed

    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        """Get result for specific stage by name."""
        return self.stage_results.get(stage_name)

    def get_all_dropped_features(self) -> Dict[str, Dict[str, str]]:
        """Get all dropped features across all stages.

        Returns:
            Dict mapping stage_name -> {feature: reason}
        """
        result = {}
        for stage_name, stage_result in self.stage_results.items():
            if stage_result.dropped_features:
                result[stage_name] = stage_result.dropped_features
        return result

    def summary(self) -> str:
        """Generate human-readable summary of pipeline results."""
        lines = [
            "=" * 60,
            "PIPELINE EXECUTION SUMMARY",
            "=" * 60,
            f"Status: {self.status}",
            f"Total execution time: {self.total_execution_time:.2f}s",
            f"Final features: {len(self.final_features)}",
            "",
            "Stage Summary:",
            "-" * 40,
        ]

        for stage_name, result in self.stage_results.items():
            status_icon = "✓" if result.status == StageStatus.FITTED else "✗"
            lines.append(
                f"  {status_icon} {stage_name}: "
                f"{len(result.selected_features)} features, "
                f"{result.execution_time:.2f}s"
            )

        if self.final_features:
            lines.extend([
                "",
                "Final Model Features:",
                "-" * 40,
            ])
            for feat in self.final_features[:10]:
                coef = self.coefficients.get(feat, 0)
                lines.append(f"  {feat}: {coef:.4f}")
            if len(self.final_features) > 10:
                lines.append(f"  ... and {len(self.final_features) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result to dictionary (without model object)."""
        return {
            "final_features": self.final_features,
            "coefficients": self.coefficients,
            "total_execution_time": self.total_execution_time,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "stage_summaries": {
                name: {
                    "selected_features": res.selected_features,
                    "dropped_count": len(res.dropped_features),
                    "execution_time": res.execution_time,
                    "status": res.status.value,
                }
                for name, res in self.stage_results.items()
            },
        }


# =============================================================================
# Pipeline Orchestrator
# =============================================================================

class ScorecardPipeline:
    """Main pipeline orchestrator for scorecard model building.

    This class manages the entire modeling workflow from raw data to
    trained scorecard model. It chains together multiple processing
    stages and handles data flow between them.

    The pipeline is configurable via PipelineConfig and supports:
    - Skipping stages (set enabled=False in config)
    - Custom stage implementations (via register_stage)
    - Checkpointing and resuming
    - Comprehensive diagnostics

    Example:
        >>> config = PipelineConfig()
        >>> pipeline = ScorecardPipeline(config)
        >>> pipeline.fit(df, target_col="target")
        >>> result = pipeline.get_result()
        >>> print(result.summary())

    Attributes:
        config: pipeline configuration
        stages: ordered list of pipeline stages
        result: PipelineResult after fitting
        _is_fitted: whether pipeline has been fit
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with configuration.

        Args:
            config: PipelineConfig instance. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
        self.stages: List[PipelineStage] = []
        self.result: Optional[PipelineResult] = None
        self._is_fitted = False
        self._current_stage_idx = 0

        # these will be set during fit
        self._target_col: str = ""
        self._sample_type_col: str = ""
        self._original_columns: List[str] = []

        # stage registry for custom stages
        self._stage_registry: Dict[str, Type[PipelineStage]] = {}

        # callbacks for monitoring progress
        self._on_stage_start: Optional[Callable] = None
        self._on_stage_complete: Optional[Callable] = None

        self._build_stages()

    def _build_stages(self) -> None:
        """Initialize all pipeline stages based on configuration.

        This method creates instances of each stage using the
        corresponding config section. Override this method to
        customize stage initialization.

        Note: actual stage classes are imported here to avoid
        circular imports and allow lazy loading.
        """
        # We'll populate this as we implement each stage
        # For now, just store the stage configs for later instantiation

        self._stage_configs = [
            ("data_cleaner", self.config.preprocessing),
            ("type_detector", self.config.type_detection),
            ("missing_handler", self.config.imputation),
            ("woe_binner", self.config.binning),
            ("postprocessor", self.config.postprocessing),
            ("clustering", self.config.clustering),
            ("stepwise_selection", self.config.selection),
            ("final_filter", self.config.final_filter),
            ("model_trainer", self.config.model),
        ]

        # Stages will be instantiated lazily when first needed
        # This allows registering custom implementations before fit()
        self._stages_initialized = False

    def _initialize_stages(self) -> None:
        """Actually create stage instances.

        Called lazily before first fit() to allow custom stage registration.
        """
        if self._stages_initialized:
            return

        # Import stage classes here (will be implemented later)
        # For now we'll use a placeholder approach
        self.stages = []

        for stage_name, stage_config in self._stage_configs:
            stage_class = self._stage_registry.get(stage_name)
            if stage_class is not None:
                stage = stage_class(config=stage_config)
                self.stages.append(stage)
            else:
                # Stage not registered yet, will be added later
                logger.debug(f"Stage '{stage_name}' not registered, skipping")

        self._stages_initialized = True

    def register_stage(self, name: str, stage_class: Type[PipelineStage]) -> "ScorecardPipeline":
        """Register custom stage implementation.

        Use this to provide your own implementation of any stage
        or to add new stages to the pipeline.

        Args:
            name: stage name (must match one of the expected names)
            stage_class: class that inherits from PipelineStage

        Returns:
            self for method chaining

        Example:
            >>> pipeline.register_stage("woe_binner", MyCustomBinner)
        """
        if not issubclass(stage_class, PipelineStage):
            raise TypeError(
                f"stage_class must inherit from PipelineStage, "
                f"got {stage_class.__name__}"
            )

        self._stage_registry[name] = stage_class
        self._stages_initialized = False  # force re-initialization
        logger.info(f"Registered stage: {name} -> {stage_class.__name__}")
        return self

    def register_stages(self, stages: Dict[str, Type[PipelineStage]]) -> "ScorecardPipeline":
        """Register multiple stages at once.

        Args:
            stages: dict mapping stage names to classes

        Returns:
            self for method chaining
        """
        for name, stage_class in stages.items():
            self.register_stage(name, stage_class)
        return self

    def set_callbacks(
        self,
        on_stage_start: Optional[Callable[[str], None]] = None,
        on_stage_complete: Optional[Callable[[str, StageResult], None]] = None,
    ) -> "ScorecardPipeline":
        """Set callback functions for progress monitoring.

        Args:
            on_stage_start: called when stage begins, receives stage name
            on_stage_complete: called when stage ends, receives name and result

        Returns:
            self for method chaining
        """
        self._on_stage_start = on_stage_start
        self._on_stage_complete = on_stage_complete
        return self

    def fit(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        sample_type_col: Optional[str] = None,
        **kwargs
    ) -> "ScorecardPipeline":
        """Fit the entire pipeline on training data.

        Executes all stages in sequence, passing output from each
        stage as input to the next. Training data is filtered by
        sample_type == 0.

        Args:
            df: input DataFrame with features, target, and sample_type
            target_col: name of target column (overrides config)
            sample_type_col: name of sample_type column (overrides config)
            **kwargs: additional arguments passed to stages

        Returns:
            self: fitted pipeline

        Raises:
            ValueError: if input data is invalid
            RuntimeError: if a stage fails
        """
        start_time = time.time()

        # Use config values if not overriden
        self._target_col = target_col or self.config.preprocessing.target_col
        self._sample_type_col = sample_type_col or self.config.preprocessing.sample_type_col

        # Validate input data
        self._validate_input_data(df)

        # Initialize stages if not done yet
        self._initialize_stages()

        if not self.stages:
            raise RuntimeError(
                "No stages registered. Please register stage implementations "
                "using register_stage() before calling fit()."
            )

        # Initialize result container
        self.result = PipelineResult(config=self.config)
        self.result.status = "running"

        # Prepare data - separate features from target and metadata
        X, y, metadata_cols = self._prepare_data(df)

        self._original_columns = list(X.columns)
        logger.info(f"Starting pipeline with {len(X.columns)} features")

        # Run each stage sequentially
        current_X = X.copy()

        for idx, stage in enumerate(self.stages):
            self._current_stage_idx = idx
            stage_name = stage.name

            if self._on_stage_start:
                self._on_stage_start(stage_name)

            logger.info(f"Running stage {idx + 1}/{len(self.stages)}: {stage_name}")
            stage_start = time.time()

            try:
                # Fit and transform
                current_X = stage.fit_transform(current_X, y, **kwargs)

                # Collect result
                stage_time = time.time() - stage_start
                stage_result = stage.get_result(current_X, stage_time)
                self.result.stage_results[stage_name] = stage_result

                logger.info(
                    f"Stage '{stage_name}' complete: "
                    f"{len(stage_result.selected_features)} features remaining "
                    f"({stage_time:.2f}s)"
                )

                if self._on_stage_complete:
                    self._on_stage_complete(stage_name, stage_result)

            except Exception as e:
                logger.error(f"Stage '{stage_name}' failed: {str(e)}")
                self.result.status = "failed"
                raise RuntimeError(f"Pipeline failed at stage '{stage_name}': {e}") from e

        # Extract final model info
        self._finalize_result(current_X)

        self.result.total_execution_time = time.time() - start_time
        self.result.status = "completed"
        self._is_fitted = True

        logger.info(
            f"Pipeline completed in {self.result.total_execution_time:.2f}s "
            f"with {len(self.result.final_features)} final features"
        )

        return self

    def _validate_input_data(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame before processing.

        Checks:
        - DataFrame is not empty
        - Target column exists and is binary
        - Sample type column exists and has valid values
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if self._target_col not in df.columns:
            raise ValueError(
                f"Target column '{self._target_col}' not found. "
                f"Available columns: {list(df.columns)[:10]}..."
            )

        # Validate target
        validate_target(df[self._target_col])

        # Validate sample_type
        validate_sample_type(df, self._sample_type_col)

        # Check we have training data
        n_train = (df[self._sample_type_col] == 0).sum()
        if n_train == 0:
            raise ValueError("No training data found (sample_type == 0)")

        logger.info(
            f"Input validated: {len(df)} rows, {len(df.columns)} columns, "
            f"{n_train} training samples"
        )

    def _prepare_data(self, df: pd.DataFrame) -> tuple:
        """Separate features from target and metadata.

        Returns:
            tuple of (X, y, metadata_cols)
            - X: feature DataFrame (training data only)
            - y: target Series (training data only)
            - metadata_cols: list of non-feature columns
        """
        # Identify metadata columns that shouldnt be used as features
        metadata_cols = [
            self._target_col,
            self._sample_type_col,
        ]

        # Add ID columns from config
        if self.config.preprocessing.id_columns:
            metadata_cols.extend(self.config.preprocessing.id_columns)

        # Add date column if specified
        if self.config.preprocessing.date_col:
            metadata_cols.append(self.config.preprocessing.date_col)

        # Remove duplicates
        metadata_cols = list(set(col for col in metadata_cols if col in df.columns))

        # Filter to training data only for fitting
        train_mask = df[self._sample_type_col] == 0
        df_train = df[train_mask].copy()

        # Separate X and y
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        X = df_train[feature_cols]
        y = df_train[self._target_col]

        logger.info(
            f"Prepared data: {len(feature_cols)} features, "
            f"{len(metadata_cols)} metadata columns"
        )

        return X, y, metadata_cols

    def _finalize_result(self, final_X: pd.DataFrame) -> None:
        """Extract final model information after all stages complete."""
        self.result.final_features = list(final_X.columns)

        # Try to get model and coefficients from last stage
        if self.stages:
            last_stage = self.stages[-1]
            if hasattr(last_stage, "_model") and last_stage._model is not None:
                self.result.final_model = last_stage._model

                # Extract coefficients if available
                if hasattr(last_stage._model, "coef_"):
                    coefs = last_stage._model.coef_.flatten()
                    self.result.coefficients = dict(
                        zip(self.result.final_features, coefs)
                    )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline.

        Applies all transformations learned during fit() to new data.
        Used for scoring new observations.

        Args:
            df: DataFrame with same structure as training data

        Returns:
            Transformed DataFrame ready for scoring

        Raises:
            RuntimeError: if pipeline hasnt been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # Prepare data (but keep all rows, not just training)
        feature_cols = [c for c in df.columns if c in self._original_columns]
        X = df[feature_cols].copy()

        # Apply each stage's transform
        for stage in self.stages:
            X = stage.transform(X)

        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions for new data.

        Args:
            df: DataFrame with features

        Returns:
            Array of predicted probabilities

        Raises:
            RuntimeError: if pipeline not fitted or no model available
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        if self.result.final_model is None:
            raise RuntimeError("No model available. Check if model_trainer stage ran.")

        X_transformed = self.transform(df)
        return self.result.final_model.predict_proba(X_transformed)[:, 1]

    def predict_score(self, df: pd.DataFrame, pdo: float = 20, base_score: float = 600, base_odds: float = 50) -> np.ndarray:
        """Generate scorecard points for new data.

        Converts probability to points using standard scorecard formula:
        score = base_score - pdo * log(odds) / log(2)

        Args:
            df: DataFrame with features
            pdo: points to double odds (typically 20)
            base_score: score at base odds (typically 600)
            base_odds: odds at base score (typically 50:1)

        Returns:
            Array of scores
        """
        proba = self.predict(df)

        # Avoid division by zero
        proba = np.clip(proba, 1e-10, 1 - 1e-10)

        # Convert probability to odds
        odds = (1 - proba) / proba

        # Calculate score
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        scores = offset + factor * np.log(odds)

        return scores

    def get_result(self) -> PipelineResult:
        """Get pipeline execution result.

        Returns:
            PipelineResult with all stage results and final model

        Raises:
            RuntimeError: if pipeline hasnt been run
        """
        if self.result is None:
            raise RuntimeError("Pipeline hasn't been run yet. Call fit() first.")
        return self.result

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from final model.

        Returns DataFrame with features ranked by importance.
        For logistic regression, importance is based on coefficient magnitude.
        """
        if not self._is_fitted or not self.result.coefficients:
            raise RuntimeError("No fitted model available")

        importance_df = pd.DataFrame([
            {"feature": feat, "coefficient": coef, "abs_coefficient": abs(coef)}
            for feat, coef in self.result.coefficients.items()
        ])

        return importance_df.sort_values("abs_coefficient", ascending=False)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Collect diagnostics from all stages.

        Returns:
            Dictionary with diagnostics grouped by stage
        """
        if self.result is None:
            return {}

        diagnostics = {}
        for stage_name, stage_result in self.result.stage_results.items():
            diagnostics[stage_name] = stage_result.diagnostics

        return diagnostics

    # =========================================================================
    # Persistence Methods
    # =========================================================================

    def save(self, filepath: str) -> None:
        """Save fitted pipeline to file.

        Args:
            filepath: path to save pipeline (pickle format)
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        logger.info(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ScorecardPipeline":
        """Load pipeline from file.

        Args:
            filepath: path to saved pipeline

        Returns:
            Loaded ScorecardPipeline instance
        """
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)

        if not isinstance(pipeline, cls):
            raise TypeError(f"Loaded object is not a ScorecardPipeline")

        logger.info(f"Pipeline loaded from {filepath}")
        return pipeline

    def export_config(self, filepath: str) -> None:
        """Export pipeline configuration to JSON.

        Args:
            filepath: path to save config
        """
        self.config.save(filepath)
        logger.info(f"Config exported to {filepath}")

    # =========================================================================
    # Dunder methods
    # =========================================================================

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        n_stages = len(self.stages) if self._stages_initialized else "pending"
        return f"ScorecardPipeline(stages={n_stages}, status={status})"

    def __len__(self) -> int:
        """Return number of stages."""
        return len(self.stages)

    def __getitem__(self, idx: int) -> PipelineStage:
        """Get stage by index."""
        return self.stages[idx]

    def __iter__(self):
        """Iterate over stages."""
        return iter(self.stages)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_pipeline() -> ScorecardPipeline:
    """Create pipeline with default configuration.

    Returns:
        ScorecardPipeline with default settings
    """
    return ScorecardPipeline(PipelineConfig())


def create_pipeline_from_config(config_path: str) -> ScorecardPipeline:
    """Create pipeline from JSON config file.

    Args:
        config_path: path to JSON config file

    Returns:
        Configured ScorecardPipeline
    """
    config = PipelineConfig.load(config_path)
    return ScorecardPipeline(config)
