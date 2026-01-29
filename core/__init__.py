"""
Core module for scorecard pipeline infrastructure.

Contains base classes, pipeline orchestrator and configuration utilities.
"""

from core.base import (
    PipelineStage,
    StageResult,
    StageStatus,
    PipelineConfig,
    PreprocessingConfig,
    BinningConfig,
    ClusteringConfig,
    SelectionConfig,
    ModelConfig,
    TypeDetectionConfig,
    ImputationConfig,
    PostprocessingConfig,
    FinalFilterConfig,
    DiagnosticsConfig,
    BinnerOptimization,
    SelectionMethod,
    ImputationStrategy,
    validate_target,
    validate_sample_type,
)

from core.pipeline import (
    ScorecardPipeline,
    PipelineResult,
    create_default_pipeline,
    create_pipeline_from_config,
)

from core.exceptions import (
    PipelineException,
    PipelineTrigger,
    DataQualityError,
    FeatureSelectionError,
    ModelTrainingError,
    ValidationError,
    TriggerManager,
    TriggerDetails,
    TriggerSeverity,
    TriggerCategory,
    TriggerCodes,
    get_trigger_manager,
    set_trigger_manager,
    create_no_features_trigger,
    create_data_quality_trigger,
)

__all__ = [
    # Base classes
    "PipelineStage",
    "StageResult",
    "StageStatus",
    # Pipeline
    "ScorecardPipeline",
    "PipelineResult",
    # Configs
    "PipelineConfig",
    "PreprocessingConfig",
    "TypeDetectionConfig",
    "ImputationConfig",
    "BinningConfig",
    "PostprocessingConfig",
    "ClusteringConfig",
    "SelectionConfig",
    "FinalFilterConfig",
    "ModelConfig",
    "DiagnosticsConfig",
    # Enums
    "BinnerOptimization",
    "SelectionMethod",
    "ImputationStrategy",
    # Utilities
    "validate_target",
    "validate_sample_type",
    "create_default_pipeline",
    "create_pipeline_from_config",
    # Exceptions and Triggers
    "PipelineException",
    "PipelineTrigger",
    "DataQualityError",
    "FeatureSelectionError",
    "ModelTrainingError",
    "ValidationError",
    "TriggerManager",
    "TriggerDetails",
    "TriggerSeverity",
    "TriggerCategory",
    "TriggerCodes",
    "get_trigger_manager",
    "set_trigger_manager",
    "create_no_features_trigger",
    "create_data_quality_trigger",
]
