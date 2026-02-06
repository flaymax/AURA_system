"""
Pipeline stages module.

Contains all pipeline stage implementations organized by category:
- preprocessing: data cleaning, type detection, imputation
- transformation: WoE binning, postprocessing
- selection: clustering, stepwise selection, final filtering
- modeling: model training, points-based scoring
- visualization: performance charts, distribution plots, AR/BR analysis
"""

from stages.preprocessing import (
    DataCleaner,
    DataCleanerConfig,
    TypeDetector,
    TypeDetectorConfig,
    FeatureType,
    FeatureTypeInfo,
    RejectInferenceStage,
    RejectInferenceInfo,
    RejectInferenceLog,
)

from stages.transformation import (
    WoEBinnerStage,
    WoEBinnerConfig,
    BinningInfo,
    MonotonicityInfo,
    MonotonicityMode,
    MonotonicityDirection,
    PSIInfo,
    PSIMode,
    PSILevel,
)

from stages.selection import (
    ClusteringStage,
    ClusterInfo,
    ClusteringInfo,
    ClusterSelectionMethod,
    StepwiseSelectionStage,
    StepwiseFeatureInfo,
    StepwiseInfo,
    EarlyStoppingInfo,
    FinalFilterStage,
    FinalFilterConfig,
    FeatureFilterInfo,
    FinalFilterInfo,
    InteractionDetectorStage,
    InteractionCandidate,
    InteractionInfo,
)

from stages.modeling import (
    ModelTrainerStage,
    FeatureCoefficient,
    ScorecardFeature,
    ModelInfo,
    ScorecardInfo,
    ConfidenceInterval,
    BootstrapResults,
)

from stages.visualization import (
    # Base
    BaseVisualization,
    VisualizationConfig,
    ColorScheme,
    # Performance
    ROCCurve,
    AUCTrend,
    KSPlot,
    # Distribution
    ScoreDistribution,
    HitsByBucket,
    ScoreDensityComparison,
    # Gains
    GainsChart,
    LiftChart,
    CaptureRateTable,
    # Rates
    ARBRCurve,
    ARBRTable,
    SwapAnalysis,
)

__all__ = [
    # Preprocessing
    "DataCleaner",
    "DataCleanerConfig",
    "TypeDetector",
    "TypeDetectorConfig",
    "FeatureType",
    "FeatureTypeInfo",
    "RejectInferenceStage",
    "RejectInferenceInfo",
    "RejectInferenceLog",
    # Transformation
    "WoEBinnerStage",
    "WoEBinnerConfig",
    "BinningInfo",
    "MonotonicityInfo",
    "MonotonicityMode",
    "MonotonicityDirection",
    "PSIInfo",
    "PSIMode",
    "PSILevel",
    # Selection
    "ClusteringStage",
    "ClusterInfo",
    "ClusteringInfo",
    "ClusterSelectionMethod",
    "StepwiseSelectionStage",
    "StepwiseFeatureInfo",
    "StepwiseInfo",
    "EarlyStoppingInfo",
    "FinalFilterStage",
    "FinalFilterConfig",
    "FeatureFilterInfo",
    "FinalFilterInfo",
    "InteractionDetectorStage",
    "InteractionCandidate",
    "InteractionInfo",
    # Modeling
    "ModelTrainerStage",
    "FeatureCoefficient",
    "ScorecardFeature",
    "ModelInfo",
    "ScorecardInfo",
    "ConfidenceInterval",
    "BootstrapResults",
    # Visualization
    "BaseVisualization",
    "VisualizationConfig",
    "ColorScheme",
    "ROCCurve",
    "AUCTrend",
    "KSPlot",
    "ScoreDistribution",
    "HitsByBucket",
    "ScoreDensityComparison",
    "GainsChart",
    "LiftChart",
    "CaptureRateTable",
    "ARBRCurve",
    "ARBRTable",
    "SwapAnalysis",
]
