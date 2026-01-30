"""
Pipeline stages module.

Contains all pipeline stage implementations organized by category:
- preprocessing: data cleaning, type detection, imputation
- transformation: WoE binning, postprocessing
- selection: clustering, stepwise selection, final filtering
- modeling: model training, scorecard building
- visualization: performance charts, distribution plots, AR/BR analysis
"""

from stages.preprocessing import (
    DataCleaner,
    DataCleanerConfig,
    TypeDetector,
    TypeDetectorConfig,
    FeatureType,
)

from stages.transformation import (
    WoEBinnerStage,
    WoEBinnerConfig,
    BinningInfo,
)

from stages.selection import (
    ClusteringStage,
    ClusterInfo,
    ClusteringInfo,
    ClusterSelectionMethod,
    StepwiseSelectionStage,
    StepwiseFeatureInfo,
    StepwiseInfo,
    FinalFilterStage,
    FinalFilterConfig,
    FeatureFilterInfo,
    FinalFilterInfo,
)

from stages.modeling import (
    ModelTrainerStage,
    FeatureCoefficient,
    ScorecardFeature,
    ModelInfo,
    ScorecardInfo,
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
    # Transformation
    "WoEBinnerStage",
    "WoEBinnerConfig",
    "BinningInfo",
    # Selection
    "ClusteringStage",
    "ClusterInfo",
    "ClusteringInfo",
    "ClusterSelectionMethod",
    "StepwiseSelectionStage",
    "StepwiseFeatureInfo",
    "StepwiseInfo",
    "FinalFilterStage",
    "FinalFilterConfig",
    "FeatureFilterInfo",
    "FinalFilterInfo",
    # Modeling
    "ModelTrainerStage",
    "FeatureCoefficient",
    "ScorecardFeature",
    "ModelInfo",
    "ScorecardInfo",
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
