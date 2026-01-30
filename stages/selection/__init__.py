"""
Selection stages for feature reduction in the scorecard pipeline.

These stages handle different aspects of feature selection:
- ClusteringStage: Reduces correlated features using hierarchical clustering
- StepwiseSelectionStage: Forward-backward stepwise logistic regression
- FinalFilterStage: P-value and VIF validation before model training
"""

from stages.selection.clustering import (
    ClusteringStage,
    ClusterInfo,
    ClusteringInfo,
    ClusterSelectionMethod,
)

from stages.selection.stepwise import (
    StepwiseSelectionStage,
    StepwiseFeatureInfo,
    StepwiseInfo,
)

from stages.selection.final_filter import (
    FinalFilterStage,
    FinalFilterConfig,
    FeatureFilterInfo,
    FinalFilterInfo,
)

__all__ = [
    # Clustering
    "ClusteringStage",
    "ClusterInfo",
    "ClusteringInfo",
    "ClusterSelectionMethod",
    # Stepwise
    "StepwiseSelectionStage",
    "StepwiseFeatureInfo",
    "StepwiseInfo",
    # Final Filter
    "FinalFilterStage",
    "FinalFilterConfig",
    "FeatureFilterInfo",
    "FinalFilterInfo",
]
