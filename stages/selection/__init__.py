"""
Selection stages for feature reduction in the reasoning pipeline.

These stages handle different aspects of feature selection:
- ClusteringStage: Reduces correlated features using hierarchical clustering
- StepwiseSelectionStage: Forward-backward stepwise logistic regression
- FinalFilterStage: P-value and VIF validation before model training
- InteractionDetectorStage: Detects meaningful feature interactions
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
    EarlyStoppingInfo,
)

from stages.selection.final_filter import (
    FinalFilterStage,
    FinalFilterConfig,
    FeatureFilterInfo,
    FinalFilterInfo,
)

from stages.selection.interactions import (
    InteractionDetectorStage,
    InteractionCandidate,
    InteractionInfo,
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
    "EarlyStoppingInfo",
    # Final Filter
    "FinalFilterStage",
    "FinalFilterConfig",
    "FeatureFilterInfo",
    "FinalFilterInfo",
    # Interactions
    "InteractionDetectorStage",
    "InteractionCandidate",
    "InteractionInfo",
]
