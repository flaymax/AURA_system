"""
Modeling stages for the reasoning pipeline.

These stages handle model training and points-based scoring:
- ModelTrainerStage: Logistic regression training and points conversion
- Bootstrap confidence intervals for model metrics
"""

from stages.modeling.trainer import (
    ModelTrainerStage,
    FeatureCoefficient,
    ScorecardFeature,
    ModelInfo,
    ScorecardInfo,
    ConfidenceInterval,
    BootstrapResults,
)

__all__ = [
    "ModelTrainerStage",
    "FeatureCoefficient",
    "ScorecardFeature",
    "ModelInfo",
    "ScorecardInfo",
    "ConfidenceInterval",
    "BootstrapResults",
]
