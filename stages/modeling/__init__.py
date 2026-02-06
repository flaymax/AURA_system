"""
Modeling stages for the scorecard pipeline.

These stages handle model training and scorecard building:
- ModelTrainerStage: Logistic regression training and scorecard conversion
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
