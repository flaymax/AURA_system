"""
Pipeline module with core binning and clustering implementations.

Contains the original/legacy implementations that are wrapped by stage classes:
- Binner: WoE binning using decision trees
- Binning: Individual feature binning with metrics
- ClusterAnalysis: Hierarchical clustering for feature selection
"""

from pipeline.binner import Binner, BinnerType
from pipeline.fit_transform import Binning, BinningSettings
from pipeline.clustering import ClusterAnalysis

__all__ = [
    "Binner",
    "BinnerType",
    "Binning",
    "BinningSettings",
    "ClusterAnalysis",
]
