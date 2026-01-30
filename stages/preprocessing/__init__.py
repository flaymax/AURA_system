"""
Preprocessing stages for the scorecard pipeline.

These stages handle initial data cleaning and preparation:
- DataCleaner: removes nulls, constants, non-numeric features
- TypeDetector: identifies feature types (binary, categorical, discrete, continuous)
- MissingValueHandler: imputation strategies (TODO)
"""

from stages.preprocessing.cleaner import DataCleaner, DataCleanerConfig
from stages.preprocessing.type_detector import (
    TypeDetector,
    TypeDetectorConfig,
    FeatureType,
    FeatureTypeInfo,
)

__all__ = [
    "DataCleaner",
    "DataCleanerConfig",
    "TypeDetector",
    "TypeDetectorConfig",
    "FeatureType",
    "FeatureTypeInfo",
]
