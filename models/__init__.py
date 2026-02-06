"""
Model evaluation module for reasoning analysis.

Provides comprehensive evaluation utilities for:
- Time-based stability analysis (PSI, Gini trends)
- Segment-level performance (subpopulation analysis)
- Model comparison (challenger vs champion)

Example usage:
    from models import StabilityAnalyzer, SegmentAnalyzer, ModelComparator

    # Stability over time
    stability = StabilityAnalyzer(EvaluationConfig(time_column='month'))
    stability.evaluate(data, predictions)
    print(stability.get_summary())

    # Segment performance
    segments = SegmentAnalyzer(EvaluationConfig(segment_columns=['age_group']))
    segments.evaluate(data, predictions, segment_column='age_group')
    print(segments.get_summary())

    # Model comparison
    comparator = ModelComparator()
    result = comparator.compare_two(data, y_true, preds_old, preds_new)
    print(f"Winner: {result.winner}")
"""

from models.base import (
    BaseEvaluator,
    EvaluationConfig,
    PerformanceMetrics,
    MetricType,
    calculate_psi,
    calculate_csi,
)

from models.stability import (
    StabilityAnalyzer,
    StabilityReport,
    PeriodMetrics,
    ScoreDistributionMonitor,
)

from models.segments import (
    SegmentAnalyzer,
    SegmentReport,
    SegmentMetrics,
    CrossSegmentAnalyzer,
)

from models.comparison import (
    ModelComparator,
    ModelComparisonResult,
    MultiModelComparison,
)

__all__ = [
    # Base
    "BaseEvaluator",
    "EvaluationConfig",
    "PerformanceMetrics",
    "MetricType",
    "calculate_psi",
    "calculate_csi",
    # Stability
    "StabilityAnalyzer",
    "StabilityReport",
    "PeriodMetrics",
    "ScoreDistributionMonitor",
    # Segments
    "SegmentAnalyzer",
    "SegmentReport",
    "SegmentMetrics",
    "CrossSegmentAnalyzer",
    # Comparison
    "ModelComparator",
    "ModelComparisonResult",
    "MultiModelComparison",
]
