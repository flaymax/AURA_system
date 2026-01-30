"""
Visualization module for scorecard model analysis.

Provides comprehensive visualizations for model evaluation:
- Performance: ROC curves, AUC trends, KS plots
- Distribution: Score histograms, density comparisons, hits by bucket
- Gains: Cumulative gains charts, lift charts, capture rate tables
- Rates: AR/BR curves and tables, swap analysis

Two visualization backends available:
- Matplotlib (default): Static images, PDF reports
- Plotly (_pl suffix): Interactive HTML, dashboards

All visualizations support:
- Export to PNG, SVG, PDF, HTML, JSON
- Data extraction for external processing
"""

# =============================================================================
# Matplotlib-based visualizations (static)
# =============================================================================

from stages.visualization.base import (
    BaseVisualization,
    VisualizationConfig,
    ColorScheme,
    OutputFormat,
    calculate_buckets,
    calculate_cumulative_stats,
)

from stages.visualization.performance import (
    ROCCurve,
    ROCData,
    AUCTrend,
    KSPlot,
)

from stages.visualization.distribution import (
    ScoreDistribution,
    HitsByBucket,
    BucketStats,
    ScoreDensityComparison,
)

from stages.visualization.gains import (
    GainsChart,
    LiftChart,
    CaptureRateTable,
)

from stages.visualization.rates import (
    ARBRCurve,
    ARBRTable,
    CutoffMetrics,
    SwapAnalysis,
)

# =============================================================================
# Plotly-based visualizations (interactive)
# =============================================================================

from stages.visualization.base_pl import (
    BasePlotlyVisualization,
    PlotlyConfig,
    PlotlyColorScheme,
)

from stages.visualization.performance_pl import (
    ROCCurvePl,
    ROCDataPl,
    AUCTrendPl,
    KSPlotPl,
)

from stages.visualization.distribution_pl import (
    ScoreDistributionPl,
    HitsByBucketPl,
    ScoreDensityComparisonPl,
)

from stages.visualization.gains_pl import (
    GainsChartPl,
    LiftChartPl,
    CaptureRateTablePl,
)

from stages.visualization.rates_pl import (
    ARBRCurvePl,
    ARBRTablePl,
    CutoffMetricsPl,
    SwapAnalysisPl,
)

__all__ = [
    # ==========================================================================
    # Matplotlib (static)
    # ==========================================================================
    # Base
    "BaseVisualization",
    "VisualizationConfig",
    "ColorScheme",
    "OutputFormat",
    "calculate_buckets",
    "calculate_cumulative_stats",
    # Performance
    "ROCCurve",
    "ROCData",
    "AUCTrend",
    "KSPlot",
    # Distribution
    "ScoreDistribution",
    "HitsByBucket",
    "BucketStats",
    "ScoreDensityComparison",
    # Gains
    "GainsChart",
    "LiftChart",
    "CaptureRateTable",
    # Rates
    "ARBRCurve",
    "ARBRTable",
    "CutoffMetrics",
    "SwapAnalysis",

    # ==========================================================================
    # Plotly (interactive)
    # ==========================================================================
    # Base
    "BasePlotlyVisualization",
    "PlotlyConfig",
    "PlotlyColorScheme",
    # Performance
    "ROCCurvePl",
    "ROCDataPl",
    "AUCTrendPl",
    "KSPlotPl",
    # Distribution
    "ScoreDistributionPl",
    "HitsByBucketPl",
    "ScoreDensityComparisonPl",
    # Gains
    "GainsChartPl",
    "LiftChartPl",
    "CaptureRateTablePl",
    # Rates
    "ARBRCurvePl",
    "ARBRTablePl",
    "CutoffMetricsPl",
    "SwapAnalysisPl",
]
