"""
Segment-level performance analysis for scorecard models.

Evaluates model performance across different subpopulations:
- Demographic segments (age, income, etc.)
- Product segments
- Channel segments
- Risk tiers
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

from models.base import (
    BaseEvaluator,
    EvaluationConfig,
    PerformanceMetrics,
)


logger = logging.getLogger(__name__)


@dataclass
class SegmentMetrics:
    """Performance metrics for a single segment."""
    segment_name: str
    segment_value: Any
    performance: PerformanceMetrics
    # comparison to overall
    gini_vs_overall: float  # difference from overall Gini
    bad_rate_vs_overall: float  # ratio to overall bad rate


@dataclass
class SegmentReport:
    """Report for segment-level analysis."""
    segment_column: str
    n_segments: int
    overall_performance: PerformanceMetrics
    # segment details
    segments: List[SegmentMetrics] = field(default_factory=list)
    # insights
    best_segment: Optional[str] = None
    worst_segment: Optional[str] = None
    high_risk_segments: List[str] = field(default_factory=list)
    low_volume_segments: List[str] = field(default_factory=list)


class SegmentAnalyzer(BaseEvaluator):
    """
    Analyzer for segment-level model performance.

    Evaluates how the model performs across different subpopulations,
    identifying segments where the model may be:
    - Underperforming
    - Over/under-predicting risk
    - Lacking sufficient data

    Example usage:
        analyzer = SegmentAnalyzer(config)
        results = analyzer.evaluate(data, predictions, segment_column='age_group')
        summary = analyzer.get_summary()
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize SegmentAnalyzer.

        Args:
            config: EvaluationConfig with segment_columns specified
        """
        super().__init__(config)
        self._segment_data: Dict[str, List[SegmentMetrics]] = {}
        self._reports: Dict[str, SegmentReport] = {}

    def evaluate(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        segment_column: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model performance by segment.

        Args:
            data: DataFrame with segment columns and target
            predictions: Predicted probabilities/scores
            segment_column: Column to segment by. If None, uses first
                          column in config.segment_columns
            **kwargs: Additional arguments

        Returns:
            Dictionary with segment analysis results
        """
        target_col = self.config.target_column

        # determine segment column
        if segment_column is None:
            if self.config.segment_columns:
                segment_column = self.config.segment_columns[0]
            else:
                raise ValueError("No segment_column specified")

        if segment_column not in data.columns:
            raise ValueError(f"Segment column '{segment_column}' not found in data")

        # add predictions to data
        df = data.copy()
        df['_pred'] = predictions

        # calculate overall metrics
        overall_perf = PerformanceMetrics.calculate(
            df[target_col].values,
            predictions
        )

        # get unique segment values
        segment_values = df[segment_column].unique()

        # calculate metrics for each segment
        segment_metrics = []

        for seg_val in segment_values:
            seg_mask = df[segment_column] == seg_val
            seg_df = df[seg_mask]

            if len(seg_df) < self.config.min_samples:
                logger.warning(
                    f"Segment {segment_column}={seg_val} has only "
                    f"{len(seg_df)} samples, marking as low volume"
                )

            y_true = seg_df[target_col].values
            y_score = seg_df['_pred'].values

            # handle edge case of no variance in target
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                perf = PerformanceMetrics(
                    auc=0.5, gini=0.0, ks=0.0,
                    n_samples=len(y_true),
                    n_bads=int(y_true.sum()),
                    bad_rate=y_true.mean()
                )
            else:
                perf = PerformanceMetrics.calculate(y_true, y_score)

            # compare to overall
            gini_diff = perf.gini - overall_perf.gini
            br_ratio = perf.bad_rate / overall_perf.bad_rate if overall_perf.bad_rate > 0 else 1.0

            segment_metrics.append(SegmentMetrics(
                segment_name=segment_column,
                segment_value=seg_val,
                performance=perf,
                gini_vs_overall=gini_diff,
                bad_rate_vs_overall=br_ratio
            ))

        # sort by Gini for best/worst identification
        sorted_by_gini = sorted(segment_metrics, key=lambda x: x.performance.gini, reverse=True)

        # identify notable segments
        best_segment = sorted_by_gini[0].segment_value if sorted_by_gini else None
        worst_segment = sorted_by_gini[-1].segment_value if sorted_by_gini else None

        high_risk = [
            str(sm.segment_value) for sm in segment_metrics
            if sm.bad_rate_vs_overall > 1.5
        ]
        low_volume = [
            str(sm.segment_value) for sm in segment_metrics
            if sm.performance.n_samples < self.config.min_samples
        ]

        # build report
        report = SegmentReport(
            segment_column=segment_column,
            n_segments=len(segment_metrics),
            overall_performance=overall_perf,
            segments=segment_metrics,
            best_segment=str(best_segment) if best_segment is not None else None,
            worst_segment=str(worst_segment) if worst_segment is not None else None,
            high_risk_segments=high_risk,
            low_volume_segments=low_volume
        )

        self._segment_data[segment_column] = segment_metrics
        self._reports[segment_column] = report

        self._results = {
            'segment_column': segment_column,
            'n_segments': len(segment_metrics),
            'overall_gini': overall_perf.gini,
            'segments': {
                str(sm.segment_value): sm.performance.to_dict()
                for sm in segment_metrics
            },
            'best_segment': best_segment,
            'worst_segment': worst_segment,
            'high_risk_segments': high_risk,
            'low_volume_segments': low_volume
        }

        return self._results

    def evaluate_multiple(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        segment_columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple segment columns at once.

        Args:
            data: DataFrame with segment columns and target
            predictions: Predicted probabilities/scores
            segment_columns: List of columns to analyze.
                           If None, uses config.segment_columns

        Returns:
            Dictionary with results for each segment column
        """
        if segment_columns is None:
            segment_columns = self.config.segment_columns

        results = {}
        for col in segment_columns:
            if col in data.columns:
                results[col] = self.evaluate(data, predictions, segment_column=col)
            else:
                logger.warning(f"Segment column '{col}' not found in data, skipping")

        return results

    def get_summary(self, segment_column: Optional[str] = None) -> pd.DataFrame:
        """
        Get segment analysis summary as DataFrame.

        Args:
            segment_column: Which segment to summarize. If None, uses last evaluated.

        Returns:
            Summary table with metrics by segment
        """
        if segment_column is None:
            if self._segment_data:
                segment_column = list(self._segment_data.keys())[-1]
            else:
                return pd.DataFrame()

        if segment_column not in self._segment_data:
            return pd.DataFrame()

        metrics = self._segment_data[segment_column]

        rows = []
        for sm in metrics:
            rows.append({
                'segment': sm.segment_value,
                'n_samples': sm.performance.n_samples,
                'n_bads': sm.performance.n_bads,
                'bad_rate': round(sm.performance.bad_rate, 4),
                'auc': round(sm.performance.auc, 4),
                'gini': round(sm.performance.gini, 4),
                'ks': round(sm.performance.ks, 4),
                'gini_vs_overall': round(sm.gini_vs_overall, 4),
                'bad_rate_ratio': round(sm.bad_rate_vs_overall, 2)
            })

        df = pd.DataFrame(rows)
        return df.sort_values('gini', ascending=False)

    def get_report(self, segment_column: Optional[str] = None) -> Optional[SegmentReport]:
        """Get the full segment report."""
        if segment_column is None and self._reports:
            segment_column = list(self._reports.keys())[-1]

        return self._reports.get(segment_column)

    def get_underperforming_segments(
        self,
        threshold: float = 0.05,
        segment_column: Optional[str] = None
    ) -> List[Tuple[Any, float]]:
        """
        Get segments where Gini is significantly below overall.

        Args:
            threshold: Minimum Gini difference to flag (e.g., 0.05 = 5 points)
            segment_column: Which segment to check

        Returns:
            List of (segment_value, gini_difference) tuples
        """
        if segment_column is None and self._segment_data:
            segment_column = list(self._segment_data.keys())[-1]

        if segment_column not in self._segment_data:
            return []

        underperforming = [
            (sm.segment_value, sm.gini_vs_overall)
            for sm in self._segment_data[segment_column]
            if sm.gini_vs_overall < -threshold
        ]

        return sorted(underperforming, key=lambda x: x[1])

    def to_visualization_data(self, segment_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare data for visualization components.

        Returns:
            Dictionary ready for bar charts or heatmaps
        """
        if segment_column is None and self._segment_data:
            segment_column = list(self._segment_data.keys())[-1]

        if segment_column not in self._segment_data:
            return {}

        metrics = self._segment_data[segment_column]

        return {
            'segments': [str(sm.segment_value) for sm in metrics],
            'gini': [sm.performance.gini for sm in metrics],
            'auc': [sm.performance.auc for sm in metrics],
            'ks': [sm.performance.ks for sm in metrics],
            'bad_rate': [sm.performance.bad_rate for sm in metrics],
            'n_samples': [sm.performance.n_samples for sm in metrics],
            'gini_vs_overall': [sm.gini_vs_overall for sm in metrics]
        }


class CrossSegmentAnalyzer:
    """
    Analyzer for cross-tabulated segment performance.

    Evaluates performance across combinations of segments
    (e.g., age_group x income_level).
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize CrossSegmentAnalyzer.

        Args:
            config: EvaluationConfig instance
        """
        self.config = config or EvaluationConfig()
        self._results: Optional[pd.DataFrame] = None

    def evaluate(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        segment_columns: List[str]
    ) -> pd.DataFrame:
        """
        Evaluate performance across segment combinations.

        Args:
            data: DataFrame with segment columns and target
            predictions: Predicted probabilities/scores
            segment_columns: List of columns to cross-tabulate (max 2-3 recommended)

        Returns:
            DataFrame with metrics for each segment combination
        """
        target_col = self.config.target_column

        df = data.copy()
        df['_pred'] = predictions

        # group by segment combinations
        groups = df.groupby(segment_columns)

        results = []
        for group_key, group_df in groups:
            if len(group_df) < self.config.min_samples:
                continue

            y_true = group_df[target_col].values
            y_score = group_df['_pred'].values

            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                gini = 0.0
                auc = 0.5
            else:
                auc = roc_auc_score(y_true, y_score)
                gini = 2 * auc - 1

            row = {col: val for col, val in zip(segment_columns, group_key)} if isinstance(group_key, tuple) else {segment_columns[0]: group_key}
            row.update({
                'n_samples': len(group_df),
                'n_bads': int(y_true.sum()),
                'bad_rate': y_true.mean(),
                'gini': gini,
                'auc': auc
            })
            results.append(row)

        self._results = pd.DataFrame(results)
        return self._results

    def get_heatmap_data(
        self,
        row_segment: str,
        col_segment: str,
        metric: str = 'gini'
    ) -> pd.DataFrame:
        """
        Pivot results into heatmap format.

        Args:
            row_segment: Column to use for rows
            col_segment: Column to use for columns
            metric: Which metric to show ('gini', 'auc', 'bad_rate', 'n_samples')

        Returns:
            Pivoted DataFrame suitable for heatmap
        """
        if self._results is None:
            raise RuntimeError("Must call evaluate() first")

        return self._results.pivot_table(
            index=row_segment,
            columns=col_segment,
            values=metric,
            aggfunc='mean'
        )


# import for cross-segment (was missing)
from sklearn.metrics import roc_auc_score
