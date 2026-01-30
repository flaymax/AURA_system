"""
Time-based stability analysis for scorecard models.

Monitors model performance and score distribution over time:
- Gini/AUC trends by period
- PSI (Population Stability Index) tracking
- Performance degradation detection
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from models.base import (
    BaseEvaluator,
    EvaluationConfig,
    PerformanceMetrics,
    calculate_psi,
    calculate_csi,
)


logger = logging.getLogger(__name__)


@dataclass
class PeriodMetrics:
    """Performance metrics for a single time period."""
    period: str
    performance: PerformanceMetrics
    psi: Optional[float] = None
    psi_status: Optional[str] = None  # 'stable', 'moderate', 'significant'


@dataclass
class StabilityReport:
    """Complete stability analysis report."""
    # overall summary
    baseline_period: str
    n_periods: int
    overall_psi: float
    overall_status: str
    # trend analysis
    gini_trend: str  # 'stable', 'improving', 'degrading'
    gini_volatility: float  # std dev of Gini across periods
    # period-by-period metrics
    period_metrics: List[PeriodMetrics] = field(default_factory=list)
    # alerts
    alerts: List[str] = field(default_factory=list)


class StabilityAnalyzer(BaseEvaluator):
    """
    Analyzer for time-based model stability.

    Tracks model performance across time periods to detect:
    - Performance degradation
    - Score distribution shifts (PSI)
    - Unusual volatility

    Example usage:
        analyzer = StabilityAnalyzer(config)
        results = analyzer.evaluate(data, predictions)
        summary = analyzer.get_summary()
        analyzer.plot_trends()  # if visualization needed
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize StabilityAnalyzer.

        Args:
            config: EvaluationConfig with time_column specified
        """
        super().__init__(config)
        self._period_data: List[PeriodMetrics] = []
        self._baseline_scores: Optional[np.ndarray] = None
        self._report: Optional[StabilityReport] = None

    def evaluate(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        baseline_period: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model stability across time periods.

        Args:
            data: DataFrame with time column and target
            predictions: Predicted probabilities/scores
            baseline_period: Period to use as baseline for PSI.
                           If None, uses earliest period.
            **kwargs: Additional arguments

        Returns:
            Dictionary with stability analysis results
        """
        time_col = self.config.time_column
        target_col = self.config.target_column

        if time_col is None:
            raise ValueError("time_column must be specified in config for stability analysis")

        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")

        # add predictions to data
        df = data.copy()
        df['_pred'] = predictions

        # get sorted unique periods
        periods = sorted(df[time_col].unique())

        if len(periods) < 2:
            logger.warning("Less than 2 time periods found, stability analysis limited")

        # determine baseline
        if baseline_period is None:
            baseline_period = periods[0]

        # get baseline scores for PSI
        baseline_mask = df[time_col] == baseline_period
        self._baseline_scores = df.loc[baseline_mask, '_pred'].values

        # calculate metrics for each period
        self._period_data = []
        alerts = []

        for period in periods:
            period_mask = df[time_col] == period
            period_df = df[period_mask]

            if len(period_df) < self.config.min_samples:
                logger.warning(f"Period {period} has only {len(period_df)} samples, skipping")
                continue

            y_true = period_df[target_col].values
            y_score = period_df['_pred'].values

            # calculate performance metrics
            perf = PerformanceMetrics.calculate(y_true, y_score)

            # calculate PSI vs baseline
            if period != baseline_period and self._baseline_scores is not None:
                psi_value, _ = calculate_psi(self._baseline_scores, y_score)
                psi_status = self._get_psi_status(psi_value)

                if psi_status == 'significant':
                    alerts.append(f"Period {period}: Significant score shift (PSI={psi_value:.3f})")
            else:
                psi_value = 0.0
                psi_status = 'baseline'

            self._period_data.append(PeriodMetrics(
                period=str(period),
                performance=perf,
                psi=psi_value,
                psi_status=psi_status
            ))

        # analyze trends
        gini_values = [pm.performance.gini for pm in self._period_data]
        gini_trend = self._analyze_trend(gini_values)
        gini_volatility = np.std(gini_values) if len(gini_values) > 1 else 0.0

        # check for performance degradation
        if len(gini_values) >= 3:
            recent_avg = np.mean(gini_values[-3:])
            baseline_gini = gini_values[0]
            if recent_avg < baseline_gini * 0.9:
                alerts.append(
                    f"Performance degradation: Recent Gini ({recent_avg:.3f}) "
                    f"is {((baseline_gini - recent_avg) / baseline_gini * 100):.1f}% below baseline"
                )

        # overall PSI (comparing first half to second half)
        if len(self._period_data) >= 4:
            mid = len(self._period_data) // 2
            first_half_scores = np.concatenate([
                df[df[time_col] == pm.period]['_pred'].values
                for pm in self._period_data[:mid]
            ])
            second_half_scores = np.concatenate([
                df[df[time_col] == pm.period]['_pred'].values
                for pm in self._period_data[mid:]
            ])
            overall_psi, _ = calculate_psi(first_half_scores, second_half_scores)
        else:
            overall_psi = 0.0

        # build report
        self._report = StabilityReport(
            baseline_period=str(baseline_period),
            n_periods=len(self._period_data),
            overall_psi=overall_psi,
            overall_status=self._get_psi_status(overall_psi),
            gini_trend=gini_trend,
            gini_volatility=gini_volatility,
            period_metrics=self._period_data,
            alerts=alerts
        )

        self._results = {
            'report': self._report,
            'periods': [pm.period for pm in self._period_data],
            'gini_by_period': {pm.period: pm.performance.gini for pm in self._period_data},
            'auc_by_period': {pm.period: pm.performance.auc for pm in self._period_data},
            'psi_by_period': {pm.period: pm.psi for pm in self._period_data},
            'alerts': alerts
        }

        return self._results

    def _get_psi_status(self, psi: float) -> str:
        """Classify PSI value into status category."""
        if psi < 0.1:
            return 'stable'
        elif psi < 0.25:
            return 'moderate'
        else:
            return 'significant'

    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend direction in a series of values."""
        if len(values) < 3:
            return 'insufficient_data'

        # simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # normalize by mean to get relative change
        mean_val = np.mean(values)
        if mean_val == 0:
            return 'stable'

        relative_slope = slope / mean_val

        if relative_slope > 0.02:
            return 'improving'
        elif relative_slope < -0.02:
            return 'degrading'
        else:
            return 'stable'

    def get_summary(self) -> pd.DataFrame:
        """
        Get evaluation summary as DataFrame.

        Returns:
            Summary table with metrics by period
        """
        if not self._period_data:
            return pd.DataFrame()

        rows = []
        for pm in self._period_data:
            rows.append({
                'period': pm.period,
                'n_samples': pm.performance.n_samples,
                'n_bads': pm.performance.n_bads,
                'bad_rate': round(pm.performance.bad_rate, 4),
                'auc': round(pm.performance.auc, 4),
                'gini': round(pm.performance.gini, 4),
                'ks': round(pm.performance.ks, 4),
                'psi': round(pm.psi, 4) if pm.psi else None,
                'psi_status': pm.psi_status
            })

        return pd.DataFrame(rows)

    def get_report(self) -> Optional[StabilityReport]:
        """Get the full stability report."""
        return self._report

    def get_alerts(self) -> List[str]:
        """Get list of stability alerts."""
        return self._report.alerts if self._report else []

    def to_visualization_data(self) -> Dict[str, Any]:
        """
        Prepare data for visualization components.

        Returns:
            Dictionary ready for AUCTrend or similar visualizations
        """
        if not self._period_data:
            return {}

        return {
            'periods': [pm.period for pm in self._period_data],
            'gini': [pm.performance.gini for pm in self._period_data],
            'auc': [pm.performance.auc for pm in self._period_data],
            'ks': [pm.performance.ks for pm in self._period_data],
            'psi': [pm.psi for pm in self._period_data],
            'n_samples': [pm.performance.n_samples for pm in self._period_data],
            'bad_rate': [pm.performance.bad_rate for pm in self._period_data]
        }


class ScoreDistributionMonitor:
    """
    Monitor score distribution changes over time.

    Tracks PSI at configurable intervals and maintains
    history for trend analysis.
    """

    def __init__(
        self,
        baseline_scores: np.ndarray,
        n_bins: int = 10,
        psi_threshold_warning: float = 0.1,
        psi_threshold_alert: float = 0.25
    ):
        """
        Initialize monitor with baseline distribution.

        Args:
            baseline_scores: Score distribution to use as reference
            n_bins: Number of bins for PSI calculation
            psi_threshold_warning: PSI threshold for warning status
            psi_threshold_alert: PSI threshold for alert status
        """
        self.baseline_scores = baseline_scores
        self.n_bins = n_bins
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_alert = psi_threshold_alert

        # history tracking
        self._history: List[Dict[str, Any]] = []

    def check(
        self,
        current_scores: np.ndarray,
        period_label: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check current scores against baseline.

        Args:
            current_scores: Current score distribution
            period_label: Optional label for this check

        Returns:
            Dictionary with PSI results and status
        """
        psi_value, breakdown = calculate_psi(
            self.baseline_scores,
            current_scores,
            n_bins=self.n_bins
        )

        if psi_value >= self.psi_threshold_alert:
            status = 'alert'
        elif psi_value >= self.psi_threshold_warning:
            status = 'warning'
        else:
            status = 'stable'

        result = {
            'period': period_label,
            'psi': psi_value,
            'status': status,
            'n_baseline': len(self.baseline_scores),
            'n_current': len(current_scores),
            'breakdown': breakdown
        }

        self._history.append(result)
        return result

    def get_history(self) -> pd.DataFrame:
        """Get PSI check history as DataFrame."""
        if not self._history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'period': h['period'],
                'psi': h['psi'],
                'status': h['status'],
                'n_samples': h['n_current']
            }
            for h in self._history
        ])

    def update_baseline(self, new_baseline: np.ndarray):
        """
        Update baseline distribution.

        Use this when retraining the model or establishing
        a new reference point.

        Args:
            new_baseline: New baseline score distribution
        """
        self.baseline_scores = new_baseline
        logger.info(f"Baseline updated with {len(new_baseline)} samples")
