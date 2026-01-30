"""
Gains and lift chart visualizations.

Includes:
- Cumulative gains chart
- Lift chart
- Capture rate analysis
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stages.visualization.base import (
    BaseVisualization,
    VisualizationConfig,
    calculate_cumulative_stats,
    format_percentage,
)


class GainsChart(BaseVisualization):
    """
    Cumulative gains chart visualization.

    Shows what percentage of total bads (defaults) are captured
    when selecting top X% of population by score. Ideal model
    would capture 100% of bads immediately.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._gains_data: Dict[str, pd.DataFrame] = {}

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_buckets: int = 10,
        dataset_name: str = "Model"
    ) -> 'GainsChart':
        """
        Calculate cumulative gains.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            n_buckets: Number of buckets for calculation
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        # create DataFrame
        df = pd.DataFrame({
            'score': scores,
            'label': labels
        })

        # use utility function
        bucket_stats = calculate_cumulative_stats(
            df, 'score', 'label', n_buckets
        )

        self._gains_data[dataset_name] = bucket_stats
        return self

    def plot(self, show_perfect: bool = True, **kwargs) -> 'GainsChart':
        """
        Generate cumulative gains chart.

        Args:
            show_perfect: If True, show perfect model line
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)
        colors = self.config.colors

        for idx, (name, data) in enumerate(self._gains_data.items()):
            color = colors.palette[idx % len(colors.palette)]

            # plot model gains
            # prepend (0, 0) point
            x = np.concatenate([[0], data['cum_count_pct'].values])
            y = np.concatenate([[0], data['cum_bad_pct'].values])

            ax.plot(x, y, 'o-', color=color, lw=2, markersize=6, label=name)

        # random model (diagonal)
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random')

        if show_perfect:
            # perfect model - captures all bads first
            # find the proportion of bads in population
            first_data = list(self._gains_data.values())[0]
            total_bads = first_data['bads'].sum()
            total_count = first_data['count'].sum()
            bad_proportion = total_bads / total_count

            # perfect model captures 100% of bads by the time we've seen bad_proportion of population
            ax.plot(
                [0, bad_proportion, 1],
                [0, 1, 1],
                '--',
                color=colors.success,
                lw=1.5,
                alpha=0.7,
                label='Perfect Model'
            )

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Proportion of Population', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Proportion of Bads Captured', fontsize=self.config.label_fontsize)
        ax.set_title('Cumulative Gains Chart', fontsize=self.config.title_fontsize)
        ax.legend(loc='lower right', fontsize=self.config.legend_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get gains data for export."""
        result = {}
        for name, data in self._gains_data.items():
            points = []
            for _, row in data.iterrows():
                points.append({
                    'bucket': int(row['bucket']),
                    'population_pct': round(row['cum_count_pct'], 4),
                    'bads_captured_pct': round(row['cum_bad_pct'], 4),
                    'goods_captured_pct': round(row['cum_good_pct'], 4)
                })
            result[name] = {
                'points': points,
                'area_under_curve': self._calculate_auc(data)
            }
        return result

    def _calculate_auc(self, data: pd.DataFrame) -> float:
        """Calculate area under gains curve using trapezoidal rule."""
        x = np.concatenate([[0], data['cum_count_pct'].values])
        y = np.concatenate([[0], data['cum_bad_pct'].values])
        return float(np.trapz(y, x))

    def get_capture_rate(self, population_pct: float, dataset_name: str = "Model") -> float:
        """
        Get bads capture rate at given population percentage.

        Args:
            population_pct: Population proportion (0 to 1)
            dataset_name: Name of dataset

        Returns:
            Proportion of bads captured
        """
        if dataset_name not in self._gains_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._gains_data[dataset_name]
        # interpolate
        return float(np.interp(
            population_pct,
            np.concatenate([[0], data['cum_count_pct'].values]),
            np.concatenate([[0], data['cum_bad_pct'].values])
        ))


class LiftChart(BaseVisualization):
    """
    Lift chart visualization.

    Shows how much better the model is at identifying bads
    compared to random selection. Lift = (% bads captured) / (% population).
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._lift_data: Dict[str, pd.DataFrame] = {}

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_buckets: int = 10,
        dataset_name: str = "Model"
    ) -> 'LiftChart':
        """
        Calculate lift by bucket.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            n_buckets: Number of buckets
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        df = pd.DataFrame({
            'score': scores,
            'label': labels
        })

        bucket_stats = calculate_cumulative_stats(df, 'score', 'label', n_buckets)
        self._lift_data[dataset_name] = bucket_stats
        return self

    def plot(self, cumulative: bool = True, **kwargs) -> 'LiftChart':
        """
        Generate lift chart.

        Args:
            cumulative: If True, show cumulative lift. If False, show per-bucket lift.
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)
        colors = self.config.colors

        for idx, (name, data) in enumerate(self._lift_data.items()):
            color = colors.palette[idx % len(colors.palette)]

            if cumulative:
                # cumulative lift
                lift_values = data['lift'].values
                x = data['cum_count_pct'].values
                ax.plot(x, lift_values, 'o-', color=color, lw=2, markersize=6, label=name)
            else:
                # per-bucket lift (bad rate in bucket / overall bad rate)
                overall_bad_rate = data['bads'].sum() / data['count'].sum()
                bucket_lift = data['bad_rate'] / overall_bad_rate
                x = range(1, len(bucket_lift) + 1)
                ax.bar(x, bucket_lift, color=color, alpha=0.8, label=name)

        # reference line at lift = 1 (random)
        if cumulative:
            ax.axhline(y=1, color='gray', linestyle='--', lw=1.5, alpha=0.7, label='No Lift')
        else:
            ax.axhline(y=1, color='gray', linestyle='--', lw=1.5, alpha=0.7)

        if cumulative:
            ax.set_xlabel('Proportion of Population', fontsize=self.config.label_fontsize)
        else:
            ax.set_xlabel('Bucket (1 = Highest Risk)', fontsize=self.config.label_fontsize)

        ax.set_ylabel('Lift', fontsize=self.config.label_fontsize)
        ax.set_title('Lift Chart' + (' (Cumulative)' if cumulative else ' (Per Bucket)'),
                     fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=self.config.legend_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get lift data for export."""
        result = {}
        for name, data in self._lift_data.items():
            overall_bad_rate = data['bads'].sum() / data['count'].sum()

            buckets = []
            for _, row in data.iterrows():
                bucket_lift = row['bad_rate'] / overall_bad_rate
                buckets.append({
                    'bucket': int(row['bucket']),
                    'population_pct': round(row['cum_count_pct'], 4),
                    'cumulative_lift': round(row['lift'], 4),
                    'bucket_lift': round(bucket_lift, 4),
                    'bad_rate': round(row['bad_rate'], 4)
                })

            result[name] = {
                'buckets': buckets,
                'overall_bad_rate': round(overall_bad_rate, 4)
            }

        return result

    def get_lift_at_pct(self, population_pct: float, dataset_name: str = "Model") -> float:
        """
        Get cumulative lift at given population percentage.

        Args:
            population_pct: Population proportion (0 to 1)
            dataset_name: Name of dataset

        Returns:
            Lift value
        """
        if dataset_name not in self._lift_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._lift_data[dataset_name]
        return float(np.interp(
            population_pct,
            data['cum_count_pct'].values,
            data['lift'].values
        ))


class CaptureRateTable(BaseVisualization):
    """
    Capture rate analysis table.

    Shows key metrics at various population cutoffs:
    - % of bads captured
    - % of goods captured
    - Bad rate at cutoff
    - Lift
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._table_data: Optional[pd.DataFrame] = None
        self._raw_data: Optional[pd.DataFrame] = None

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        cutoff_percentiles: Optional[List[int]] = None
    ) -> 'CaptureRateTable':
        """
        Calculate capture rates at various cutoffs.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            cutoff_percentiles: List of percentiles to evaluate (default: 10, 20, ..., 90)

        Returns:
            self for method chaining
        """
        if cutoff_percentiles is None:
            cutoff_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        df = pd.DataFrame({
            'score': scores,
            'label': labels
        })

        # calculate stats at many buckets for interpolation
        bucket_stats = calculate_cumulative_stats(df, 'score', 'label', n_buckets=100)
        self._raw_data = bucket_stats

        # build table at specified cutoffs
        total_bads = bucket_stats['bads'].sum()
        total_goods = bucket_stats['goods'].sum()
        total_count = bucket_stats['count'].sum()
        overall_bad_rate = total_bads / total_count

        rows = []
        for pct in cutoff_percentiles:
            target_pct = pct / 100

            # find closest bucket
            idx = (bucket_stats['cum_count_pct'] - target_pct).abs().argmin()
            row = bucket_stats.iloc[idx]

            rows.append({
                'population_pct': pct,
                'score_cutoff': row['min_score'],
                'n_approved': row['cum_count'],
                'n_bads_captured': row['cum_bads'],
                'n_goods_captured': row['cum_goods'],
                'bads_captured_pct': row['cum_bad_pct'] * 100,
                'goods_captured_pct': row['cum_good_pct'] * 100,
                'approval_bad_rate': row['cumulative_bad_rate'] * 100,
                'lift': row['lift']
            })

        self._table_data = pd.DataFrame(rows)
        return self

    def plot(self, **kwargs) -> 'CaptureRateTable':
        """
        Generate capture rate visualization as table figure.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        if self._table_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('off')

        # format table data
        table_display = self._table_data.copy()
        table_display['population_pct'] = table_display['population_pct'].apply(lambda x: f"{x}%")
        table_display['score_cutoff'] = table_display['score_cutoff'].apply(lambda x: f"{x:.0f}")
        table_display['n_approved'] = table_display['n_approved'].apply(lambda x: f"{x:,}")
        table_display['bads_captured_pct'] = table_display['bads_captured_pct'].apply(lambda x: f"{x:.1f}%")
        table_display['goods_captured_pct'] = table_display['goods_captured_pct'].apply(lambda x: f"{x:.1f}%")
        table_display['approval_bad_rate'] = table_display['approval_bad_rate'].apply(lambda x: f"{x:.2f}%")
        table_display['lift'] = table_display['lift'].apply(lambda x: f"{x:.2f}x")

        columns = ['population_pct', 'score_cutoff', 'n_approved',
                   'bads_captured_pct', 'goods_captured_pct', 'approval_bad_rate', 'lift']
        col_labels = ['Population', 'Score Cutoff', 'N Approved',
                      'Bads Captured', 'Goods Captured', 'Bad Rate', 'Lift']

        table = ax.table(
            cellText=table_display[columns].values,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(self.config.tick_fontsize)
        table.scale(1.2, 1.5)

        # style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor(self.config.colors.primary)
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title('Capture Rate Analysis', fontsize=self.config.title_fontsize, pad=20)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get table data for export."""
        if self._table_data is None:
            return {}

        return {
            'capture_rates': self._table_data.to_dict(orient='records')
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Get data as DataFrame."""
        if self._table_data is None:
            raise RuntimeError("Must call calculate() first")
        return self._table_data.copy()
