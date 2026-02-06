"""
Distribution visualizations for reasoning models using Plotly.

Interactive versions with warm, eye-pleasing colors:
- Score distribution histogram
- Hits by bucket
- Score density comparisons
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stages.visualization.base_pl import (
    BasePlotlyVisualization,
    PlotlyConfig,
)


class ScoreDistributionPl(BasePlotlyVisualization):
    """
    Interactive score distribution visualization using Plotly.

    Shows histogram with optional goods/bads separation and
    hover information.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._scores: Dict[str, np.ndarray] = {}
        self._labels: Optional[np.ndarray] = None
        self._n_bins: int = 50

    def set_data(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        dataset_name: str = "Overall"
    ) -> 'ScoreDistributionPl':
        """
        Set score data for distribution plot.

        Args:
            scores: Array of scores
            labels: Optional binary labels (0=good, 1=bad)
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        self._scores[dataset_name] = scores
        if labels is not None:
            self._labels = labels
        return self

    def set_bins(self, n_bins: int) -> 'ScoreDistributionPl':
        """Set number of histogram bins."""
        self._n_bins = n_bins
        return self

    def plot(self, separate_by_label: bool = True, **kwargs) -> 'ScoreDistributionPl':
        """
        Generate interactive score distribution plot.

        Args:
            separate_by_label: If True and labels provided, show goods/bads separately
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        fig = go.Figure()
        colors = self.config.colors

        for name, scores in self._scores.items():
            if separate_by_label and self._labels is not None:
                goods_scores = scores[self._labels == 0]
                bads_scores = scores[self._labels == 1]

                # Goods histogram
                fig.add_trace(go.Histogram(
                    x=goods_scores,
                    name=f'{name} - Goods (n={len(goods_scores):,})',
                    nbinsx=self._n_bins,
                    marker_color=colors.success,
                    opacity=0.7,
                    histnorm='probability density',
                    hovertemplate="Score: %{x:.0f}<br>Density: %{y:.4f}<extra>Goods</extra>"
                ))

                # Bads histogram
                fig.add_trace(go.Histogram(
                    x=bads_scores,
                    name=f'{name} - Bads (n={len(bads_scores):,})',
                    nbinsx=self._n_bins,
                    marker_color=colors.danger,
                    opacity=0.7,
                    histnorm='probability density',
                    hovertemplate="Score: %{x:.0f}<br>Density: %{y:.4f}<extra>Bads</extra>"
                ))
            else:
                fig.add_trace(go.Histogram(
                    x=scores,
                    name=f'{name} (n={len(scores):,})',
                    nbinsx=self._n_bins,
                    marker_color=colors.primary,
                    opacity=0.8,
                    histnorm='probability density',
                    hovertemplate="Score: %{x:.0f}<br>Density: %{y:.4f}<extra></extra>"
                ))

        layout = self._get_layout_defaults("Score Distribution")
        layout.update({
            'xaxis_title': 'Score',
            'yaxis_title': 'Density',
            'barmode': 'overlay'
        })
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get distribution statistics for export."""
        result = {}
        for name, scores in self._scores.items():
            stats = {
                'count': len(scores),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'percentiles': {
                    '5th': float(np.percentile(scores, 5)),
                    '25th': float(np.percentile(scores, 25)),
                    '75th': float(np.percentile(scores, 75)),
                    '95th': float(np.percentile(scores, 95))
                }
            }

            if self._labels is not None:
                goods_scores = scores[self._labels == 0]
                bads_scores = scores[self._labels == 1]
                stats['goods'] = {
                    'count': len(goods_scores),
                    'mean': float(np.mean(goods_scores)),
                    'std': float(np.std(goods_scores))
                }
                stats['bads'] = {
                    'count': len(bads_scores),
                    'mean': float(np.mean(bads_scores)),
                    'std': float(np.std(bads_scores))
                }

            result[name] = stats

        return result


class HitsByBucketPl(BasePlotlyVisualization):
    """
    Interactive hits by bucket visualization using Plotly.

    Shows population distribution across score buckets with
    bad rate overlay and detailed hover information.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._bucket_data: Optional[pd.DataFrame] = None
        self._n_buckets: int = 10

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_buckets: int = 10
    ) -> 'HitsByBucketPl':
        """
        Calculate bucket statistics.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            n_buckets: Number of buckets

        Returns:
            self for method chaining
        """
        self._n_buckets = n_buckets

        df = pd.DataFrame({'score': scores, 'label': labels})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        df['bucket'] = pd.qcut(
            df['score'].rank(method='first'),
            q=n_buckets,
            labels=range(1, n_buckets + 1)
        )

        bucket_stats = df.groupby('bucket').agg(
            count=('label', 'count'),
            bads=('label', 'sum'),
            min_score=('score', 'min'),
            max_score=('score', 'max'),
            mean_score=('score', 'mean')
        ).reset_index()

        bucket_stats['goods'] = bucket_stats['count'] - bucket_stats['bads']
        bucket_stats['bad_rate'] = bucket_stats['bads'] / bucket_stats['count']
        bucket_stats['population_pct'] = bucket_stats['count'] / bucket_stats['count'].sum()

        bucket_stats['cum_count'] = bucket_stats['count'].cumsum()
        bucket_stats['cum_bads'] = bucket_stats['bads'].cumsum()
        bucket_stats['cum_bad_rate'] = bucket_stats['cum_bads'] / bucket_stats['cum_count']

        self._bucket_data = bucket_stats
        return self

    def plot(self, show_bad_rate: bool = True, **kwargs) -> 'HitsByBucketPl':
        """
        Generate interactive hits by bucket chart.

        Args:
            show_bad_rate: If True, overlay bad rate line
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        if self._bucket_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        colors = self.config.colors

        bucket_labels = [
            f"Bucket {int(row['bucket'])}<br>({row['min_score']:.0f}-{row['max_score']:.0f})"
            for _, row in self._bucket_data.iterrows()
        ]

        # Goods bars
        fig.add_trace(
            go.Bar(
                x=bucket_labels,
                y=self._bucket_data['goods'],
                name='Goods',
                marker_color=colors.success,
                opacity=0.85,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Goods: %{y:,}<br>"
                    "<extra></extra>"
                )
            ),
            secondary_y=False
        )

        # Bads bars
        fig.add_trace(
            go.Bar(
                x=bucket_labels,
                y=self._bucket_data['bads'],
                name='Bads',
                marker_color=colors.danger,
                opacity=0.85,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Bads: %{y:,}<br>"
                    "<extra></extra>"
                )
            ),
            secondary_y=False
        )

        if show_bad_rate:
            # Bad rate line on secondary axis
            fig.add_trace(
                go.Scatter(
                    x=bucket_labels,
                    y=self._bucket_data['bad_rate'] * 100,
                    name='Bad Rate %',
                    mode='lines+markers',
                    line=dict(color=colors.primary, width=3),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Bad Rate: %{y:.2f}%<br>"
                        "<extra></extra>"
                    )
                ),
                secondary_y=True
            )

        layout = self._get_layout_defaults("Hits by Score Bucket")
        layout.update({
            'barmode': 'group',
            'xaxis_title': 'Score Bucket (1 = Highest Risk)',
        })
        fig.update_layout(**layout)

        fig.update_yaxes(title_text="Count", secondary_y=False)
        if show_bad_rate:
            fig.update_yaxes(
                title_text="Bad Rate (%)",
                secondary_y=True,
                ticksuffix="%",
                showgrid=False
            )

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get bucket data for export."""
        if self._bucket_data is None:
            return {}

        buckets = []
        for _, row in self._bucket_data.iterrows():
            buckets.append({
                'bucket': int(row['bucket']),
                'min_score': round(row['min_score'], 2),
                'max_score': round(row['max_score'], 2),
                'count': int(row['count']),
                'goods': int(row['goods']),
                'bads': int(row['bads']),
                'bad_rate': round(row['bad_rate'], 4),
                'population_pct': round(row['population_pct'], 4),
                'cum_bad_rate': round(row['cum_bad_rate'], 4)
            })

        return {'buckets': buckets, 'n_buckets': self._n_buckets}

    def to_table(self) -> pd.DataFrame:
        """Get bucket data as DataFrame."""
        if self._bucket_data is None:
            raise RuntimeError("Must call calculate() first")
        return self._bucket_data.copy()


class ScoreDensityComparisonPl(BasePlotlyVisualization):
    """
    Interactive score density comparison using Plotly.

    Shows smooth density curves for goods vs bads with
    fill and interactive hover.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._data: Dict[str, Dict] = {}

    def set_data(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        dataset_name: str = "Model"
    ) -> 'ScoreDensityComparisonPl':
        """Set score and label data."""
        self._data[dataset_name] = {
            'scores': scores,
            'labels': labels
        }
        return self

    def plot(self, **kwargs) -> 'ScoreDensityComparisonPl':
        """Generate interactive density comparison plot."""
        from scipy import stats as scipy_stats

        fig = go.Figure()
        colors = self.config.colors

        for name, data in self._data.items():
            scores = data['scores']
            labels = data['labels']

            goods_scores = scores[labels == 0]
            bads_scores = scores[labels == 1]

            score_range = np.linspace(scores.min(), scores.max(), 500)

            # Goods density
            if len(goods_scores) > 1:
                kde_goods = scipy_stats.gaussian_kde(goods_scores)
                goods_density = kde_goods(score_range)

                fig.add_trace(go.Scatter(
                    x=score_range,
                    y=goods_density,
                    mode='lines',
                    name=f'{name} - Goods (n={len(goods_scores):,})',
                    line=dict(color=colors.success, width=3),
                    hovertemplate="Score: %{x:.0f}<br>Density: %{y:.4f}<extra>Goods</extra>"
                ))

                # Fill under goods curve
                fig.add_trace(go.Scatter(
                    x=score_range,
                    y=goods_density,
                    fill='tozeroy',
                    fillcolor=f'rgba(129, 178, 154, 0.25)',  # Soft sage green
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Bads density
            if len(bads_scores) > 1:
                kde_bads = scipy_stats.gaussian_kde(bads_scores)
                bads_density = kde_bads(score_range)

                fig.add_trace(go.Scatter(
                    x=score_range,
                    y=bads_density,
                    mode='lines',
                    name=f'{name} - Bads (n={len(bads_scores):,})',
                    line=dict(color=colors.danger, width=3),
                    hovertemplate="Score: %{x:.0f}<br>Density: %{y:.4f}<extra>Bads</extra>"
                ))

                # Fill under bads curve
                fig.add_trace(go.Scatter(
                    x=score_range,
                    y=bads_density,
                    fill='tozeroy',
                    fillcolor=f'rgba(242, 84, 91, 0.25)',  # Soft coral red
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            # Add mean lines
            goods_mean = np.mean(goods_scores)
            bads_mean = np.mean(bads_scores)

            fig.add_vline(
                x=goods_mean,
                line_dash="dash",
                line_color=colors.success,
                annotation_text=f"Goods μ={goods_mean:.0f}",
                annotation_position="top"
            )
            fig.add_vline(
                x=bads_mean,
                line_dash="dash",
                line_color=colors.danger,
                annotation_text=f"Bads μ={bads_mean:.0f}",
                annotation_position="top"
            )

        layout = self._get_layout_defaults("Score Density: Goods vs Bads")
        layout.update({
            'xaxis_title': 'Score',
            'yaxis_title': 'Density',
        })
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get summary statistics for export."""
        result = {}
        for name, data in self._data.items():
            scores = data['scores']
            labels = data['labels']

            goods_scores = scores[labels == 0]
            bads_scores = scores[labels == 1]

            result[name] = {
                'goods': {
                    'count': len(goods_scores),
                    'mean': float(np.mean(goods_scores)),
                    'std': float(np.std(goods_scores)),
                    'median': float(np.median(goods_scores))
                },
                'bads': {
                    'count': len(bads_scores),
                    'mean': float(np.mean(bads_scores)),
                    'std': float(np.std(bads_scores)),
                    'median': float(np.median(bads_scores))
                },
                'separation': {
                    'mean_diff': float(np.mean(goods_scores) - np.mean(bads_scores))
                }
            }

        return result
