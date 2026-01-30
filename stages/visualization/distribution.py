"""
Distribution visualizations for scorecard models.

Includes:
- Score distribution histogram
- Hits by bucket (population distribution across score ranges)
- Score density plots comparing goods vs bads
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stages.visualization.base import (
    BaseVisualization,
    VisualizationConfig,
    calculate_buckets,
    format_percentage,
    format_number,
)


@dataclass
class BucketStats:
    """Statistics for a single score bucket."""
    bucket_id: int
    min_score: float
    max_score: float
    count: int
    goods: int
    bads: int
    bad_rate: float
    population_pct: float


class ScoreDistribution(BaseVisualization):
    """
    Score distribution visualization.

    Shows histogram of scores with optional overlay of
    goods vs bads distributions. Useful for understanding
    score spread and separation.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._scores: Dict[str, np.ndarray] = {}
        self._labels: Optional[np.ndarray] = None
        self._n_bins: int = 50

    def set_data(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        dataset_name: str = "Overall"
    ) -> 'ScoreDistribution':
        """
        Set score data for distribution plot.

        Args:
            scores: Array of scores
            labels: Optional binary labels (0=good, 1=bad) for separation
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        self._scores[dataset_name] = scores
        if labels is not None:
            self._labels = labels
        return self

    def set_bins(self, n_bins: int) -> 'ScoreDistribution':
        """Set number of histogram bins."""
        self._n_bins = n_bins
        return self

    def plot(self, separate_by_label: bool = True, **kwargs) -> 'ScoreDistribution':
        """
        Generate score distribution plot.

        Args:
            separate_by_label: If True and labels provided, show goods/bads separately
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)
        colors = self.config.colors

        for name, scores in self._scores.items():
            if separate_by_label and self._labels is not None:
                # separate goods and bads
                goods_scores = scores[self._labels == 0]
                bads_scores = scores[self._labels == 1]

                # determine common bins
                all_min = min(scores.min(), goods_scores.min(), bads_scores.min())
                all_max = max(scores.max(), goods_scores.max(), bads_scores.max())
                bins = np.linspace(all_min, all_max, self._n_bins + 1)

                # plot goods
                ax.hist(
                    goods_scores, bins=bins, alpha=0.6,
                    color=colors.success, label=f'{name} - Goods',
                    density=True
                )
                # plot bads
                ax.hist(
                    bads_scores, bins=bins, alpha=0.6,
                    color=colors.danger, label=f'{name} - Bads',
                    density=True
                )
            else:
                # single distribution
                ax.hist(
                    scores, bins=self._n_bins, alpha=0.7,
                    color=colors.primary, label=name,
                    density=True
                )

        ax.set_xlabel('Score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Density', fontsize=self.config.label_fontsize)
        ax.set_title('Score Distribution', fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=self.config.legend_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='y')

        if self.config.tight_layout:
            fig.tight_layout()

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


class HitsByBucket(BaseVisualization):
    """
    Hits by bucket visualization.

    Shows how the population is distributed across score buckets,
    along with bad rate per bucket. Essential for understanding
    model discrimination and setting cutoffs.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._bucket_data: Optional[pd.DataFrame] = None
        self._n_buckets: int = 10

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_buckets: int = 10
    ) -> 'HitsByBucket':
        """
        Calculate bucket statistics.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            n_buckets: Number of buckets to create

        Returns:
            self for method chaining
        """
        self._n_buckets = n_buckets

        # create DataFrame for easier manipulation
        df = pd.DataFrame({
            'score': scores,
            'label': labels
        })

        # sort by score descending (higher score = lower risk)
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        # create equal-frequency buckets
        df['bucket'] = pd.qcut(
            df['score'].rank(method='first'),
            q=n_buckets,
            labels=range(1, n_buckets + 1)
        )

        # aggregate
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

        # cumulative stats
        bucket_stats['cum_count'] = bucket_stats['count'].cumsum()
        bucket_stats['cum_bads'] = bucket_stats['bads'].cumsum()
        bucket_stats['cum_bad_rate'] = bucket_stats['cum_bads'] / bucket_stats['cum_count']

        self._bucket_data = bucket_stats
        return self

    def plot(self, show_bad_rate: bool = True, **kwargs) -> 'HitsByBucket':
        """
        Generate hits by bucket bar chart.

        Args:
            show_bad_rate: If True, overlay bad rate line
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        if self._bucket_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        fig, ax1 = plt.subplots(figsize=self.config.figsize)
        colors = self.config.colors

        buckets = self._bucket_data['bucket'].values
        x = np.arange(len(buckets))
        width = 0.35

        # bar chart for goods and bads
        goods_bars = ax1.bar(
            x - width/2,
            self._bucket_data['goods'],
            width,
            label='Goods',
            color=colors.success,
            alpha=0.8
        )
        bads_bars = ax1.bar(
            x + width/2,
            self._bucket_data['bads'],
            width,
            label='Bads',
            color=colors.danger,
            alpha=0.8
        )

        ax1.set_xlabel('Score Bucket (1=Highest Risk)', fontsize=self.config.label_fontsize)
        ax1.set_ylabel('Count', fontsize=self.config.label_fontsize)
        ax1.set_xticks(x)

        # create bucket labels with score ranges
        bucket_labels = []
        for _, row in self._bucket_data.iterrows():
            bucket_labels.append(f"{int(row['bucket'])}\n({row['min_score']:.0f}-{row['max_score']:.0f})")
        ax1.set_xticklabels(bucket_labels, fontsize=self.config.tick_fontsize - 2)

        if show_bad_rate:
            # overlay bad rate on secondary axis
            ax2 = ax1.twinx()
            ax2.plot(
                x,
                self._bucket_data['bad_rate'] * 100,
                'o-',
                color=colors.primary,
                lw=2,
                markersize=8,
                label='Bad Rate %'
            )
            ax2.set_ylabel('Bad Rate (%)', fontsize=self.config.label_fontsize, color=colors.primary)
            ax2.tick_params(axis='y', labelcolor=colors.primary)

            # combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                       fontsize=self.config.legend_fontsize)
        else:
            ax1.legend(loc='upper right', fontsize=self.config.legend_fontsize)

        ax1.set_title('Hits by Score Bucket', fontsize=self.config.title_fontsize)

        if self.config.show_grid:
            ax1.grid(True, alpha=0.3, axis='y')

        if self.config.tight_layout:
            fig.tight_layout()

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
        """Get bucket data as formatted DataFrame."""
        if self._bucket_data is None:
            raise RuntimeError("Must call calculate() first")

        table = self._bucket_data.copy()
        table['score_range'] = table.apply(
            lambda r: f"{r['min_score']:.0f} - {r['max_score']:.0f}",
            axis=1
        )
        table['bad_rate_fmt'] = table['bad_rate'].apply(lambda x: format_percentage(x))
        table['population_pct_fmt'] = table['population_pct'].apply(lambda x: format_percentage(x))

        return table[[
            'bucket', 'score_range', 'count', 'goods', 'bads',
            'bad_rate_fmt', 'population_pct_fmt'
        ]]


class ScoreDensityComparison(BaseVisualization):
    """
    Score density comparison using kernel density estimation.

    Shows smooth density curves for goods and bads to visualize
    separation between populations.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._data: Dict[str, Dict] = {}

    def set_data(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        dataset_name: str = "Model"
    ) -> 'ScoreDensityComparison':
        """
        Set score and label data.

        Args:
            scores: Array of scores
            labels: Binary labels (0=good, 1=bad)
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        self._data[dataset_name] = {
            'scores': scores,
            'labels': labels
        }
        return self

    def plot(self, **kwargs) -> 'ScoreDensityComparison':
        """
        Generate density comparison plot.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats

        fig, ax = plt.subplots(figsize=self.config.figsize)
        colors = self.config.colors

        for name, data in self._data.items():
            scores = data['scores']
            labels = data['labels']

            goods_scores = scores[labels == 0]
            bads_scores = scores[labels == 1]

            # create smooth density using KDE
            score_range = np.linspace(scores.min(), scores.max(), 500)

            if len(goods_scores) > 1:
                kde_goods = scipy_stats.gaussian_kde(goods_scores)
                ax.plot(
                    score_range,
                    kde_goods(score_range),
                    '-',
                    color=colors.success,
                    lw=2,
                    label=f'{name} - Goods (n={len(goods_scores):,})'
                )
                ax.fill_between(
                    score_range,
                    kde_goods(score_range),
                    alpha=0.3,
                    color=colors.success
                )

            if len(bads_scores) > 1:
                kde_bads = scipy_stats.gaussian_kde(bads_scores)
                ax.plot(
                    score_range,
                    kde_bads(score_range),
                    '-',
                    color=colors.danger,
                    lw=2,
                    label=f'{name} - Bads (n={len(bads_scores):,})'
                )
                ax.fill_between(
                    score_range,
                    kde_bads(score_range),
                    alpha=0.3,
                    color=colors.danger
                )

        ax.set_xlabel('Score', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Density', fontsize=self.config.label_fontsize)
        ax.set_title('Score Density: Goods vs Bads', fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=self.config.legend_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

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
                    'mean_diff': float(np.mean(goods_scores) - np.mean(bads_scores)),
                    'overlap_index': self._calculate_overlap(goods_scores, bads_scores)
                }
            }

        return result

    def _calculate_overlap(self, goods: np.ndarray, bads: np.ndarray) -> float:
        """Calculate overlap index between two distributions."""
        # simplified overlap calculation using histograms
        min_val = min(goods.min(), bads.min())
        max_val = max(goods.max(), bads.max())
        bins = np.linspace(min_val, max_val, 100)

        hist_goods, _ = np.histogram(goods, bins=bins, density=True)
        hist_bads, _ = np.histogram(bads, bins=bins, density=True)

        # overlap = sum of minimum at each bin
        overlap = np.sum(np.minimum(hist_goods, hist_bads)) * (bins[1] - bins[0])
        return float(overlap)
