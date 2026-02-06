"""
Performance visualizations for reasoning models.

Includes:
- ROC curves with AUC
- AUC trend over time/samples
- Gini coefficient visualization
- KS (Kolmogorov-Smirnov) statistic plot
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from stages.visualization.base import (
    BaseVisualization,
    VisualizationConfig,
    format_percentage,
)


@dataclass
class ROCData:
    """Data for ROC curve."""
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    gini: float


class ROCCurve(BaseVisualization):
    """
    ROC (Receiver Operating Characteristic) curve visualization.

    Shows trade-off between true positive rate and false positive rate
    at various threshold settings. Area under curve (AUC) indicates
    overall model discriminatory power.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._roc_data: Dict[str, ROCData] = {}

    def add_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        label: str = "Model"
    ) -> 'ROCCurve':
        """
        Add a ROC curve to the plot.

        Args:
            y_true: True binary labels
            y_score: Predicted probabilities or scores
            label: Label for this curve (e.g., "Train", "Test")

        Returns:
            self for method chaining
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1

        self._roc_data[label] = ROCData(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc=auc,
            gini=gini
        )

        return self

    def plot(self, **kwargs) -> 'ROCCurve':
        """
        Generate ROC curve plot.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)

        colors = self.config.colors.palette

        for idx, (label, data) in enumerate(self._roc_data.items()):
            color = colors[idx % len(colors)]
            ax.plot(
                data.fpr,
                data.tpr,
                color=color,
                lw=2,
                label=f"{label} (AUC={data.auc:.4f}, Gini={data.gini:.4f})"
            )

        # diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=self.config.label_fontsize)
        ax.set_ylabel('True Positive Rate', fontsize=self.config.label_fontsize)
        ax.set_title('ROC Curve', fontsize=self.config.title_fontsize)
        ax.legend(loc='lower right', fontsize=self.config.legend_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get ROC curve data for export."""
        result = {}
        for label, data in self._roc_data.items():
            result[label] = {
                'fpr': data.fpr.tolist(),
                'tpr': data.tpr.tolist(),
                'thresholds': data.thresholds.tolist(),
                'auc': data.auc,
                'gini': data.gini
            }
        return result


class AUCTrend(BaseVisualization):
    """
    AUC trend visualization over time or samples.

    Shows how model performance changes:
    - Over different time periods (monthly/quarterly)
    - Across different sample populations
    - During model monitoring
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._trend_data: List[Dict[str, Any]] = []

    def add_point(
        self,
        period: str,
        train_auc: float,
        test_auc: Optional[float] = None,
        validation_auc: Optional[float] = None,
        **kwargs
    ) -> 'AUCTrend':
        """
        Add a point to the AUC trend.

        Args:
            period: Time period or sample name (e.g., "2024-Q1")
            train_auc: AUC on training data
            test_auc: AUC on test data
            validation_auc: AUC on validation/OOT data
            **kwargs: Additional metrics to track

        Returns:
            self for method chaining
        """
        point = {
            'period': period,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'validation_auc': validation_auc,
            **kwargs
        }
        self._trend_data.append(point)
        return self

    def from_dataframe(
        self,
        df: pd.DataFrame,
        period_col: str,
        y_true_col: str,
        y_score_col: str,
        sample_type_col: Optional[str] = None
    ) -> 'AUCTrend':
        """
        Calculate AUC trend from DataFrame.

        Args:
            df: DataFrame with predictions
            period_col: Column with time period
            y_true_col: Column with true labels
            y_score_col: Column with predictions
            sample_type_col: Optional column indicating train/test/valid

        Returns:
            self for method chaining
        """
        for period in df[period_col].unique():
            period_data = df[df[period_col] == period]

            if sample_type_col and sample_type_col in df.columns:
                train_data = period_data[period_data[sample_type_col] == 0]
                test_data = period_data[period_data[sample_type_col] == 1]
                valid_data = period_data[period_data[sample_type_col] == 2]

                train_auc = roc_auc_score(train_data[y_true_col], train_data[y_score_col]) if len(train_data) > 0 else None
                test_auc = roc_auc_score(test_data[y_true_col], test_data[y_score_col]) if len(test_data) > 0 else None
                valid_auc = roc_auc_score(valid_data[y_true_col], valid_data[y_score_col]) if len(valid_data) > 0 else None
            else:
                train_auc = roc_auc_score(period_data[y_true_col], period_data[y_score_col])
                test_auc = None
                valid_auc = None

            self.add_point(
                period=str(period),
                train_auc=train_auc,
                test_auc=test_auc,
                validation_auc=valid_auc
            )

        return self

    def plot(self, **kwargs) -> 'AUCTrend':
        """
        Generate AUC trend plot.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)

        periods = [d['period'] for d in self._trend_data]
        x = range(len(periods))

        colors = self.config.colors

        # plot train AUC
        train_aucs = [d['train_auc'] for d in self._trend_data]
        if any(v is not None for v in train_aucs):
            ax.plot(x, train_aucs, 'o-', color=colors.primary, lw=2, label='Train AUC')

        # plot test AUC
        test_aucs = [d['test_auc'] for d in self._trend_data]
        if any(v is not None for v in test_aucs):
            ax.plot(x, test_aucs, 's-', color=colors.secondary, lw=2, label='Test AUC')

        # plot validation AUC
        valid_aucs = [d['validation_auc'] for d in self._trend_data]
        if any(v is not None for v in valid_aucs):
            ax.plot(x, valid_aucs, '^-', color=colors.success, lw=2, label='Validation AUC')

        # reference line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')

        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45, ha='right')
        ax.set_xlabel('Period', fontsize=self.config.label_fontsize)
        ax.set_ylabel('AUC', fontsize=self.config.label_fontsize)
        ax.set_title('AUC Trend', fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=self.config.legend_fontsize)

        # set y-axis limits
        all_aucs = [v for v in train_aucs + test_aucs + valid_aucs if v is not None]
        if all_aucs:
            min_auc = min(all_aucs)
            max_auc = max(all_aucs)
            margin = (max_auc - min_auc) * 0.1 or 0.05
            ax.set_ylim([max(0.4, min_auc - margin), min(1.0, max_auc + margin)])

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get trend data for export."""
        return {'trend': self._trend_data}


class KSPlot(BaseVisualization):
    """
    Kolmogorov-Smirnov statistic visualization.

    Shows cumulative distribution functions for goods and bads,
    and highlights the maximum separation (KS statistic).
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._ks_data: Dict[str, Any] = {}

    def calculate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        label: str = "Model"
    ) -> 'KSPlot':
        """
        Calculate KS statistic and CDFs.

        Args:
            y_true: True binary labels (0=good, 1=bad)
            y_score: Predicted scores (higher = lower risk)
            label: Label for this dataset

        Returns:
            self for method chaining
        """
        # sort by score descending
        sorted_idx = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[sorted_idx]
        y_score_sorted = y_score[sorted_idx]

        n_total = len(y_true)
        n_bads = y_true.sum()
        n_goods = n_total - n_bads

        # cumulative counts
        cum_bads = np.cumsum(y_true_sorted)
        cum_goods = np.cumsum(1 - y_true_sorted)

        # cumulative percentages
        cum_bad_pct = cum_bads / n_bads
        cum_good_pct = cum_goods / n_goods

        # KS = max difference
        ks_diff = cum_bad_pct - cum_good_pct
        ks_stat = np.max(np.abs(ks_diff))
        ks_idx = np.argmax(np.abs(ks_diff))

        self._ks_data[label] = {
            'scores': y_score_sorted,
            'cum_bad_pct': cum_bad_pct,
            'cum_good_pct': cum_good_pct,
            'ks_stat': ks_stat,
            'ks_idx': ks_idx,
            'ks_score': y_score_sorted[ks_idx],
            'population_pct': np.arange(1, n_total + 1) / n_total
        }

        return self

    def plot(self, **kwargs) -> 'KSPlot':
        """
        Generate KS plot.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)

        colors = self.config.colors

        for label, data in self._ks_data.items():
            pop_pct = data['population_pct']

            # plot CDFs
            ax.plot(pop_pct, data['cum_bad_pct'], '-',
                    color=colors.danger, lw=2, label=f'{label} - Bads')
            ax.plot(pop_pct, data['cum_good_pct'], '-',
                    color=colors.success, lw=2, label=f'{label} - Goods')

            # mark KS point
            ks_x = pop_pct[data['ks_idx']]
            ks_bad = data['cum_bad_pct'][data['ks_idx']]
            ks_good = data['cum_good_pct'][data['ks_idx']]

            ax.vlines(x=ks_x, ymin=ks_good, ymax=ks_bad,
                      colors=colors.primary, linestyles='--', lw=2)
            ax.annotate(
                f"KS = {data['ks_stat']:.4f}",
                xy=(ks_x, (ks_bad + ks_good) / 2),
                xytext=(ks_x + 0.1, (ks_bad + ks_good) / 2),
                fontsize=self.config.label_fontsize,
                arrowprops=dict(arrowstyle='->', color=colors.primary)
            )

        # diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Population Percentage', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Cumulative Percentage', fontsize=self.config.label_fontsize)
        ax.set_title('KS (Kolmogorov-Smirnov) Chart', fontsize=self.config.title_fontsize)
        ax.legend(loc='lower right', fontsize=self.config.legend_fontsize)

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get KS data for export."""
        result = {}
        for label, data in self._ks_data.items():
            result[label] = {
                'ks_stat': data['ks_stat'],
                'ks_score_threshold': data['ks_score'],
                'population_at_ks': data['population_pct'][data['ks_idx']]
            }
        return result

    def get_ks_statistic(self, label: str = "Model") -> float:
        """Get KS statistic value."""
        if label in self._ks_data:
            return self._ks_data[label]['ks_stat']
        raise KeyError(f"No data for label: {label}")
