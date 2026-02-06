"""
Performance visualizations for reasoning models using Plotly.

Interactive versions with warm, eye-pleasing colors:
- ROC curves with AUC
- AUC trend over time/samples
- KS (Kolmogorov-Smirnov) statistic plot
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stages.visualization.base_pl import (
    BasePlotlyVisualization,
    PlotlyConfig,
)


@dataclass
class ROCDataPl:
    """Data for ROC curve."""
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float
    gini: float


class ROCCurvePl(BasePlotlyVisualization):
    """
    Interactive ROC curve visualization using Plotly.

    Shows trade-off between true positive rate and false positive rate
    with hover information showing threshold values.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._roc_data: Dict[str, ROCDataPl] = {}

    def add_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        label: str = "Model"
    ) -> 'ROCCurvePl':
        """
        Add a ROC curve to the plot.

        Args:
            y_true: True binary labels
            y_score: Predicted probabilities or scores
            label: Label for this curve

        Returns:
            self for method chaining
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1

        self._roc_data[label] = ROCDataPl(
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
            auc=auc,
            gini=gini
        )
        return self

    def plot(self, **kwargs) -> 'ROCCurvePl':
        """Generate interactive ROC curve plot."""
        fig = go.Figure()
        colors = self.config.colors.palette

        for idx, (label, data) in enumerate(self._roc_data.items()):
            color = colors[idx % len(colors)]

            # main ROC curve with hover info
            fig.add_trace(go.Scatter(
                x=data.fpr,
                y=data.tpr,
                mode='lines',
                name=f"{label} (AUC={data.auc:.4f}, Gini={data.gini:.4f})",
                line=dict(color=color, width=3),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "FPR: %{x:.3f}<br>"
                    "TPR: %{y:.3f}<br>"
                    "<extra></extra>"
                ),
                text=[f"Threshold: {t:.3f}" for t in data.thresholds]
            ))

        # diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random (AUC=0.5)',
            line=dict(color=self.config.colors.text_secondary, width=2, dash='dash'),
            hoverinfo='skip'
        ))

        # fill area under best curve
        if self._roc_data:
            best_label = max(self._roc_data, key=lambda k: self._roc_data[k].auc)
            best_data = self._roc_data[best_label]
            fig.add_trace(go.Scatter(
                x=best_data.fpr,
                y=best_data.tpr,
                fill='tozeroy',
                fillcolor=f'rgba(224, 122, 95, 0.15)',  # Soft terracotta fill
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

        layout = self._get_layout_defaults("ROC Curve")
        layout.update({
            'xaxis_title': 'False Positive Rate',
            'yaxis_title': 'True Positive Rate',
            'xaxis': {**layout['xaxis'], 'range': [0, 1]},
            'yaxis': {**layout['yaxis'], 'range': [0, 1.02]},
            'legend': {**layout['legend'], 'x': 0.99, 'y': 0.01, 'xanchor': 'right', 'yanchor': 'bottom'}
        })
        fig.update_layout(**layout)

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


class AUCTrendPl(BasePlotlyVisualization):
    """
    Interactive AUC trend visualization using Plotly.

    Shows model performance over time with hover details.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._trend_data: List[Dict[str, Any]] = []

    def add_point(
        self,
        period: str,
        train_auc: float,
        test_auc: Optional[float] = None,
        validation_auc: Optional[float] = None,
        **kwargs
    ) -> 'AUCTrendPl':
        """
        Add a point to the AUC trend.

        Args:
            period: Time period or sample name
            train_auc: AUC on training data
            test_auc: AUC on test data
            validation_auc: AUC on validation data
            **kwargs: Additional metrics

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
    ) -> 'AUCTrendPl':
        """Calculate AUC trend from DataFrame."""
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

            self.add_point(str(period), train_auc, test_auc, valid_auc)

        return self

    def plot(self, **kwargs) -> 'AUCTrendPl':
        """Generate interactive AUC trend plot."""
        fig = go.Figure()
        colors = self.config.colors

        periods = [d['period'] for d in self._trend_data]

        # Train AUC
        train_aucs = [d['train_auc'] for d in self._trend_data]
        if any(v is not None for v in train_aucs):
            fig.add_trace(go.Scatter(
                x=periods,
                y=train_aucs,
                mode='lines+markers',
                name='Train AUC',
                line=dict(color=colors.primary, width=3),
                marker=dict(size=10, symbol='circle'),
                hovertemplate="<b>%{x}</b><br>Train AUC: %{y:.4f}<extra></extra>"
            ))

        # Test AUC
        test_aucs = [d['test_auc'] for d in self._trend_data]
        if any(v is not None for v in test_aucs):
            fig.add_trace(go.Scatter(
                x=periods,
                y=test_aucs,
                mode='lines+markers',
                name='Test AUC',
                line=dict(color=colors.secondary, width=3),
                marker=dict(size=10, symbol='square'),
                hovertemplate="<b>%{x}</b><br>Test AUC: %{y:.4f}<extra></extra>"
            ))

        # Validation AUC
        valid_aucs = [d['validation_auc'] for d in self._trend_data]
        if any(v is not None for v in valid_aucs):
            fig.add_trace(go.Scatter(
                x=periods,
                y=valid_aucs,
                mode='lines+markers',
                name='Validation AUC',
                line=dict(color=colors.success, width=3),
                marker=dict(size=10, symbol='diamond'),
                hovertemplate="<b>%{x}</b><br>Validation AUC: %{y:.4f}<extra></extra>"
            ))

        # Reference line at 0.5
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color=colors.text_secondary,
            annotation_text="Random",
            annotation_position="right"
        )

        # Calculate y-axis range
        all_aucs = [v for v in train_aucs + test_aucs + valid_aucs if v is not None]
        if all_aucs:
            min_auc = min(all_aucs)
            max_auc = max(all_aucs)
            margin = (max_auc - min_auc) * 0.15 or 0.05
            y_range = [max(0.4, min_auc - margin), min(1.0, max_auc + margin)]
        else:
            y_range = [0.4, 1.0]

        layout = self._get_layout_defaults("AUC Trend")
        layout.update({
            'xaxis_title': 'Period',
            'yaxis_title': 'AUC',
            'yaxis': {**layout['yaxis'], 'range': y_range},
        })
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get trend data for export."""
        return {'trend': self._trend_data}


class KSPlotPl(BasePlotlyVisualization):
    """
    Interactive Kolmogorov-Smirnov visualization using Plotly.

    Shows cumulative distributions with interactive KS point.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._ks_data: Dict[str, Any] = {}

    def calculate(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        label: str = "Model"
    ) -> 'KSPlotPl':
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

        cum_bads = np.cumsum(y_true_sorted)
        cum_goods = np.cumsum(1 - y_true_sorted)

        cum_bad_pct = cum_bads / n_bads
        cum_good_pct = cum_goods / n_goods

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

    def plot(self, **kwargs) -> 'KSPlotPl':
        """Generate interactive KS plot."""
        fig = go.Figure()
        colors = self.config.colors

        for label, data in self._ks_data.items():
            pop_pct = data['population_pct']

            # Bads CDF
            fig.add_trace(go.Scatter(
                x=pop_pct,
                y=data['cum_bad_pct'],
                mode='lines',
                name=f'{label} - Bads',
                line=dict(color=colors.danger, width=3),
                hovertemplate="Population: %{x:.1%}<br>Bads Captured: %{y:.1%}<extra></extra>"
            ))

            # Goods CDF
            fig.add_trace(go.Scatter(
                x=pop_pct,
                y=data['cum_good_pct'],
                mode='lines',
                name=f'{label} - Goods',
                line=dict(color=colors.success, width=3),
                hovertemplate="Population: %{x:.1%}<br>Goods Captured: %{y:.1%}<extra></extra>"
            ))

            # KS point annotation
            ks_x = pop_pct[data['ks_idx']]
            ks_bad = data['cum_bad_pct'][data['ks_idx']]
            ks_good = data['cum_good_pct'][data['ks_idx']]

            # Vertical line at KS point
            fig.add_trace(go.Scatter(
                x=[ks_x, ks_x],
                y=[ks_good, ks_bad],
                mode='lines+markers',
                name=f'KS = {data["ks_stat"]:.4f}',
                line=dict(color=colors.primary, width=3, dash='dash'),
                marker=dict(size=12, symbol='diamond'),
                hovertemplate=f"KS Statistic: {data['ks_stat']:.4f}<br>Score: {data['ks_score']:.0f}<extra></extra>"
            ))

            # Add annotation
            fig.add_annotation(
                x=ks_x,
                y=(ks_bad + ks_good) / 2,
                text=f"KS = {data['ks_stat']:.4f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors.primary,
                ax=60,
                ay=0,
                font=dict(size=14, color=colors.primary)
            )

        # Diagonal reference
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color=colors.text_secondary, width=1, dash='dot'),
            hoverinfo='skip'
        ))

        layout = self._get_layout_defaults("KS (Kolmogorov-Smirnov) Chart")
        layout.update({
            'xaxis_title': 'Population Percentage',
            'yaxis_title': 'Cumulative Percentage',
            'xaxis': {**layout['xaxis'], 'range': [0, 1], 'tickformat': '.0%'},
            'yaxis': {**layout['yaxis'], 'range': [0, 1.02], 'tickformat': '.0%'},
            'legend': {**layout['legend'], 'x': 0.99, 'y': 0.01, 'xanchor': 'right', 'yanchor': 'bottom'}
        })
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get KS data for export."""
        result = {}
        for label, data in self._ks_data.items():
            result[label] = {
                'ks_stat': data['ks_stat'],
                'ks_score_threshold': float(data['ks_score']),
                'population_at_ks': float(data['population_pct'][data['ks_idx']])
            }
        return result

    def get_ks_statistic(self, label: str = "Model") -> float:
        """Get KS statistic value."""
        if label in self._ks_data:
            return self._ks_data[label]['ks_stat']
        raise KeyError(f"No data for label: {label}")
