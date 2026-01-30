"""
Approval Rate / Bad Rate visualizations using Plotly.

Interactive versions with warm, eye-pleasing colors:
- AR/BR curve showing trade-off
- AR/BR table with detailed metrics
- Swap analysis for model comparison
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


@dataclass
class CutoffMetricsPl:
    """Metrics at a specific score cutoff."""
    score_cutoff: float
    approval_rate: float
    bad_rate: float
    n_approved: int
    n_rejected: int
    n_bads_approved: int
    n_goods_rejected: int
    expected_loss_rate: float


class ARBRCurvePl(BasePlotlyVisualization):
    """
    Interactive Approval Rate vs Bad Rate curve using Plotly.

    Shows the trade-off with interactive cutoff exploration.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._curve_data: Dict[str, pd.DataFrame] = {}

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        dataset_name: str = "Model"
    ) -> 'ARBRCurvePl':
        """
        Calculate AR/BR curve data.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        df = pd.DataFrame({'score': scores, 'label': labels})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        n_points = min(100, len(df))
        step = len(df) // n_points

        total_count = len(df)
        total_bads = df['label'].sum()

        points = []
        for i in range(step, len(df) + 1, step):
            approved = df.iloc[:i]
            n_approved = len(approved)
            n_bads_approved = approved['label'].sum()

            approval_rate = n_approved / total_count
            bad_rate = n_bads_approved / n_approved if n_approved > 0 else 0
            score_at_cutoff = approved['score'].iloc[-1]

            points.append({
                'score_cutoff': score_at_cutoff,
                'approval_rate': approval_rate,
                'bad_rate': bad_rate,
                'n_approved': n_approved,
                'n_bads_approved': n_bads_approved
            })

        self._curve_data[dataset_name] = pd.DataFrame(points)
        return self

    def plot(self, show_reference: bool = True, **kwargs) -> 'ARBRCurvePl':
        """
        Generate interactive AR/BR curve plot.

        Args:
            show_reference: If True, show overall bad rate line
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        fig = go.Figure()
        colors = self.config.colors

        for idx, (name, data) in enumerate(self._curve_data.items()):
            color = colors.palette[idx % len(colors.palette)]

            # Main AR/BR curve
            fig.add_trace(go.Scatter(
                x=data['approval_rate'] * 100,
                y=data['bad_rate'] * 100,
                mode='lines',
                name=name,
                line=dict(color=color, width=3),
                hovertemplate=(
                    "<b>Score Cutoff: %{customdata:.0f}</b><br>"
                    "Approval Rate: %{x:.1f}%<br>"
                    "Bad Rate: %{y:.2f}%<br>"
                    "<extra></extra>"
                ),
                customdata=data['score_cutoff']
            ))

            # Mark key points (50%, 70%, 90% approval)
            for ar_target in [0.5, 0.7, 0.9]:
                closest_idx = (data['approval_rate'] - ar_target).abs().argmin()
                point = data.iloc[closest_idx]

                fig.add_trace(go.Scatter(
                    x=[point['approval_rate'] * 100],
                    y=[point['bad_rate'] * 100],
                    mode='markers+text',
                    marker=dict(size=12, color=color, symbol='circle'),
                    text=[f"{point['bad_rate']*100:.1f}%"],
                    textposition='top right',
                    textfont=dict(size=11),
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{ar_target*100:.0f}% Approval</b><br>"
                        f"Score ≥ {point['score_cutoff']:.0f}<br>"
                        f"Bad Rate: {point['bad_rate']*100:.2f}%<br>"
                        f"N Approved: {point['n_approved']:,}<br>"
                        "<extra></extra>"
                    )
                ))

        if show_reference:
            first_data = list(self._curve_data.values())[0]
            overall_br = first_data['n_bads_approved'].iloc[-1] / first_data['n_approved'].iloc[-1]

            fig.add_hline(
                y=overall_br * 100,
                line_dash="dash",
                line_color=colors.text_secondary,
                annotation_text=f"Overall BR ({overall_br*100:.1f}%)",
                annotation_position="right"
            )

        layout = self._get_layout_defaults("Approval Rate vs Bad Rate Trade-off")
        layout.update({
            'xaxis_title': 'Approval Rate (%)',
            'yaxis_title': 'Bad Rate (%)',
            'xaxis': {**layout['xaxis'], 'range': [0, 105], 'ticksuffix': '%'},
            'yaxis': {**layout['yaxis'], 'ticksuffix': '%'},
        })
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get curve data for export."""
        result = {}
        for name, data in self._curve_data.items():
            result[name] = data.to_dict(orient='records')
        return result

    def get_bad_rate_at_ar(self, approval_rate: float, dataset_name: str = "Model") -> float:
        """Get bad rate at given approval rate."""
        if dataset_name not in self._curve_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._curve_data[dataset_name]
        return float(np.interp(
            approval_rate,
            data['approval_rate'].values,
            data['bad_rate'].values
        ))

    def find_cutoff_for_target(
        self,
        target_bad_rate: Optional[float] = None,
        target_approval_rate: Optional[float] = None,
        dataset_name: str = "Model"
    ) -> CutoffMetricsPl:
        """Find score cutoff to achieve target."""
        if dataset_name not in self._curve_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._curve_data[dataset_name]

        if target_bad_rate is not None:
            eligible = data[data['bad_rate'] <= target_bad_rate]
            point = eligible.iloc[-1] if len(eligible) > 0 else data.iloc[0]
        elif target_approval_rate is not None:
            idx = (data['approval_rate'] - target_approval_rate).abs().argmin()
            point = data.iloc[idx]
        else:
            raise ValueError("Must specify either target_bad_rate or target_approval_rate")

        total_count = data['n_approved'].iloc[-1]
        n_rejected = total_count - point['n_approved']
        n_goods_rejected = n_rejected - (data['n_bads_approved'].iloc[-1] - point['n_bads_approved'])

        return CutoffMetricsPl(
            score_cutoff=point['score_cutoff'],
            approval_rate=point['approval_rate'],
            bad_rate=point['bad_rate'],
            n_approved=int(point['n_approved']),
            n_rejected=int(n_rejected),
            n_bads_approved=int(point['n_bads_approved']),
            n_goods_rejected=int(max(0, n_goods_rejected)),
            expected_loss_rate=point['bad_rate']
        )


class ARBRTablePl(BasePlotlyVisualization):
    """
    Interactive AR/BR analysis table using Plotly.

    Shows detailed metrics at various score cutoffs.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._table_data: Optional[pd.DataFrame] = None

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        score_cutoffs: Optional[List[float]] = None,
        n_buckets: int = 10
    ) -> 'ARBRTablePl':
        """Calculate AR/BR table at specified cutoffs."""
        df = pd.DataFrame({'score': scores, 'label': labels})
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        total_count = len(df)
        total_bads = df['label'].sum()
        total_goods = total_count - total_bads

        if score_cutoffs is None:
            percentiles = np.linspace(0, 100, n_buckets + 1)[1:-1]
            score_cutoffs = np.percentile(scores, percentiles)
            score_cutoffs = sorted(set(score_cutoffs), reverse=True)

        rows = []
        for cutoff in score_cutoffs:
            approved = df[df['score'] >= cutoff]
            rejected = df[df['score'] < cutoff]

            n_approved = len(approved)
            n_rejected = len(rejected)
            n_bads_approved = approved['label'].sum()
            n_bads_rejected = rejected['label'].sum()
            n_goods_approved = n_approved - n_bads_approved
            n_goods_rejected = n_rejected - n_bads_rejected

            approval_rate = n_approved / total_count if total_count > 0 else 0
            bad_rate = n_bads_approved / n_approved if n_approved > 0 else 0
            rejection_rate = n_rejected / total_count if total_count > 0 else 0
            rejection_bad_rate = n_bads_rejected / n_rejected if n_rejected > 0 else 0

            bads_captured = n_bads_rejected / total_bads if total_bads > 0 else 0
            goods_lost = n_goods_rejected / total_goods if total_goods > 0 else 0

            rows.append({
                'score_cutoff': cutoff,
                'n_approved': n_approved,
                'n_rejected': n_rejected,
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate,
                'approved_bad_rate': bad_rate,
                'rejected_bad_rate': rejection_bad_rate,
                'bads_rejected_pct': bads_captured,
                'goods_rejected_pct': goods_lost,
                'n_bads_approved': n_bads_approved,
                'n_goods_rejected': n_goods_rejected
            })

        self._table_data = pd.DataFrame(rows)
        return self

    def plot(self, **kwargs) -> 'ARBRTablePl':
        """Generate interactive AR/BR table."""
        if self._table_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        colors = self.config.colors

        header_values = [
            '<b>Score<br>Cutoff</b>',
            '<b>N<br>Approved</b>',
            '<b>N<br>Rejected</b>',
            '<b>Approval<br>Rate</b>',
            '<b>Approved<br>Bad Rate</b>',
            '<b>Rejected<br>Bad Rate</b>',
            '<b>Bads<br>Rejected</b>',
            '<b>Goods<br>Rejected</b>'
        ]

        cell_values = [
            [f"{x:.0f}" for x in self._table_data['score_cutoff']],
            [f"{x:,.0f}" for x in self._table_data['n_approved']],
            [f"{x:,.0f}" for x in self._table_data['n_rejected']],
            [f"{x:.1%}" for x in self._table_data['approval_rate']],
            [f"{x:.2%}" for x in self._table_data['approved_bad_rate']],
            [f"{x:.2%}" for x in self._table_data['rejected_bad_rate']],
            [f"{x:.1%}" for x in self._table_data['bads_rejected_pct']],
            [f"{x:.1%}" for x in self._table_data['goods_rejected_pct']]
        ]

        # Color code bad rate columns
        def get_br_color(val, min_v, max_v):
            if max_v == min_v:
                return 'white'
            norm = (val - min_v) / (max_v - min_v)
            r = int(129 + (242 - 129) * norm)
            g = int(178 - (178 - 84) * norm)
            b = int(154 - (154 - 91) * norm)
            return f'rgba({r},{g},{b},0.25)'

        approved_brs = self._table_data['approved_bad_rate'].values
        fill_colors = [
            ['white'] * len(self._table_data),
            ['white'] * len(self._table_data),
            ['white'] * len(self._table_data),
            ['white'] * len(self._table_data),
            [get_br_color(x, approved_brs.min(), approved_brs.max()) for x in approved_brs],
            ['white'] * len(self._table_data),
            ['white'] * len(self._table_data),
            ['white'] * len(self._table_data),
        ]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color=colors.secondary,
                font=dict(color='white', size=12),
                align='center',
                height=40
            ),
            cells=dict(
                values=cell_values,
                fill_color=fill_colors,
                font=dict(size=11),
                align='center',
                height=32
            )
        )])

        layout = self._get_layout_defaults("Approval Rate / Bad Rate Analysis")
        layout['height'] = 500
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get table data for export."""
        if self._table_data is None:
            return {}
        return {'analysis': self._table_data.to_dict(orient='records')}

    def to_dataframe(self) -> pd.DataFrame:
        """Get data as DataFrame."""
        if self._table_data is None:
            raise RuntimeError("Must call calculate() first")
        return self._table_data.copy()


class SwapAnalysisPl(BasePlotlyVisualization):
    """
    Interactive swap analysis visualization using Plotly.

    Compares challenger vs champion model performance.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._swap_data: Optional[Dict] = None

    def calculate(
        self,
        scores_new: np.ndarray,
        scores_old: np.ndarray,
        labels: np.ndarray,
        cutoff_new: float,
        cutoff_old: float
    ) -> 'SwapAnalysisPl':
        """
        Calculate swap analysis.

        Args:
            scores_new: Scores from new (challenger) model
            scores_old: Scores from old (champion) model
            labels: True binary labels
            cutoff_new: Cutoff score for new model
            cutoff_old: Cutoff score for old model

        Returns:
            self for method chaining
        """
        n = len(labels)

        approved_new = scores_new >= cutoff_new
        approved_old = scores_old >= cutoff_old

        swap_in = approved_new & ~approved_old
        swap_out = ~approved_new & approved_old
        both_approve = approved_new & approved_old
        both_reject = ~approved_new & ~approved_old

        self._swap_data = {
            'summary': {
                'total': n,
                'both_approve': int(both_approve.sum()),
                'both_reject': int(both_reject.sum()),
                'swap_in': int(swap_in.sum()),
                'swap_out': int(swap_out.sum())
            },
            'swap_in': {
                'count': int(swap_in.sum()),
                'bads': int(labels[swap_in].sum()),
                'bad_rate': float(labels[swap_in].mean()) if swap_in.sum() > 0 else 0
            },
            'swap_out': {
                'count': int(swap_out.sum()),
                'bads': int(labels[swap_out].sum()),
                'bad_rate': float(labels[swap_out].mean()) if swap_out.sum() > 0 else 0
            },
            'both_approve': {
                'count': int(both_approve.sum()),
                'bads': int(labels[both_approve].sum()),
                'bad_rate': float(labels[both_approve].mean()) if both_approve.sum() > 0 else 0
            },
            'new_model': {
                'approved': int(approved_new.sum()),
                'approval_rate': float(approved_new.mean()),
                'bad_rate': float(labels[approved_new].mean()) if approved_new.sum() > 0 else 0
            },
            'old_model': {
                'approved': int(approved_old.sum()),
                'approval_rate': float(approved_old.mean()),
                'bad_rate': float(labels[approved_old].mean()) if approved_old.sum() > 0 else 0
            }
        }
        return self

    def plot(self, **kwargs) -> 'SwapAnalysisPl':
        """Generate interactive swap analysis visualization."""
        if self._swap_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        colors = self.config.colors

        # Create 2x1 subplot
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Population Distribution", "Bad Rates by Segment"),
            horizontal_spacing=0.12
        )

        # Left: Pie chart for population segments
        labels_pie = ['Both Approve', 'Both Reject', 'Swap In (New only)', 'Swap Out (Old only)']
        values_pie = [
            self._swap_data['summary']['both_approve'],
            self._swap_data['summary']['both_reject'],
            self._swap_data['summary']['swap_in'],
            self._swap_data['summary']['swap_out']
        ]
        colors_pie = [colors.success, colors.info, colors.warning, colors.danger]

        fig.add_trace(
            go.Pie(
                labels=labels_pie,
                values=values_pie,
                marker_colors=colors_pie,
                textinfo='label+percent',
                textposition='outside',
                hole=0.35,
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>"
            ),
            row=1, col=1
        )

        # Right: Bar chart for bad rates
        categories = ['Swap In', 'Swap Out', 'Both Approve', 'New Model', 'Old Model']
        bad_rates = [
            self._swap_data['swap_in']['bad_rate'] * 100,
            self._swap_data['swap_out']['bad_rate'] * 100,
            self._swap_data['both_approve']['bad_rate'] * 100,
            self._swap_data['new_model']['bad_rate'] * 100,
            self._swap_data['old_model']['bad_rate'] * 100
        ]
        bar_colors = [colors.warning, colors.danger, colors.success, colors.primary, colors.secondary]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=bad_rates,
                marker_color=bar_colors,
                text=[f"{x:.2f}%" for x in bad_rates],
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Bad Rate: %{y:.2f}%<extra></extra>"
            ),
            row=1, col=2
        )

        # Determine if new model is better
        swap_in_br = self._swap_data['swap_in']['bad_rate']
        swap_out_br = self._swap_data['swap_out']['bad_rate']
        improvement = swap_in_br < swap_out_br

        title_suffix = " ✓ New Model Better" if improvement else " ⚠ Old Model Better"
        title_color = colors.success if improvement else colors.danger

        layout = self._get_layout_defaults("Swap Analysis: New vs Old Model")
        layout.update({
            'height': 500,
            'showlegend': False,
        })
        fig.update_layout(**layout)

        fig.update_yaxes(title_text="Bad Rate (%)", ticksuffix="%", row=1, col=2)

        # Add annotation for verdict
        fig.add_annotation(
            text=f"<b>{title_suffix}</b>",
            xref="paper", yref="paper",
            x=0.5, y=1.08,
            showarrow=False,
            font=dict(size=14, color=title_color)
        )

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get swap analysis data for export."""
        return self._swap_data or {}

    def is_improvement(self) -> bool:
        """Check if new model improves over old model."""
        if self._swap_data is None:
            raise RuntimeError("Must call calculate() first")

        swap_in_br = self._swap_data['swap_in']['bad_rate']
        swap_out_br = self._swap_data['swap_out']['bad_rate']
        return swap_in_br < swap_out_br
