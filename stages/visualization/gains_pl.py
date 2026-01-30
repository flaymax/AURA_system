"""
Gains and lift chart visualizations using Plotly.

Interactive versions with warm, eye-pleasing colors:
- Cumulative gains chart
- Lift chart
- Capture rate analysis
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
from stages.visualization.base import calculate_cumulative_stats


class GainsChartPl(BasePlotlyVisualization):
    """
    Interactive cumulative gains chart using Plotly.

    Shows what percentage of bads are captured when selecting
    top X% of population, with hover details.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._gains_data: Dict[str, pd.DataFrame] = {}

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_buckets: int = 10,
        dataset_name: str = "Model"
    ) -> 'GainsChartPl':
        """
        Calculate cumulative gains.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            n_buckets: Number of buckets
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        df = pd.DataFrame({'score': scores, 'label': labels})
        bucket_stats = calculate_cumulative_stats(df, 'score', 'label', n_buckets)
        self._gains_data[dataset_name] = bucket_stats
        return self

    def plot(self, show_perfect: bool = True, **kwargs) -> 'GainsChartPl':
        """
        Generate interactive cumulative gains chart.

        Args:
            show_perfect: If True, show perfect model line
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        fig = go.Figure()
        colors = self.config.colors

        for idx, (name, data) in enumerate(self._gains_data.items()):
            color = colors.palette[idx % len(colors.palette)]

            # Prepend (0, 0) point
            x = np.concatenate([[0], data['cum_count_pct'].values])
            y = np.concatenate([[0], data['cum_bad_pct'].values])

            # Fill area under curve
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                fill='tozeroy',
                fillcolor=f'rgba(224, 122, 95, 0.2)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Main gains curve
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=3),
                marker=dict(size=8),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Population: %{x:.1%}<br>"
                    "Bads Captured: %{y:.1%}<br>"
                    "<extra></extra>"
                ),
                text=[f"Bucket {i}" for i in range(len(x))]
            ))

        # Random model (diagonal)
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color=colors.text_secondary, width=2, dash='dash'),
            hoverinfo='skip'
        ))

        if show_perfect and self._gains_data:
            # Perfect model line
            first_data = list(self._gains_data.values())[0]
            total_bads = first_data['bads'].sum()
            total_count = first_data['count'].sum()
            bad_proportion = total_bads / total_count

            fig.add_trace(go.Scatter(
                x=[0, bad_proportion, 1],
                y=[0, 1, 1],
                mode='lines',
                name='Perfect Model',
                line=dict(color=colors.success, width=2, dash='dot'),
                hovertemplate="Perfect model captures 100% of bads<br>at %{x:.1%} of population<extra></extra>"
            ))

        layout = self._get_layout_defaults("Cumulative Gains Chart")
        layout.update({
            'xaxis_title': 'Proportion of Population',
            'yaxis_title': 'Proportion of Bads Captured',
            'xaxis': {**layout['xaxis'], 'range': [0, 1], 'tickformat': '.0%'},
            'yaxis': {**layout['yaxis'], 'range': [0, 1.05], 'tickformat': '.0%'},
            'legend': {**layout['legend'], 'x': 0.99, 'y': 0.01, 'xanchor': 'right', 'yanchor': 'bottom'}
        })
        fig.update_layout(**layout)

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
        """Calculate area under gains curve."""
        x = np.concatenate([[0], data['cum_count_pct'].values])
        y = np.concatenate([[0], data['cum_bad_pct'].values])
        return float(np.trapz(y, x))

    def get_capture_rate(self, population_pct: float, dataset_name: str = "Model") -> float:
        """Get bads capture rate at given population percentage."""
        if dataset_name not in self._gains_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._gains_data[dataset_name]
        return float(np.interp(
            population_pct,
            np.concatenate([[0], data['cum_count_pct'].values]),
            np.concatenate([[0], data['cum_bad_pct'].values])
        ))


class LiftChartPl(BasePlotlyVisualization):
    """
    Interactive lift chart using Plotly.

    Shows how much better the model is at identifying bads
    compared to random selection.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._lift_data: Dict[str, pd.DataFrame] = {}

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        n_buckets: int = 10,
        dataset_name: str = "Model"
    ) -> 'LiftChartPl':
        """Calculate lift by bucket."""
        df = pd.DataFrame({'score': scores, 'label': labels})
        bucket_stats = calculate_cumulative_stats(df, 'score', 'label', n_buckets)
        self._lift_data[dataset_name] = bucket_stats
        return self

    def plot(self, cumulative: bool = True, **kwargs) -> 'LiftChartPl':
        """
        Generate interactive lift chart.

        Args:
            cumulative: If True, show cumulative lift. If False, per-bucket lift.
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        fig = go.Figure()
        colors = self.config.colors

        for idx, (name, data) in enumerate(self._lift_data.items()):
            color = colors.palette[idx % len(colors.palette)]
            overall_bad_rate = data['bads'].sum() / data['count'].sum()

            if cumulative:
                # Cumulative lift
                fig.add_trace(go.Scatter(
                    x=data['cum_count_pct'],
                    y=data['lift'],
                    mode='lines+markers',
                    name=name,
                    line=dict(color=color, width=3),
                    marker=dict(size=10),
                    hovertemplate=(
                        "<b>Bucket %{text}</b><br>"
                        "Population: %{x:.1%}<br>"
                        "Cumulative Lift: %{y:.2f}x<br>"
                        "<extra></extra>"
                    ),
                    text=[str(int(b)) for b in data['bucket']]
                ))

                # Fill under curve (above 1.0)
                fig.add_trace(go.Scatter(
                    x=data['cum_count_pct'],
                    y=[max(1, l) for l in data['lift']],
                    fill='tonexty',
                    fillcolor=f'rgba(244, 162, 97, 0.2)',  # Soft orange
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            else:
                # Per-bucket lift
                bucket_lift = data['bad_rate'] / overall_bad_rate

                fig.add_trace(go.Bar(
                    x=[f"Bucket {int(b)}" for b in data['bucket']],
                    y=bucket_lift,
                    name=name,
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate=(
                        "<b>%{x}</b><br>"
                        "Lift: %{y:.2f}x<br>"
                        "Bad Rate: " + data['bad_rate'].apply(lambda x: f"{x:.2%}").tolist()[0] + "<br>"
                        "<extra></extra>"
                    )
                ))

        # Reference line at lift = 1
        if cumulative:
            fig.add_hline(
                y=1,
                line_dash="dash",
                line_color=colors.text_secondary,
                annotation_text="No Lift (1.0x)",
                annotation_position="right"
            )
        else:
            fig.add_hline(
                y=1,
                line_dash="dash",
                line_color=colors.text_secondary
            )

        title = 'Lift Chart (Cumulative)' if cumulative else 'Lift Chart (Per Bucket)'
        layout = self._get_layout_defaults(title)

        if cumulative:
            layout.update({
                'xaxis_title': 'Proportion of Population',
                'yaxis_title': 'Lift',
                'xaxis': {**layout['xaxis'], 'tickformat': '.0%'},
            })
        else:
            layout.update({
                'xaxis_title': 'Score Bucket (1 = Highest Risk)',
                'yaxis_title': 'Lift',
            })

        fig.update_layout(**layout)

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
        """Get cumulative lift at given population percentage."""
        if dataset_name not in self._lift_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._lift_data[dataset_name]
        return float(np.interp(
            population_pct,
            data['cum_count_pct'].values,
            data['lift'].values
        ))


class CaptureRateTablePl(BasePlotlyVisualization):
    """
    Interactive capture rate analysis table using Plotly.

    Shows key metrics at various population cutoffs with
    formatted table display.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        super().__init__(config)
        self._table_data: Optional[pd.DataFrame] = None
        self._raw_data: Optional[pd.DataFrame] = None

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        cutoff_percentiles: Optional[List[int]] = None
    ) -> 'CaptureRateTablePl':
        """
        Calculate capture rates at various cutoffs.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            cutoff_percentiles: List of percentiles (default: 10, 20, ..., 90)

        Returns:
            self for method chaining
        """
        if cutoff_percentiles is None:
            cutoff_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        df = pd.DataFrame({'score': scores, 'label': labels})
        bucket_stats = calculate_cumulative_stats(df, 'score', 'label', n_buckets=100)
        self._raw_data = bucket_stats

        total_bads = bucket_stats['bads'].sum()
        total_goods = bucket_stats['goods'].sum()
        total_count = bucket_stats['count'].sum()
        overall_bad_rate = total_bads / total_count

        rows = []
        for pct in cutoff_percentiles:
            target_pct = pct / 100
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

    def plot(self, **kwargs) -> 'CaptureRateTablePl':
        """Generate interactive capture rate table."""
        if self._table_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        colors = self.config.colors

        # Format columns for display
        header_values = [
            '<b>Population</b>',
            '<b>Score<br>Cutoff</b>',
            '<b>N<br>Approved</b>',
            '<b>Bads<br>Captured</b>',
            '<b>Goods<br>Captured</b>',
            '<b>Bad<br>Rate</b>',
            '<b>Lift</b>'
        ]

        cell_values = [
            [f"{x}%" for x in self._table_data['population_pct']],
            [f"{x:.0f}" for x in self._table_data['score_cutoff']],
            [f"{x:,.0f}" for x in self._table_data['n_approved']],
            [f"{x:.1f}%" for x in self._table_data['bads_captured_pct']],
            [f"{x:.1f}%" for x in self._table_data['goods_captured_pct']],
            [f"{x:.2f}%" for x in self._table_data['approval_bad_rate']],
            [f"{x:.2f}x" for x in self._table_data['lift']]
        ]

        # Color code bad rate column (red for high, green for low)
        bad_rates = self._table_data['approval_bad_rate'].values
        max_br = bad_rates.max()
        min_br = bad_rates.min()

        def get_color(val):
            if max_br == min_br:
                return 'white'
            # Normalize to 0-1
            norm = (val - min_br) / (max_br - min_br)
            # Interpolate between green and red
            r = int(129 + (242 - 129) * norm)
            g = int(178 - (178 - 84) * norm)
            b = int(154 - (154 - 91) * norm)
            return f'rgba({r},{g},{b},0.3)'

        fill_colors = [
            ['white'] * len(self._table_data),  # population
            ['white'] * len(self._table_data),  # score
            ['white'] * len(self._table_data),  # n_approved
            ['white'] * len(self._table_data),  # bads captured
            ['white'] * len(self._table_data),  # goods captured
            [get_color(x) for x in bad_rates],   # bad rate - colored
            ['white'] * len(self._table_data),  # lift
        ]

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=header_values,
                fill_color=colors.primary,
                font=dict(color='white', size=13),
                align='center',
                height=40
            ),
            cells=dict(
                values=cell_values,
                fill_color=fill_colors,
                font=dict(size=12),
                align='center',
                height=35
            )
        )])

        layout = self._get_layout_defaults("Capture Rate Analysis")
        layout['height'] = 450
        fig.update_layout(**layout)

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get table data for export."""
        if self._table_data is None:
            return {}
        return {'capture_rates': self._table_data.to_dict(orient='records')}

    def to_dataframe(self) -> pd.DataFrame:
        """Get data as DataFrame."""
        if self._table_data is None:
            raise RuntimeError("Must call calculate() first")
        return self._table_data.copy()
