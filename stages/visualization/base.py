"""
Base classes and utilities for visualization components.

Provides common functionality for all visualization types including:
- Base visualization class with save/show methods
- Color schemes and styling constants
- Data preparation utilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import json

import numpy as np
import pandas as pd


class OutputFormat(Enum):
    """Supported output formats for visualizations."""
    PNG = "png"
    SVG = "svg"
    PDF = "pdf"
    HTML = "html"  # for interactive plots
    JSON = "json"  # for data export


@dataclass
class ColorScheme:
    """Color scheme for consistent visualization styling."""
    primary: str = "#1f77b4"  # blue
    secondary: str = "#ff7f0e"  # orange
    success: str = "#2ca02c"  # green
    danger: str = "#d62728"  # red
    warning: str = "#ffbb78"  # light orange
    info: str = "#17becf"  # cyan
    # for multi-series plots
    palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ])
    # for heatmaps
    gradient_low: str = "#f7fbff"
    gradient_high: str = "#08306b"
    # background
    background: str = "#ffffff"
    grid: str = "#e5e5e5"


@dataclass
class VisualizationConfig:
    """Configuration for visualization output."""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 150
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    colors: ColorScheme = field(default_factory=ColorScheme)
    style: str = "seaborn-v0_8-whitegrid"  # matplotlib style
    show_grid: bool = True
    tight_layout: bool = True


class BaseVisualization(ABC):
    """
    Abstract base class for all visualizations.

    Provides common interface for creating, saving, and exporting
    visualization data. Subclasses implement specific plot types.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualization with config.

        Args:
            config: VisualizationConfig for styling. Uses defaults if None.
        """
        self.config = config or VisualizationConfig()
        self._figure = None
        self._data: Dict[str, Any] = {}

    @abstractmethod
    def plot(self, **kwargs) -> 'BaseVisualization':
        """
        Generate the visualization.

        Args:
            **kwargs: Plot-specific arguments

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        Get underlying data for the visualization.

        Returns:
            Dictionary with plot data (for JSON export or tables)
        """
        pass

    def save(
        self,
        path: str,
        format: Optional[OutputFormat] = None,
        **kwargs
    ) -> str:
        """
        Save visualization to file.

        Args:
            path: Output file path
            format: Output format. If None, infers from path extension.
            **kwargs: Additional save arguments

        Returns:
            Path to saved file
        """
        import matplotlib.pyplot as plt

        if format is None:
            # infer from extension
            ext = path.split('.')[-1].lower()
            format = OutputFormat(ext) if ext in [f.value for f in OutputFormat] else OutputFormat.PNG

        if format == OutputFormat.JSON:
            # export data as JSON
            data = self.get_data()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            if self._figure is None:
                raise RuntimeError("Must call plot() before save()")
            self._figure.savefig(
                path,
                format=format.value,
                dpi=kwargs.get('dpi', self.config.dpi),
                bbox_inches='tight'
            )

        return path

    def show(self):
        """Display the visualization interactively."""
        import matplotlib.pyplot as plt

        if self._figure is None:
            raise RuntimeError("Must call plot() before show()")
        plt.show()

    def to_html(self) -> str:
        """
        Convert visualization to HTML string.

        Returns:
            HTML representation of the plot
        """
        import io
        import base64

        if self._figure is None:
            raise RuntimeError("Must call plot() before to_html()")

        buf = io.BytesIO()
        self._figure.savefig(buf, format='png', dpi=self.config.dpi, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')

        return f'<img src="data:image/png;base64,{img_base64}" />'

    def close(self):
        """Close the figure to free memory."""
        import matplotlib.pyplot as plt

        if self._figure is not None:
            plt.close(self._figure)
            self._figure = None


def calculate_buckets(
    scores: np.ndarray,
    n_buckets: int = 10,
    method: str = "quantile"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate score buckets.

    Args:
        scores: Array of scores
        n_buckets: Number of buckets to create
        method: "quantile" for equal-frequency, "equal" for equal-width

    Returns:
        Tuple of (bucket_labels, bucket_edges)
    """
    if method == "quantile":
        # equal frequency buckets
        percentiles = np.linspace(0, 100, n_buckets + 1)
        edges = np.percentile(scores, percentiles)
        # ensure unique edges
        edges = np.unique(edges)
    else:
        # equal width buckets
        edges = np.linspace(scores.min(), scores.max(), n_buckets + 1)

    # assign to buckets
    labels = np.digitize(scores, edges[1:-1])

    return labels, edges


def calculate_cumulative_stats(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    n_buckets: int = 10
) -> pd.DataFrame:
    """
    Calculate cumulative statistics by score bucket.

    Used for gain charts, lift curves, and AR/BR analysis.

    Args:
        df: DataFrame with scores and target
        score_col: Name of score column
        target_col: Name of target column (1=bad, 0=good)
        n_buckets: Number of buckets

    Returns:
        DataFrame with bucket statistics
    """
    # create copy and sort by score (descending - higher score = lower risk)
    data = df[[score_col, target_col]].copy()
    data = data.sort_values(score_col, ascending=False).reset_index(drop=True)

    # assign buckets
    data['bucket'] = pd.qcut(
        data[score_col].rank(method='first'),
        q=n_buckets,
        labels=range(1, n_buckets + 1)
    )

    # aggregate by bucket
    bucket_stats = data.groupby('bucket').agg(
        count=(target_col, 'count'),
        bads=(target_col, 'sum'),
        min_score=(score_col, 'min'),
        max_score=(score_col, 'max'),
        mean_score=(score_col, 'mean')
    ).reset_index()

    bucket_stats['goods'] = bucket_stats['count'] - bucket_stats['bads']
    bucket_stats['bad_rate'] = bucket_stats['bads'] / bucket_stats['count']

    # cumulative stats
    total_bads = bucket_stats['bads'].sum()
    total_goods = bucket_stats['goods'].sum()
    total_count = bucket_stats['count'].sum()

    bucket_stats['cum_count'] = bucket_stats['count'].cumsum()
    bucket_stats['cum_bads'] = bucket_stats['bads'].cumsum()
    bucket_stats['cum_goods'] = bucket_stats['goods'].cumsum()

    bucket_stats['cum_count_pct'] = bucket_stats['cum_count'] / total_count
    bucket_stats['cum_bad_pct'] = bucket_stats['cum_bads'] / total_bads
    bucket_stats['cum_good_pct'] = bucket_stats['cum_goods'] / total_goods

    # lift = cum_bad_pct / cum_count_pct
    bucket_stats['lift'] = bucket_stats['cum_bad_pct'] / bucket_stats['cum_count_pct']

    # approval rate at this cutoff
    bucket_stats['approval_rate'] = bucket_stats['cum_count_pct']
    # bad rate if we approve up to this bucket
    bucket_stats['cumulative_bad_rate'] = bucket_stats['cum_bads'] / bucket_stats['cum_count']

    return bucket_stats


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 0) -> str:
    """Format number with thousand separators."""
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"
