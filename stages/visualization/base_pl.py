"""
Base classes and utilities for Plotly visualizations.

Provides common functionality for all Plotly-based visualization types:
- Base visualization class with save/show methods
- Warm, eye-pleasing color schemes
- Data preparation utilities
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import json

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class PlotlyColorScheme:
    """
    Warm, eye-pleasing color scheme for Plotly visualizations.

    Uses soft, modern colors that are easy on the eyes while
    maintaining good contrast for readability.
    """
    # Primary colors - warm tones
    primary: str = "#E07A5F"      # Terracotta red
    secondary: str = "#3D405B"    # Dark blue-gray
    success: str = "#81B29A"      # Sage green
    danger: str = "#F2545B"       # Soft coral red
    warning: str = "#F4A261"      # Sandy orange
    info: str = "#7EB8DA"         # Soft sky blue

    # Extended warm palette for multi-series
    palette: List[str] = field(default_factory=lambda: [
        "#E07A5F",  # Terracotta
        "#3D405B",  # Dark blue-gray
        "#81B29A",  # Sage green
        "#F4A261",  # Sandy orange
        "#7EB8DA",  # Sky blue
        "#9B8EA0",  # Dusty lavender
        "#D4A373",  # Warm tan
        "#E9C46A",  # Muted gold
        "#6D6875",  # Warm gray
        "#B5838D",  # Dusty rose
    ])

    # Background colors
    background: str = "#FEFAE0"   # Warm cream
    paper: str = "#FFFFFF"        # White for plot area
    grid: str = "#E8E4D9"         # Soft warm gray

    # Text colors
    text_primary: str = "#2D3436"   # Dark gray (not pure black)
    text_secondary: str = "#636E72"  # Medium gray

    # Gradient for heatmaps - warm tones
    gradient_scale: List[List] = field(default_factory=lambda: [
        [0.0, "#FEF9EF"],    # Very light cream
        [0.25, "#FFEAA7"],   # Soft yellow
        [0.5, "#F4A261"],    # Sandy orange
        [0.75, "#E07A5F"],   # Terracotta
        [1.0, "#9B2335"],    # Deep burgundy
    ])


@dataclass
class PlotlyConfig:
    """Configuration for Plotly visualizations."""
    width: int = 900
    height: int = 550
    title_fontsize: int = 18
    axis_fontsize: int = 14
    tick_fontsize: int = 12
    legend_fontsize: int = 12
    colors: PlotlyColorScheme = field(default_factory=PlotlyColorScheme)
    template: str = "plotly_white"
    show_logo: bool = False
    margin: Dict[str, int] = field(default_factory=lambda: {
        'l': 60, 'r': 40, 't': 60, 'b': 60
    })


class BasePlotlyVisualization(ABC):
    """
    Abstract base class for all Plotly visualizations.

    Provides common interface for creating, saving, and exporting
    interactive Plotly visualizations.
    """

    def __init__(self, config: Optional[PlotlyConfig] = None):
        """
        Initialize visualization with config.

        Args:
            config: PlotlyConfig for styling. Uses defaults if None.
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for this visualization. Install with: pip install plotly")

        self.config = config or PlotlyConfig()
        self._figure: Optional[go.Figure] = None
        self._data: Dict[str, Any] = {}

    def _get_layout_defaults(self, title: str = "") -> Dict:
        """Get default layout settings with warm styling."""
        return {
            'title': {
                'text': title,
                'font': {'size': self.config.title_fontsize, 'color': self.config.colors.text_primary},
                'x': 0.5,
                'xanchor': 'center'
            },
            'font': {
                'family': "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                'color': self.config.colors.text_primary
            },
            'plot_bgcolor': self.config.colors.paper,
            'paper_bgcolor': self.config.colors.paper,
            'width': self.config.width,
            'height': self.config.height,
            'margin': self.config.margin,
            'showlegend': True,
            'legend': {
                'font': {'size': self.config.legend_fontsize},
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': self.config.colors.grid,
                'borderwidth': 1
            },
            'xaxis': {
                'gridcolor': self.config.colors.grid,
                'linecolor': self.config.colors.grid,
                'tickfont': {'size': self.config.tick_fontsize},
                'title_font': {'size': self.config.axis_fontsize}
            },
            'yaxis': {
                'gridcolor': self.config.colors.grid,
                'linecolor': self.config.colors.grid,
                'tickfont': {'size': self.config.tick_fontsize},
                'title_font': {'size': self.config.axis_fontsize}
            }
        }

    @abstractmethod
    def plot(self, **kwargs) -> 'BasePlotlyVisualization':
        """
        Generate the visualization.

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def get_data(self) -> Dict[str, Any]:
        """
        Get underlying data for the visualization.

        Returns:
            Dictionary with plot data
        """
        pass

    def save(
        self,
        path: str,
        format: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Save visualization to file.

        Args:
            path: Output file path
            format: Output format (html, png, svg, pdf, json). Infers from extension if None.
            **kwargs: Additional arguments for plotly.io.write_*

        Returns:
            Path to saved file
        """
        if self._figure is None:
            raise RuntimeError("Must call plot() before save()")

        if format is None:
            format = path.split('.')[-1].lower()

        if format == 'json':
            data = self.get_data()
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == 'html':
            self._figure.write_html(
                path,
                include_plotlyjs=kwargs.get('include_plotlyjs', True),
                full_html=kwargs.get('full_html', True)
            )
        elif format in ['png', 'svg', 'pdf', 'jpeg', 'webp']:
            self._figure.write_image(
                path,
                format=format,
                scale=kwargs.get('scale', 2)
            )
        else:
            raise ValueError(f"Unsupported format: {format}")

        return path

    def show(self):
        """Display the visualization interactively."""
        if self._figure is None:
            raise RuntimeError("Must call plot() before show()")
        self._figure.show()

    def to_html(self, full_html: bool = False, include_plotlyjs: str = 'cdn') -> str:
        """
        Convert visualization to HTML string.

        Args:
            full_html: If True, returns complete HTML document
            include_plotlyjs: How to include plotly.js ('cdn', True, False)

        Returns:
            HTML representation
        """
        if self._figure is None:
            raise RuntimeError("Must call plot() before to_html()")

        return self._figure.to_html(
            full_html=full_html,
            include_plotlyjs=include_plotlyjs
        )

    def get_figure(self) -> Optional[go.Figure]:
        """Get the Plotly figure object."""
        return self._figure

    def update_layout(self, **kwargs) -> 'BasePlotlyVisualization':
        """
        Update figure layout after plotting.

        Args:
            **kwargs: Layout parameters to update

        Returns:
            self for method chaining
        """
        if self._figure is None:
            raise RuntimeError("Must call plot() before update_layout()")
        self._figure.update_layout(**kwargs)
        return self
