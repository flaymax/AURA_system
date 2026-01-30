"""
Base classes and utilities for model evaluation.

Provides foundational components for:
- Performance metrics calculation
- Time-based analysis
- Segment-level evaluation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    AUC = "auc"
    GINI = "gini"
    KS = "ks"
    BAD_RATE = "bad_rate"
    ACCURACY = "accuracy"


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""
    auc: float
    gini: float
    ks: float
    n_samples: int
    n_bads: int
    bad_rate: float
    # optional detailed metrics
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None

    @classmethod
    def calculate(
        cls,
        y_true: np.ndarray,
        y_score: np.ndarray,
        store_curves: bool = False
    ) -> 'PerformanceMetrics':
        """
        Calculate all performance metrics.

        Args:
            y_true: True binary labels
            y_score: Predicted probabilities or scores
            store_curves: If True, store ROC curve data

        Returns:
            PerformanceMetrics instance
        """
        n_samples = len(y_true)
        n_bads = int(y_true.sum())
        bad_rate = n_bads / n_samples if n_samples > 0 else 0

        # handle edge cases
        if n_bads == 0 or n_bads == n_samples:
            return cls(
                auc=0.5,
                gini=0.0,
                ks=0.0,
                n_samples=n_samples,
                n_bads=n_bads,
                bad_rate=bad_rate
            )

        # AUC and Gini
        auc = roc_auc_score(y_true, y_score)
        gini = 2 * auc - 1

        # KS statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        ks = max(tpr - fpr)

        return cls(
            auc=auc,
            gini=gini,
            ks=ks,
            n_samples=n_samples,
            n_bads=n_bads,
            bad_rate=bad_rate,
            fpr=fpr if store_curves else None,
            tpr=tpr if store_curves else None,
            thresholds=thresholds if store_curves else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'auc': round(self.auc, 4),
            'gini': round(self.gini, 4),
            'ks': round(self.ks, 4),
            'n_samples': self.n_samples,
            'n_bads': self.n_bads,
            'bad_rate': round(self.bad_rate, 4)
        }


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    # time column for stability analysis
    time_column: Optional[str] = None
    # segment columns for subpopulation analysis
    segment_columns: List[str] = field(default_factory=list)
    # sample type column (0=train, 1=test, 2=valid)
    sample_type_column: str = "sample_type"
    # target column
    target_column: str = "target"
    # minimum samples for valid metric calculation
    min_samples: int = 100
    # score column name (if using scores instead of probabilities)
    score_column: Optional[str] = None


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.

    Provides common interface for different types of evaluation:
    - Time-based stability
    - Segment-level performance
    - Model comparison
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize evaluator.

        Args:
            config: EvaluationConfig with column names and thresholds
        """
        self.config = config or EvaluationConfig()
        self._results: Dict[str, Any] = {}

    @abstractmethod
    def evaluate(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation and return results.

        Args:
            data: DataFrame with features and target
            predictions: Predicted probabilities or scores
            **kwargs: Additional arguments

        Returns:
            Dictionary with evaluation results
        """
        pass

    @abstractmethod
    def get_summary(self) -> pd.DataFrame:
        """
        Get evaluation summary as DataFrame.

        Returns:
            Summary table
        """
        pass

    def get_results(self) -> Dict[str, Any]:
        """Get raw evaluation results."""
        return self._results.copy()


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index (PSI).

    PSI measures distribution shift between two populations.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change
    - PSI >= 0.25: Significant change

    Args:
        expected: Expected/baseline distribution (e.g., training data)
        actual: Actual/new distribution (e.g., validation data)
        n_bins: Number of bins for discretization
        eps: Small value to avoid log(0)

    Returns:
        Tuple of (PSI value, detailed breakdown DataFrame)
    """
    # create bins based on expected distribution
    _, bin_edges = np.histogram(expected, bins=n_bins)

    # calculate proportions in each bin
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    expected_pct = expected_counts / len(expected) + eps
    actual_pct = actual_counts / len(actual) + eps

    # PSI calculation
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = np.sum(psi_values)

    # build breakdown table
    breakdown = pd.DataFrame({
        'bin_min': bin_edges[:-1],
        'bin_max': bin_edges[1:],
        'expected_count': expected_counts,
        'actual_count': actual_counts,
        'expected_pct': expected_pct - eps,
        'actual_pct': actual_pct - eps,
        'psi_contribution': psi_values
    })

    return psi_total, breakdown


def calculate_csi(
    expected_rate: float,
    actual_rate: float,
    eps: float = 1e-6
) -> float:
    """
    Calculate Characteristic Stability Index (CSI).

    CSI measures change in a rate metric (e.g., bad rate) over time.

    Args:
        expected_rate: Expected/baseline rate
        actual_rate: Actual/current rate
        eps: Small value to avoid division by zero

    Returns:
        CSI value
    """
    expected_rate = max(expected_rate, eps)
    actual_rate = max(actual_rate, eps)

    csi = (actual_rate - expected_rate) * np.log(actual_rate / expected_rate)
    return csi
