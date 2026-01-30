"""
Model comparison utilities for challenger/champion analysis.

Compares multiple models on the same data:
- Performance comparison (AUC, Gini, KS)
- Rank correlation analysis
- Swap set analysis
- Statistical significance testing
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from models.base import (
    BaseEvaluator,
    EvaluationConfig,
    PerformanceMetrics,
)


logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonResult:
    """Result of comparing two models."""
    model_a_name: str
    model_b_name: str
    # performance metrics
    model_a_metrics: PerformanceMetrics
    model_b_metrics: PerformanceMetrics
    # comparison metrics
    gini_difference: float
    auc_difference: float
    rank_correlation: float  # Spearman correlation of predictions
    # statistical test
    is_significant: bool
    p_value: float
    # swap analysis
    swap_in_count: int  # A approves, B rejects
    swap_out_count: int  # B approves, A rejects
    swap_in_bad_rate: float
    swap_out_bad_rate: float
    winner: str  # 'model_a', 'model_b', or 'tie'


@dataclass
class MultiModelComparison:
    """Comparison of multiple models."""
    model_names: List[str]
    # performance by model
    metrics: Dict[str, PerformanceMetrics]
    # pairwise comparisons
    pairwise: List[ModelComparisonResult]
    # ranking
    ranking: List[Tuple[str, float]]  # [(model_name, gini), ...]
    best_model: str


class ModelComparator(BaseEvaluator):
    """
    Comparator for evaluating multiple models on the same data.

    Provides comprehensive comparison including:
    - Head-to-head performance metrics
    - Statistical significance testing
    - Rank correlation analysis
    - Swap set analysis for decision impact

    Example usage:
        comparator = ModelComparator(config)
        result = comparator.compare_two(
            data, y_true,
            predictions_a, predictions_b,
            name_a="Champion", name_b="Challenger"
        )
        summary = comparator.get_summary()
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize ModelComparator.

        Args:
            config: EvaluationConfig instance
        """
        super().__init__(config)
        self._comparison_result: Optional[ModelComparisonResult] = None
        self._multi_comparison: Optional[MultiModelComparison] = None

    def evaluate(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Basic evaluation interface (not typically used directly).

        Use compare_two() or compare_multiple() instead.
        """
        target_col = self.config.target_column
        y_true = data[target_col].values
        metrics = PerformanceMetrics.calculate(y_true, predictions)
        return {'metrics': metrics.to_dict()}

    def compare_two(
        self,
        data: pd.DataFrame,
        y_true: np.ndarray,
        predictions_a: np.ndarray,
        predictions_b: np.ndarray,
        name_a: str = "Model A",
        name_b: str = "Model B",
        cutoff_a: Optional[float] = None,
        cutoff_b: Optional[float] = None
    ) -> ModelComparisonResult:
        """
        Compare two models head-to-head.

        Args:
            data: DataFrame with features (for reference)
            y_true: True binary labels
            predictions_a: Predictions from model A
            predictions_b: Predictions from model B
            name_a: Name for model A
            name_b: Name for model B
            cutoff_a: Optional decision cutoff for model A (for swap analysis)
            cutoff_b: Optional decision cutoff for model B

        Returns:
            ModelComparisonResult with full comparison
        """
        # calculate performance metrics
        metrics_a = PerformanceMetrics.calculate(y_true, predictions_a)
        metrics_b = PerformanceMetrics.calculate(y_true, predictions_b)

        # differences
        gini_diff = metrics_a.gini - metrics_b.gini
        auc_diff = metrics_a.auc - metrics_b.auc

        # rank correlation (how similar are the orderings?)
        rank_corr, _ = stats.spearmanr(predictions_a, predictions_b)

        # statistical significance test (DeLong test approximation)
        is_sig, p_val = self._test_significance(y_true, predictions_a, predictions_b)

        # swap analysis (if cutoffs provided)
        if cutoff_a is not None and cutoff_b is not None:
            swap_in, swap_out, swap_in_br, swap_out_br = self._swap_analysis(
                y_true, predictions_a, predictions_b, cutoff_a, cutoff_b
            )
        else:
            # use median as default cutoff
            median_a = np.median(predictions_a)
            median_b = np.median(predictions_b)
            swap_in, swap_out, swap_in_br, swap_out_br = self._swap_analysis(
                y_true, predictions_a, predictions_b, median_a, median_b
            )

        # determine winner
        if is_sig:
            winner = name_a if gini_diff > 0 else name_b
        else:
            winner = 'tie'

        result = ModelComparisonResult(
            model_a_name=name_a,
            model_b_name=name_b,
            model_a_metrics=metrics_a,
            model_b_metrics=metrics_b,
            gini_difference=gini_diff,
            auc_difference=auc_diff,
            rank_correlation=rank_corr,
            is_significant=is_sig,
            p_value=p_val,
            swap_in_count=swap_in,
            swap_out_count=swap_out,
            swap_in_bad_rate=swap_in_br,
            swap_out_bad_rate=swap_out_br,
            winner=winner
        )

        self._comparison_result = result
        self._results = self._result_to_dict(result)

        return result

    def compare_multiple(
        self,
        y_true: np.ndarray,
        predictions_dict: Dict[str, np.ndarray]
    ) -> MultiModelComparison:
        """
        Compare multiple models.

        Args:
            y_true: True binary labels
            predictions_dict: Dictionary mapping model names to predictions

        Returns:
            MultiModelComparison with all pairwise comparisons
        """
        model_names = list(predictions_dict.keys())

        # calculate metrics for each model
        metrics = {}
        for name, preds in predictions_dict.items():
            metrics[name] = PerformanceMetrics.calculate(y_true, preds)

        # pairwise comparisons
        pairwise = []
        for i, name_a in enumerate(model_names):
            for name_b in model_names[i+1:]:
                result = self.compare_two(
                    pd.DataFrame(),  # empty df, not used
                    y_true,
                    predictions_dict[name_a],
                    predictions_dict[name_b],
                    name_a=name_a,
                    name_b=name_b
                )
                pairwise.append(result)

        # rank models by Gini
        ranking = sorted(
            [(name, m.gini) for name, m in metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        best_model = ranking[0][0]

        multi = MultiModelComparison(
            model_names=model_names,
            metrics=metrics,
            pairwise=pairwise,
            ranking=ranking,
            best_model=best_model
        )

        self._multi_comparison = multi
        return multi

    def _test_significance(
        self,
        y_true: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Test if AUC difference is statistically significant.

        Uses bootstrap approach for simplicity (approximation of DeLong test).

        Args:
            y_true: True labels
            pred_a: Predictions from model A
            pred_b: Predictions from model B
            alpha: Significance level

        Returns:
            Tuple of (is_significant, p_value)
        """
        n_bootstrap = 1000
        auc_diffs = []

        n = len(y_true)

        for _ in range(n_bootstrap):
            # bootstrap sample
            idx = np.random.choice(n, size=n, replace=True)
            y_boot = y_true[idx]
            a_boot = pred_a[idx]
            b_boot = pred_b[idx]

            # need both classes in bootstrap sample
            if y_boot.sum() == 0 or y_boot.sum() == len(y_boot):
                continue

            auc_a = roc_auc_score(y_boot, a_boot)
            auc_b = roc_auc_score(y_boot, b_boot)
            auc_diffs.append(auc_a - auc_b)

        if len(auc_diffs) < 100:
            # not enough valid bootstrap samples
            return False, 1.0

        auc_diffs = np.array(auc_diffs)

        # two-tailed test: is difference significantly different from 0?
        # p-value = proportion of bootstrap diffs on wrong side of 0
        observed_diff = roc_auc_score(y_true, pred_a) - roc_auc_score(y_true, pred_b)

        if observed_diff > 0:
            p_value = np.mean(auc_diffs <= 0) * 2  # two-tailed
        else:
            p_value = np.mean(auc_diffs >= 0) * 2

        p_value = min(p_value, 1.0)

        return p_value < alpha, p_value

    def _swap_analysis(
        self,
        y_true: np.ndarray,
        pred_a: np.ndarray,
        pred_b: np.ndarray,
        cutoff_a: float,
        cutoff_b: float
    ) -> Tuple[int, int, float, float]:
        """
        Perform swap set analysis.

        Args:
            y_true: True labels
            pred_a: Predictions from model A
            pred_b: Predictions from model B
            cutoff_a: Decision cutoff for A
            cutoff_b: Decision cutoff for B

        Returns:
            Tuple of (swap_in_count, swap_out_count, swap_in_br, swap_out_br)
        """
        approve_a = pred_a >= cutoff_a
        approve_b = pred_b >= cutoff_b

        # swap in: A approves, B rejects
        swap_in_mask = approve_a & ~approve_b
        swap_in_count = swap_in_mask.sum()
        swap_in_br = y_true[swap_in_mask].mean() if swap_in_count > 0 else 0.0

        # swap out: B approves, A rejects
        swap_out_mask = ~approve_a & approve_b
        swap_out_count = swap_out_mask.sum()
        swap_out_br = y_true[swap_out_mask].mean() if swap_out_count > 0 else 0.0

        return swap_in_count, swap_out_count, swap_in_br, swap_out_br

    def _result_to_dict(self, result: ModelComparisonResult) -> Dict[str, Any]:
        """Convert comparison result to dictionary."""
        return {
            'model_a': result.model_a_name,
            'model_b': result.model_b_name,
            'model_a_gini': result.model_a_metrics.gini,
            'model_b_gini': result.model_b_metrics.gini,
            'gini_difference': result.gini_difference,
            'rank_correlation': result.rank_correlation,
            'is_significant': result.is_significant,
            'p_value': result.p_value,
            'winner': result.winner,
            'swap_in_count': result.swap_in_count,
            'swap_out_count': result.swap_out_count,
            'swap_in_bad_rate': result.swap_in_bad_rate,
            'swap_out_bad_rate': result.swap_out_bad_rate
        }

    def get_summary(self) -> pd.DataFrame:
        """
        Get comparison summary as DataFrame.

        Returns:
            Summary table
        """
        if self._multi_comparison is not None:
            # multiple model summary
            rows = []
            for name, metrics in self._multi_comparison.metrics.items():
                rows.append({
                    'model': name,
                    'auc': round(metrics.auc, 4),
                    'gini': round(metrics.gini, 4),
                    'ks': round(metrics.ks, 4),
                    'n_samples': metrics.n_samples
                })
            df = pd.DataFrame(rows)
            return df.sort_values('gini', ascending=False)

        elif self._comparison_result is not None:
            # two model comparison
            r = self._comparison_result
            return pd.DataFrame([
                {
                    'model': r.model_a_name,
                    'gini': round(r.model_a_metrics.gini, 4),
                    'auc': round(r.model_a_metrics.auc, 4),
                    'ks': round(r.model_a_metrics.ks, 4)
                },
                {
                    'model': r.model_b_name,
                    'gini': round(r.model_b_metrics.gini, 4),
                    'auc': round(r.model_b_metrics.auc, 4),
                    'ks': round(r.model_b_metrics.ks, 4)
                }
            ])

        return pd.DataFrame()

    def get_pairwise_summary(self) -> pd.DataFrame:
        """
        Get pairwise comparison summary (for multiple models).

        Returns:
            DataFrame with all pairwise comparisons
        """
        if self._multi_comparison is None:
            return pd.DataFrame()

        rows = []
        for pw in self._multi_comparison.pairwise:
            rows.append({
                'model_a': pw.model_a_name,
                'model_b': pw.model_b_name,
                'gini_a': round(pw.model_a_metrics.gini, 4),
                'gini_b': round(pw.model_b_metrics.gini, 4),
                'gini_diff': round(pw.gini_difference, 4),
                'rank_corr': round(pw.rank_correlation, 4),
                'significant': pw.is_significant,
                'p_value': round(pw.p_value, 4),
                'winner': pw.winner
            })

        return pd.DataFrame(rows)

    def to_visualization_data(self) -> Dict[str, Any]:
        """
        Prepare data for visualization components.

        Returns:
            Dictionary ready for comparison charts
        """
        if self._multi_comparison:
            return {
                'models': self._multi_comparison.model_names,
                'gini': [self._multi_comparison.metrics[m].gini for m in self._multi_comparison.model_names],
                'auc': [self._multi_comparison.metrics[m].auc for m in self._multi_comparison.model_names],
                'ks': [self._multi_comparison.metrics[m].ks for m in self._multi_comparison.model_names],
                'ranking': self._multi_comparison.ranking,
                'best_model': self._multi_comparison.best_model
            }
        elif self._comparison_result:
            r = self._comparison_result
            return {
                'models': [r.model_a_name, r.model_b_name],
                'gini': [r.model_a_metrics.gini, r.model_b_metrics.gini],
                'auc': [r.model_a_metrics.auc, r.model_b_metrics.auc],
                'winner': r.winner,
                'is_significant': r.is_significant
            }
        return {}
