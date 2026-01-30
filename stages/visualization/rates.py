"""
Approval Rate / Bad Rate (AR/BR) visualizations.

Includes:
- AR/BR curve showing trade-off
- AR/BR table with detailed metrics
- Cutoff analysis for decision making
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from stages.visualization.base import (
    BaseVisualization,
    VisualizationConfig,
    calculate_cumulative_stats,
    format_percentage,
    format_number,
)


@dataclass
class CutoffMetrics:
    """Metrics at a specific score cutoff."""
    score_cutoff: float
    approval_rate: float
    bad_rate: float
    n_approved: int
    n_rejected: int
    n_bads_approved: int
    n_goods_rejected: int
    expected_loss_rate: float  # assuming all approved bads default


class ARBRCurve(BaseVisualization):
    """
    Approval Rate vs Bad Rate curve.

    Shows the fundamental trade-off in credit decisions:
    - Approve more -> Higher bad rate (more defaults)
    - Approve less -> Lower bad rate but less business

    This curve helps find optimal cutoff points.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._curve_data: Dict[str, pd.DataFrame] = {}

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        dataset_name: str = "Model"
    ) -> 'ARBRCurve':
        """
        Calculate AR/BR curve data.

        Args:
            scores: Array of scores (higher = lower risk)
            labels: Binary labels (0=good, 1=bad)
            dataset_name: Name of this dataset

        Returns:
            self for method chaining
        """
        df = pd.DataFrame({
            'score': scores,
            'label': labels
        })

        # sort by score descending
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        # calculate at many points for smooth curve
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

    def plot(self, show_reference: bool = True, **kwargs) -> 'ARBRCurve':
        """
        Generate AR/BR curve plot.

        Args:
            show_reference: If True, show overall bad rate line
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.config.figsize)
        colors = self.config.colors

        for idx, (name, data) in enumerate(self._curve_data.items()):
            color = colors.palette[idx % len(colors.palette)]

            # AR on x-axis, BR on y-axis
            ax.plot(
                data['approval_rate'] * 100,
                data['bad_rate'] * 100,
                '-',
                color=color,
                lw=2,
                label=name
            )

            # mark some key points
            for ar_target in [0.5, 0.7, 0.9]:
                closest_idx = (data['approval_rate'] - ar_target).abs().argmin()
                point = data.iloc[closest_idx]
                ax.plot(
                    point['approval_rate'] * 100,
                    point['bad_rate'] * 100,
                    'o',
                    color=color,
                    markersize=8
                )
                ax.annotate(
                    f"{point['bad_rate']*100:.1f}%",
                    xy=(point['approval_rate'] * 100, point['bad_rate'] * 100),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=self.config.tick_fontsize
                )

        if show_reference:
            # overall bad rate line
            first_data = list(self._curve_data.values())[0]
            overall_br = first_data['n_bads_approved'].iloc[-1] / first_data['n_approved'].iloc[-1]
            ax.axhline(
                y=overall_br * 100,
                color='gray',
                linestyle='--',
                lw=1.5,
                alpha=0.7,
                label=f'Overall BR ({overall_br*100:.1f}%)'
            )

        ax.set_xlabel('Approval Rate (%)', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Bad Rate (%)', fontsize=self.config.label_fontsize)
        ax.set_title('Approval Rate vs Bad Rate Trade-off', fontsize=self.config.title_fontsize)
        ax.legend(loc='best', fontsize=self.config.legend_fontsize)

        ax.set_xlim([0, 105])
        ax.set_ylim([0, ax.get_ylim()[1] * 1.1])

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get curve data for export."""
        result = {}
        for name, data in self._curve_data.items():
            result[name] = data.to_dict(orient='records')
        return result

    def get_bad_rate_at_ar(self, approval_rate: float, dataset_name: str = "Model") -> float:
        """
        Get bad rate at given approval rate.

        Args:
            approval_rate: Target approval rate (0 to 1)
            dataset_name: Name of dataset

        Returns:
            Bad rate at that approval level
        """
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
    ) -> CutoffMetrics:
        """
        Find score cutoff to achieve target bad rate or approval rate.

        Args:
            target_bad_rate: Target bad rate (0 to 1)
            target_approval_rate: Target approval rate (0 to 1)
            dataset_name: Name of dataset

        Returns:
            CutoffMetrics at the optimal point
        """
        if dataset_name not in self._curve_data:
            raise KeyError(f"No data for: {dataset_name}")

        data = self._curve_data[dataset_name]

        if target_bad_rate is not None:
            # find lowest AR that achieves target BR
            eligible = data[data['bad_rate'] <= target_bad_rate]
            if len(eligible) == 0:
                # cant achieve target, return strictest cutoff
                point = data.iloc[0]
            else:
                point = eligible.iloc[-1]  # highest AR among eligible
        elif target_approval_rate is not None:
            # find closest AR
            idx = (data['approval_rate'] - target_approval_rate).abs().argmin()
            point = data.iloc[idx]
        else:
            raise ValueError("Must specify either target_bad_rate or target_approval_rate")

        total_count = data['n_approved'].iloc[-1]
        n_rejected = total_count - point['n_approved']
        n_goods_rejected = n_rejected - (data['n_bads_approved'].iloc[-1] - point['n_bads_approved'])

        return CutoffMetrics(
            score_cutoff=point['score_cutoff'],
            approval_rate=point['approval_rate'],
            bad_rate=point['bad_rate'],
            n_approved=int(point['n_approved']),
            n_rejected=int(n_rejected),
            n_bads_approved=int(point['n_bads_approved']),
            n_goods_rejected=int(max(0, n_goods_rejected)),
            expected_loss_rate=point['bad_rate']
        )


class ARBRTable(BaseVisualization):
    """
    Detailed AR/BR analysis table.

    Shows metrics at various score cutoffs for decision making.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._table_data: Optional[pd.DataFrame] = None

    def calculate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        score_cutoffs: Optional[List[float]] = None,
        n_buckets: int = 10
    ) -> 'ARBRTable':
        """
        Calculate AR/BR table.

        Args:
            scores: Array of scores
            labels: Binary labels (0=good, 1=bad)
            score_cutoffs: Specific cutoffs to evaluate. If None, uses bucket boundaries.
            n_buckets: Number of buckets if cutoffs not specified

        Returns:
            self for method chaining
        """
        df = pd.DataFrame({
            'score': scores,
            'label': labels
        })
        df = df.sort_values('score', ascending=False).reset_index(drop=True)

        total_count = len(df)
        total_bads = df['label'].sum()
        total_goods = total_count - total_bads

        if score_cutoffs is None:
            # use decile boundaries
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

            # capture rates
            bads_captured = n_bads_rejected / total_bads if total_bads > 0 else 0  # bads we correctly rejected
            goods_lost = n_goods_rejected / total_goods if total_goods > 0 else 0  # goods we wrongly rejected

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

    def plot(self, **kwargs) -> 'ARBRTable':
        """
        Generate AR/BR table as figure.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        if self._table_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')

        # format for display
        display_df = self._table_data.copy()
        display_df['score_cutoff'] = display_df['score_cutoff'].apply(lambda x: f"{x:.0f}")
        display_df['n_approved'] = display_df['n_approved'].apply(lambda x: f"{x:,}")
        display_df['n_rejected'] = display_df['n_rejected'].apply(lambda x: f"{x:,}")
        display_df['approval_rate'] = display_df['approval_rate'].apply(lambda x: f"{x*100:.1f}%")
        display_df['approved_bad_rate'] = display_df['approved_bad_rate'].apply(lambda x: f"{x*100:.2f}%")
        display_df['rejected_bad_rate'] = display_df['rejected_bad_rate'].apply(lambda x: f"{x*100:.2f}%")
        display_df['bads_rejected_pct'] = display_df['bads_rejected_pct'].apply(lambda x: f"{x*100:.1f}%")
        display_df['goods_rejected_pct'] = display_df['goods_rejected_pct'].apply(lambda x: f"{x*100:.1f}%")

        columns = ['score_cutoff', 'n_approved', 'n_rejected', 'approval_rate',
                   'approved_bad_rate', 'rejected_bad_rate', 'bads_rejected_pct', 'goods_rejected_pct']
        col_labels = ['Score\nCutoff', 'N\nApproved', 'N\nRejected', 'Approval\nRate',
                      'Approved\nBad Rate', 'Rejected\nBad Rate', 'Bads\nRejected', 'Goods\nRejected']

        table = ax.table(
            cellText=display_df[columns].values,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(self.config.tick_fontsize)
        table.scale(1.2, 1.5)

        # style header
        for i in range(len(col_labels)):
            table[(0, i)].set_facecolor(self.config.colors.primary)
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        ax.set_title('Approval Rate / Bad Rate Analysis', fontsize=self.config.title_fontsize, pad=20)

        if self.config.tight_layout:
            fig.tight_layout()

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


class SwapAnalysis(BaseVisualization):
    """
    Swap set analysis visualization.

    Compares current model vs champion model to show:
    - Applications approved by new but rejected by old (and vice versa)
    - Bad rates in swap-in and swap-out populations
    """

    def __init__(self, config: Optional[VisualizationConfig] = None):
        super().__init__(config)
        self._swap_data: Optional[Dict] = None

    def calculate(
        self,
        scores_new: np.ndarray,
        scores_old: np.ndarray,
        labels: np.ndarray,
        cutoff_new: float,
        cutoff_old: float
    ) -> 'SwapAnalysis':
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

        # approval decisions
        approved_new = scores_new >= cutoff_new
        approved_old = scores_old >= cutoff_old

        # swap sets
        swap_in = approved_new & ~approved_old  # new approves, old rejects
        swap_out = ~approved_new & approved_old  # new rejects, old approves
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

    def plot(self, **kwargs) -> 'SwapAnalysis':
        """
        Generate swap analysis visualization.

        Args:
            **kwargs: Additional matplotlib arguments

        Returns:
            self for method chaining
        """
        import matplotlib.pyplot as plt

        if self._swap_data is None:
            raise RuntimeError("Must call calculate() before plot()")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        colors = self.config.colors

        # Left: Venn-like diagram showing swap sets
        ax1 = axes[0]

        labels_pie = ['Both Approve', 'Both Reject', 'Swap In\n(New only)', 'Swap Out\n(Old only)']
        sizes = [
            self._swap_data['summary']['both_approve'],
            self._swap_data['summary']['both_reject'],
            self._swap_data['summary']['swap_in'],
            self._swap_data['summary']['swap_out']
        ]
        colors_pie = [colors.success, colors.info, colors.warning, colors.danger]

        wedges, texts, autotexts = ax1.pie(
            sizes,
            labels=labels_pie,
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90
        )
        ax1.set_title('Population Distribution', fontsize=self.config.title_fontsize)

        # Right: Bar chart comparing bad rates
        ax2 = axes[1]

        categories = ['Swap In', 'Swap Out', 'Both Approve', 'New Model\nOverall', 'Old Model\nOverall']
        bad_rates = [
            self._swap_data['swap_in']['bad_rate'] * 100,
            self._swap_data['swap_out']['bad_rate'] * 100,
            self._swap_data['both_approve']['bad_rate'] * 100,
            self._swap_data['new_model']['bad_rate'] * 100,
            self._swap_data['old_model']['bad_rate'] * 100
        ]
        bar_colors = [colors.warning, colors.danger, colors.success, colors.primary, colors.secondary]

        bars = ax2.bar(categories, bad_rates, color=bar_colors, alpha=0.8)

        # add value labels
        for bar, rate in zip(bars, bad_rates):
            ax2.annotate(
                f'{rate:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                fontsize=self.config.tick_fontsize
            )

        ax2.set_ylabel('Bad Rate (%)', fontsize=self.config.label_fontsize)
        ax2.set_title('Bad Rates by Segment', fontsize=self.config.title_fontsize)

        if self.config.show_grid:
            ax2.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Swap Analysis: New Model vs Old Model', fontsize=self.config.title_fontsize + 2)

        if self.config.tight_layout:
            fig.tight_layout()

        self._figure = fig
        return self

    def get_data(self) -> Dict[str, Any]:
        """Get swap analysis data for export."""
        return self._swap_data or {}

    def is_improvement(self) -> bool:
        """
        Check if new model improves over old model.

        Returns True if swap-in bad rate < swap-out bad rate,
        meaning the new model is approving better quality
        applications that old model rejected.
        """
        if self._swap_data is None:
            raise RuntimeError("Must call calculate() first")

        swap_in_br = self._swap_data['swap_in']['bad_rate']
        swap_out_br = self._swap_data['swap_out']['bad_rate']

        # new model is better if what it adds (swap_in) has lower BR
        # than what it removes (swap_out)
        return swap_in_br < swap_out_br
