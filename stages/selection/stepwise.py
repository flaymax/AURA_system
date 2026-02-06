"""
Stepwise feature selection stage using logistic regression.

Wraps the existing iterative_logistic_selection function from binary_logistic module.
Performs forward/backward stepwise selection based on likelihood ratio and Wald tests.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np

from core.base import PipelineStage, StageResult, SelectionConfig


@dataclass
class EarlyStoppingInfo:
    """Information about early stopping during stepwise selection."""
    enabled: bool
    triggered: bool
    reason: str  # "patience_exceeded", "no_improvement", "completed", "disabled"
    stopped_at_step: int
    best_step: int
    best_test_auc: float
    best_train_auc: float
    final_test_auc: float  # before early stopping
    features_removed: int  # how many features were removed by early stopping
    patience_used: int
    patience_limit: int
    min_improvement: float
    monitor_metric: str


@dataclass
class StepwiseFeatureInfo:
    """Information about a feature's journey through stepwise selection."""
    feature_name: str
    entrance_order: int  # order in which feature entered the model
    train_auc_at_entry: float
    test_auc_at_entry: float
    # did the feature survive to final model?
    in_final_model: bool
    removal_reason: Optional[str] = None  # if removed, why


@dataclass
class StepwiseInfo:
    """Comprehensive information about stepwise selection results."""
    n_input_features: int
    n_selected_features: int
    n_steps: int
    alpha: float  # significance level used
    # feature details
    feature_journey: List[StepwiseFeatureInfo] = field(default_factory=list)
    entrance_log: List[str] = field(default_factory=list)
    final_features: List[str] = field(default_factory=list)
    # AUC progression
    auc_progression: Optional[pd.DataFrame] = None
    final_train_auc: float = 0.0
    final_test_auc: float = 0.0
    # timing
    elapsed_seconds: float = 0.0
    # early stopping
    early_stopping: Optional[EarlyStoppingInfo] = None


class StepwiseSelectionStage(PipelineStage):
    """
    Stage for stepwise feature selection using logistic regression.

    This stage uses forward-backward stepwise selection with:
    - Likelihood ratio test for statistical significance
    - Wald test for coefficient significance
    - AUC improvement tracking

    Features are added if they improve the model significantly and
    removed if they become insignificant.
    """

    def __init__(self, config: Optional[SelectionConfig] = None):
        """
        Initialize StepwiseSelectionStage.

        Args:
            config: SelectionConfig with alpha level and max_steps.
                   If None, uses defaults (alpha=0.05, max_steps=200)
        """
        super().__init__(config or SelectionConfig())
        self._stepwise_info: Optional[StepwiseInfo] = None
        self._selected_features: List[str] = []
        self._is_fitted = False
        self._entrance_log: List[str] = []
        self._auc_at_entrance: Dict[str, Tuple[float, float]] = {}

    def _apply_early_stopping(
        self,
        auc_df: pd.DataFrame,
        entrance_log: List[str],
        final_vars: List[str]
    ) -> Tuple[List[str], EarlyStoppingInfo]:
        """
        Analyze AUC progression and apply early stopping if beneficial.

        Args:
            auc_df: DataFrame with AUC progression (num_features, train_auc, test_auc)
            entrance_log: Order of feature entrance
            final_vars: Features selected by stepwise

        Returns:
            Tuple of (adjusted_features, early_stopping_info)
        """
        # get early stopping config
        enabled = getattr(self.config, 'early_stopping_enabled', True)
        patience = getattr(self.config, 'patience', 5)
        min_improvement = getattr(self.config, 'min_improvement', 0.001)
        restore_best = getattr(self.config, 'restore_best', True)
        monitor_metric = getattr(self.config, 'monitor_metric', 'test_auc')

        if not enabled or len(auc_df) == 0:
            return final_vars, EarlyStoppingInfo(
                enabled=enabled,
                triggered=False,
                reason="disabled" if not enabled else "no_data",
                stopped_at_step=len(auc_df),
                best_step=len(auc_df),
                best_test_auc=auc_df['test_auc'].iloc[-1] if len(auc_df) > 0 else 0.0,
                best_train_auc=auc_df['train_auc'].iloc[-1] if len(auc_df) > 0 else 0.0,
                final_test_auc=auc_df['test_auc'].iloc[-1] if len(auc_df) > 0 else 0.0,
                features_removed=0,
                patience_used=0,
                patience_limit=patience,
                min_improvement=min_improvement,
                monitor_metric=monitor_metric
            )

        # find best step based on monitor metric
        metric_col = monitor_metric if monitor_metric in auc_df.columns else 'test_auc'
        metric_values = auc_df[metric_col].values

        best_step = 0
        best_value = metric_values[0]
        patience_counter = 0
        early_stop_step = len(metric_values)

        for i in range(1, len(metric_values)):
            improvement = metric_values[i] - best_value
            if improvement > min_improvement:
                best_value = metric_values[i]
                best_step = i
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    early_stop_step = i
                    break

        # determine if early stopping was triggered
        triggered = early_stop_step < len(metric_values) or best_step < len(metric_values) - 1
        reason = "completed"
        if patience_counter >= patience:
            reason = "patience_exceeded"
        elif best_step < len(metric_values) - 1:
            reason = "no_improvement"

        # calculate how many features to remove
        final_test_auc = auc_df['test_auc'].iloc[-1]
        best_test_auc = auc_df['test_auc'].iloc[best_step]
        best_train_auc = auc_df['train_auc'].iloc[best_step]

        # determine adjusted features
        adjusted_features = final_vars
        features_removed = 0

        if triggered and restore_best and best_step < len(auc_df) - 1:
            # restore to best step - keep only features up to that point
            best_num_features = int(auc_df['num_features'].iloc[best_step])
            # features entered in order, so keep only the first best_num_features
            features_in_order = [f for f in entrance_log if f in final_vars]
            if len(features_in_order) > best_num_features:
                adjusted_features = features_in_order[:best_num_features]
                features_removed = len(final_vars) - len(adjusted_features)

        early_stop_info = EarlyStoppingInfo(
            enabled=enabled,
            triggered=triggered,
            reason=reason,
            stopped_at_step=early_stop_step,
            best_step=best_step + 1,  # 1-indexed for display
            best_test_auc=best_test_auc,
            best_train_auc=best_train_auc,
            final_test_auc=final_test_auc,
            features_removed=features_removed,
            patience_used=patience_counter,
            patience_limit=patience,
            min_improvement=min_improvement,
            monitor_metric=monitor_metric
        )

        return adjusted_features, early_stop_info

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'StepwiseSelectionStage':
        """
        Fit stepwise selection to find optimal feature subset.

        Args:
            data: DataFrame with features and target (must have sample_type column)
            target: Name of target variable
            feature_columns: Features to consider. If None, auto-detects
            **kwargs: Additional arguments

        Returns:
            self for method chaining
        """
        # import here to avoid circular import
        from src.binary_logistic import iterative_logistic_selection

        start_time = time.time()

        # determine feature columns
        if feature_columns is None:
            exclude_cols = [target]
            if 'sample_type' in data.columns:
                exclude_cols.append('sample_type')
            feature_columns = [
                col for col in data.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
            ]

        n_input = len(feature_columns)

        # split data by sample_type (required by underlying function)
        if 'sample_type' not in data.columns:
            # create train/test split if not present
            n_samples = len(data)
            train_size = int(0.8 * n_samples)
            train_idx = np.random.choice(n_samples, size=train_size, replace=False)
            test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

            X_train = data.iloc[train_idx][feature_columns]
            y_train = data.iloc[train_idx][target]
            X_test = data.iloc[test_idx][feature_columns]
            y_test = data.iloc[test_idx][target]
        else:
            # use provided split
            train_mask = data['sample_type'] == 0
            test_mask = data['sample_type'] == 1

            X_train = data.loc[train_mask, feature_columns]
            y_train = data.loc[train_mask, target]
            X_test = data.loc[test_mask, feature_columns]
            y_test = data.loc[test_mask, target]

        # get config params
        alpha = getattr(self.config, 'alpha', 0.05)
        max_steps = getattr(self.config, 'max_steps', 200)

        # run stepwise selection
        entrance_log, auc_df, final_vars, auc_at_entrance = iterative_logistic_selection(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            alpha=alpha,
            max_steps=max_steps
        )

        # apply early stopping analysis
        adjusted_features, early_stop_info = self._apply_early_stopping(
            auc_df, entrance_log, final_vars
        )

        elapsed = time.time() - start_time

        # store results (use adjusted features if early stopping restored best)
        self._entrance_log = entrance_log
        self._auc_at_entrance = auc_at_entrance
        self._selected_features = adjusted_features

        # build detailed feature journey info
        feature_journey = []
        for idx, feat in enumerate(entrance_log):
            auc_train, auc_test = auc_at_entrance.get(feat, (0.0, 0.0))
            in_final = feat in adjusted_features

            removal_reason = None
            if not in_final:
                if feat in final_vars:
                    removal_reason = "Removed by early stopping"
                else:
                    removal_reason = "Removed due to insignificance"

            info = StepwiseFeatureInfo(
                feature_name=feat,
                entrance_order=idx + 1,
                train_auc_at_entry=auc_train,
                test_auc_at_entry=auc_test,
                in_final_model=in_final,
                removal_reason=removal_reason
            )
            feature_journey.append(info)

        # get final AUC from progression (use best if early stopping triggered)
        if early_stop_info.triggered and early_stop_info.features_removed > 0:
            final_train_auc = early_stop_info.best_train_auc
            final_test_auc = early_stop_info.best_test_auc
        else:
            final_train_auc = auc_df['train_auc'].iloc[-1] if len(auc_df) > 0 else 0.0
            final_test_auc = auc_df['test_auc'].iloc[-1] if len(auc_df) > 0 else 0.0

        self._stepwise_info = StepwiseInfo(
            n_input_features=n_input,
            n_selected_features=len(adjusted_features),
            n_steps=len(auc_df),
            alpha=alpha,
            feature_journey=feature_journey,
            entrance_log=entrance_log,
            final_features=adjusted_features,
            auc_progression=auc_df,
            final_train_auc=final_train_auc,
            final_test_auc=final_test_auc,
            elapsed_seconds=elapsed,
            early_stopping=early_stop_info
        )

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data by keeping only selected features.

        Args:
            data: DataFrame to transform
            **kwargs: Additional arguments (target name for preservation)

        Returns:
            DataFrame with only selected features
        """
        if not self._is_fitted:
            raise RuntimeError("StepwiseSelectionStage must be fitted before transform")

        cols_to_keep = [col for col in self._selected_features if col in data.columns]

        # preserve special columns
        target = kwargs.get('target')
        for col in data.columns:
            if col not in cols_to_keep:
                if col in ['sample_type'] or col == target:
                    cols_to_keep.append(col)

        return data[cols_to_keep]

    def fit_transform(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> StageResult:
        """
        Fit and transform in one step.

        Args:
            data: Input DataFrame
            target: Target variable name
            feature_columns: Features to consider
            **kwargs: Additional arguments

        Returns:
            StageResult with transformed data and logging
        """
        self.fit(data, target, feature_columns, **kwargs)
        transformed = self.transform(data, target=target, **kwargs)

        logs = self._build_logs()

        return StageResult(
            data=transformed,
            logs=logs,
            metadata={
                'selected_features': self._selected_features.copy(),
                'entrance_log': self._entrance_log.copy(),
                'final_train_auc': self._stepwise_info.final_train_auc,
                'final_test_auc': self._stepwise_info.final_test_auc
            }
        )

    def _build_logs(self) -> Dict[str, Any]:
        """Build comprehensive logs for visualization."""
        if self._stepwise_info is None:
            return {}

        info = self._stepwise_info

        # feature journey details
        journey_details = []
        for fj in info.feature_journey:
            journey_details.append({
                'feature': fj.feature_name,
                'entrance_order': fj.entrance_order,
                'train_auc_at_entry': round(fj.train_auc_at_entry, 4),
                'test_auc_at_entry': round(fj.test_auc_at_entry, 4),
                'in_final_model': fj.in_final_model,
                'removal_reason': fj.removal_reason
            })

        # AUC progression as list of dicts for JSON serializaiton
        auc_prog = []
        if info.auc_progression is not None:
            for _, row in info.auc_progression.iterrows():
                auc_prog.append({
                    'num_features': int(row['num_features']),
                    'train_auc': round(row['train_auc'], 4),
                    'test_auc': round(row['test_auc'], 4)
                })

        # early stopping info
        early_stop_log = None
        if info.early_stopping is not None:
            es = info.early_stopping
            early_stop_log = {
                'enabled': es.enabled,
                'triggered': es.triggered,
                'reason': es.reason,
                'stopped_at_step': es.stopped_at_step,
                'best_step': es.best_step,
                'best_test_auc': round(es.best_test_auc, 4),
                'best_train_auc': round(es.best_train_auc, 4),
                'final_test_auc_before_restore': round(es.final_test_auc, 4),
                'features_removed': es.features_removed,
                'patience_used': es.patience_used,
                'patience_limit': es.patience_limit,
                'min_improvement': es.min_improvement,
                'monitor_metric': es.monitor_metric
            }

        return {
            'stage_name': 'StepwiseSelectionStage',
            'summary': {
                'input_features': info.n_input_features,
                'output_features': info.n_selected_features,
                'n_steps': info.n_steps,
                'alpha': info.alpha,
                'final_train_auc': round(info.final_train_auc, 4),
                'final_test_auc': round(info.final_test_auc, 4),
                'elapsed_seconds': round(info.elapsed_seconds, 3)
            },
            'early_stopping': early_stop_log,
            'feature_journey': journey_details,
            'auc_progression': auc_prog,
            'entrance_log': info.entrance_log,
            'final_features': info.final_features
        }

    def get_stepwise_info(self) -> Optional[StepwiseInfo]:
        """Get detailed stepwise selection information."""
        return self._stepwise_info

    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self._selected_features.copy()

    def get_auc_at_entrance(self) -> Dict[str, Tuple[float, float]]:
        """Get AUC values (train, test) for each feature at entrance."""
        return self._auc_at_entrance.copy()

    def visualize_auc_path(self, output_path: Optional[str] = None):
        """
        Visualize AUC evolution along the feature selection path.

        Args:
            output_path: Path to save plot. If None, displays interactively.
        """
        from src.binary_logistic import visualize_auc_path

        if not self._is_fitted:
            raise RuntimeError("Must fit before visualizing")

        visualize_auc_path(
            entrance_log=self._entrance_log,
            final_vars=self._selected_features,
            auc_at_entrance=self._auc_at_entrance
        )

        # note: the underlying function shows plot, we could extend it to save

    def validate(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate stepwise selection results.

        Args:
            data: DataFrame to validate
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self._is_fitted:
            issues.append("StepwiseSelectionStage has not been fitted")
            return False, issues

        if len(self._selected_features) == 0:
            issues.append("No features were selected by stepwise selection")
            return False, issues

        # check for reasonable AUC
        if self._stepwise_info:
            if self._stepwise_info.final_test_auc < 0.5:
                issues.append(
                    f"Final test AUC is below 0.5: {self._stepwise_info.final_test_auc:.4f}"
                )

            # check for overfitting (large train-test gap)
            auc_gap = (
                self._stepwise_info.final_train_auc -
                self._stepwise_info.final_test_auc
            )
            if auc_gap > 0.1:
                issues.append(
                    f"Large train-test AUC gap detected: {auc_gap:.4f} "
                    f"(train={self._stepwise_info.final_train_auc:.4f}, "
                    f"test={self._stepwise_info.final_test_auc:.4f})"
                )

        return len(issues) == 0, issues
