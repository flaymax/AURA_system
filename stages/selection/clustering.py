"""
Clustering stage for feature selection based on correlation analysis.

Wraps the existing ClusterAnalysis class to fit into the pipeline architecture.
Uses hierarchical clustering to group correlated features and selects
representative from each cluster.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import time

import pandas as pd
import numpy as np

from core.base import PipelineStage, StageResult, ClusteringConfig


class ClusterSelectionMethod(Enum):
    """Methods for selecting represantative feature from cluster."""
    MAX_TRAIN = "max_train"  # highest Gini on train set
    MAX_TEST = "max_test"  # highest Gini on test set
    MAX_VALID = "max_valid"  # highest Gini on validation set
    CLOSEST_TRAIN_TEST = "closest_train_test"  # smallest train-test Gini diff
    CENTER_CLUSTER = "center_cluster"  # closest to cluster centroid


@dataclass
class ClusterInfo:
    """Information about a single cluster of features."""
    cluster_id: int
    selected_feature: str
    all_features: List[str]
    n_features: int
    # Gini scores if available
    gini_train: Optional[float] = None
    gini_test: Optional[float] = None
    gini_valid: Optional[float] = None


@dataclass
class ClusteringInfo:
    """Comprehensive information about clustering results."""
    n_input_features: int
    n_clusters: int
    n_selected_features: int
    correlation_threshold: float
    selection_method: str
    # cluster details
    clusters: List[ClusterInfo] = field(default_factory=list)
    # mapping feature -> cluster_id (for dropped features tracking)
    feature_to_cluster: Dict[str, int] = field(default_factory=dict)
    # features that were dropped (not selected from clusters)
    dropped_features: List[str] = field(default_factory=list)
    # timing info
    elapsed_seconds: float = 0.0


class ClusteringStage(PipelineStage):
    """
    Stage for reducing correlated features using hierarchical clustering.

    This stage wraps the existing ClusterAnalysis class and provides:
    - Detailed logging of cluster composition
    - Tracking of which features were dropped and why
    - Consistent interface with other pipeline stages

    The clustering uses correlation-based distance (1 - |corr|) and
    selects one representative feature from each cluster.
    """

    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize ClusteringStage.

        Args:
            config: ClusteringConfig with threshold and selection method.
                   If None, uses default config with threshold=0.7
        """
        super().__init__(config or ClusteringConfig())
        self._cluster_analysis = None
        self._clustering_info: Optional[ClusteringInfo] = None
        self._selected_features: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]] = None,
        **kwargs
    ) -> 'ClusteringStage':
        """
        Fit the clustering model to identify correlated feature groups.

        Args:
            data: DataFrame with features and target
            target: Name of target variable column
            feature_columns: List of features to cluster. If None, uses all
                           numeric columns except target and sample_type
            **kwargs: Additional arguments (sample_type_col for train/test split)

        Returns:
            self for method chaining
        """
        # import here to avoid circular imports and keep it encapsulated
        from pipeline.clustering import ClusterAnalysis

        start_time = time.time()

        # determine which columns to use
        if feature_columns is None:
            # auto-detect: all numeric excpet target and sample_type
            exclude_cols = [target]
            if 'sample_type' in data.columns:
                exclude_cols.append('sample_type')
            feature_columns = [
                col for col in data.columns
                if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col])
            ]

        n_input = len(feature_columns)

        # check if sample_type column exists, its required by ClusterAnalysis
        if 'sample_type' not in data.columns:
            # create fake sample_type: 80% train, 10% test, 10% valid
            # this is a fallback if user didn't provide train/test split
            n_samples = len(data)
            sample_type = np.zeros(n_samples, dtype=int)
            # 10% test
            test_idx = np.random.choice(
                n_samples, size=int(0.1 * n_samples), replace=False
            )
            sample_type[test_idx] = 1
            # 10% valid from remaining
            remaining = np.setdiff1d(np.arange(n_samples), test_idx)
            valid_idx = np.random.choice(
                remaining, size=int(0.1 * n_samples), replace=False
            )
            sample_type[valid_idx] = 2

            data = data.copy()
            data['sample_type'] = sample_type

        # map selection method string to ClusterAnalysis format
        selection_type = self.config.selection_method
        if hasattr(selection_type, 'value'):
            selection_type = selection_type.value

        # create and fit ClusterAnalysis
        self._cluster_analysis = ClusterAnalysis(
            correlation_threshold=self.config.correlation_threshold
        )

        # fit_transform returns selected features but we need more info
        _ = self._cluster_analysis.fit_transform(
            df=data,
            target=target,
            input_cols=feature_columns,
            selection_type=selection_type,
            verbose=False
        )

        elapsed = time.time() - start_time

        # extract cluster information for logging
        self._selected_features = self._cluster_analysis.selected_features or []
        clusters_dict = self._cluster_analysis.get_clusters()

        # build detailed logging info
        cluster_infos = []
        feature_to_cluster = {}
        dropped_features = []

        for cluster_name, cluster_data in clusters_dict.items():
            cluster_id = int(cluster_name.split('_')[1])
            selected_feat = cluster_data[0]
            all_feats = cluster_data[1]

            info = ClusterInfo(
                cluster_id=cluster_id,
                selected_feature=selected_feat,
                all_features=all_feats,
                n_features=len(all_feats)
            )
            cluster_infos.append(info)

            # track which cluster each feature belongs to
            for feat in all_feats:
                feature_to_cluster[feat] = cluster_id
                if feat != selected_feat:
                    dropped_features.append(feat)

        # store clustering info for later retrieval
        self._clustering_info = ClusteringInfo(
            n_input_features=n_input,
            n_clusters=len(cluster_infos),
            n_selected_features=len(self._selected_features),
            correlation_threshold=self.config.correlation_threshold,
            selection_method=selection_type,
            clusters=cluster_infos,
            feature_to_cluster=feature_to_cluster,
            dropped_features=dropped_features,
            elapsed_seconds=elapsed
        )

        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform data by keeping only selected features.

        Args:
            data: DataFrame to transform
            **kwargs: Additional arguments (ignored)

        Returns:
            DataFrame with only selected features
        """
        if not self._is_fitted:
            raise RuntimeError("ClusteringStage must be fitted before transform")

        # keep only selected features that exist in data
        cols_to_keep = [col for col in self._selected_features if col in data.columns]

        # also keep non-feature columns like target, sample_type
        for col in data.columns:
            if col not in self._selected_features:
                # check if its a special column that should be preserved
                if col in ['sample_type'] or col.endswith('_target') or col == kwargs.get('target'):
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
            feature_columns: Features to cluster
            **kwargs: Additional arguments

        Returns:
            StageResult with transformed data and logging info
        """
        self.fit(data, target, feature_columns, **kwargs)
        transformed = self.transform(data, target=target, **kwargs)

        # build result with detailed logs
        logs = self._build_logs()

        return StageResult(
            data=transformed,
            logs=logs,
            metadata={
                'selected_features': self._selected_features.copy(),
                'n_clusters': self._clustering_info.n_clusters,
                'dropped_features': self._clustering_info.dropped_features.copy()
            }
        )

    def _build_logs(self) -> Dict[str, Any]:
        """Build comprehensive logs for visualization."""
        if self._clustering_info is None:
            return {}

        info = self._clustering_info

        # cluster details for visualization
        cluster_details = []
        for cl in info.clusters:
            cluster_details.append({
                'cluster_id': cl.cluster_id,
                'selected_feature': cl.selected_feature,
                'all_features': cl.all_features,
                'n_features': cl.n_features,
                'dropped_features': [f for f in cl.all_features if f != cl.selected_feature]
            })

        return {
            'stage_name': 'ClusteringStage',
            'summary': {
                'input_features': info.n_input_features,
                'output_features': info.n_selected_features,
                'n_clusters': info.n_clusters,
                'correlation_threshold': info.correlation_threshold,
                'selection_method': info.selection_method,
                'elapsed_seconds': round(info.elapsed_seconds, 3)
            },
            'clusters': cluster_details,
            'dropped_features': info.dropped_features,
            'feature_to_cluster': info.feature_to_cluster
        }

    def get_clustering_info(self) -> Optional[ClusteringInfo]:
        """Get detailed clustering information."""
        return self._clustering_info

    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self._selected_features.copy()

    def get_cluster_for_feature(self, feature: str) -> Optional[int]:
        """Get cluster ID for a given feature."""
        if self._clustering_info is None:
            return None
        return self._clustering_info.feature_to_cluster.get(feature)

    def get_dendrogram(self, output_path: Optional[str] = None):
        """
        Generate dendrogram plot showing feature clusters.

        Args:
            output_path: Path to save the plot. If None, shows interactively.
        """
        if self._cluster_analysis is None:
            raise RuntimeError("Must fit ClusteringStage before plotting")

        savefig = output_path is not None
        output_name = output_path or 'dendrogram.png'

        self._cluster_analysis.plot_dendrogram(
            savefig=savefig,
            output_name=output_name
        )

    def validate(self, data: pd.DataFrame, **kwargs) -> Tuple[bool, List[str]]:
        """
        Validate that clustering produced reasonable results.

        Args:
            data: DataFrame to validate
            **kwargs: Additional arguments

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        if not self._is_fitted:
            issues.append("ClusteringStage has not been fitted")
            return False, issues

        if len(self._selected_features) == 0:
            issues.append("No features were selected from clustering")
            return False, issues

        # check if we have too few features
        if self._clustering_info:
            reduction_ratio = (
                self._clustering_info.n_selected_features /
                max(self._clustering_info.n_input_features, 1)
            )
            if reduction_ratio < 0.1:
                issues.append(
                    f"Clustering reduced features too aggressively: "
                    f"{self._clustering_info.n_input_features} -> "
                    f"{self._clustering_info.n_selected_features}"
                )

        return len(issues) == 0, issues
