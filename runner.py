#!/usr/bin/env python3
"""
AURA Pipeline Runner

Command-line interface for running the AURA reasoning pipeline.
Supports running individual stages or the complete pipeline end-to-end.

Usage:
    python runner.py data.csv --target default_flag
    python runner.py data.csv --target default_flag --config config.yaml
    python runner.py data.csv --target default_flag --stage binning
    python runner.py data.csv --target default_flag --verbose
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for config files. Install with: pip install pyyaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def get_config_value(config: Dict[str, Any], *keys, default=None):
    """Safely get nested config value."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value

# Configure logging before imports
def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with console and optional file output."""
    logger = logging.getLogger("aura")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Format: timestamp - level - stage - message
    console_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


class PipelineRunner:
    """
    Orchestrates the AURA pipeline stages with comprehensive logging.
    """

    STAGES = [
        "cleaning",
        "type_detection",
        "binning",
        "clustering",
        "stepwise",
        "interactions",
        "final_filter",
        "training"
    ]

    def __init__(
        self,
        verbose: bool = False,
        log_file: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.logger = setup_logging(verbose, log_file)
        self.verbose = verbose
        self.config = config or {}
        self.stage_results: Dict[str, Any] = {}
        self.stage_times: Dict[str, float] = {}

    def _get_stage_logger(self, stage_name: str) -> logging.Logger:
        """Get a logger for a specific stage."""
        return logging.getLogger(f"aura.{stage_name}")

    def _log_stage_start(self, stage_name: str, description: str):
        """Log the start of a pipeline stage."""
        logger = self._get_stage_logger(stage_name)
        logger.info("=" * 60)
        logger.info(f"STARTING: {description}")
        logger.info("=" * 60)

    def _log_stage_end(self, stage_name: str, elapsed: float, features_in: int, features_out: int):
        """Log the end of a pipeline stage."""
        logger = self._get_stage_logger(stage_name)
        logger.info(f"COMPLETED in {elapsed:.2f}s | Features: {features_in} -> {features_out}")
        logger.info("")

    def _log_stage_error(self, stage_name: str, error: Exception):
        """Log a stage error."""
        logger = self._get_stage_logger(stage_name)
        logger.error(f"FAILED: {str(error)}")
        if self.verbose:
            import traceback
            logger.debug(traceback.format_exc())

    def load_data(self, csv_path: str, target: str) -> pd.DataFrame:
        """Load and validate input data."""
        logger = self._get_stage_logger("data_loader")

        logger.info(f"Loading data from: {csv_path}")
        start = time.time()

        try:
            data = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV: {e}")
            raise

        elapsed = time.time() - start
        logger.info(f"Loaded {len(data):,} rows x {len(data.columns)} columns in {elapsed:.2f}s")

        # Validate target column
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found. Available: {list(data.columns)}")

        logger.info(f"Target column: {target}")
        logger.info(f"Target distribution: {dict(data[target].value_counts())}")

        # Check for sample_type
        sample_type_col = get_config_value(self.config, 'data', 'sample_type_column', default='sample_type')
        if sample_type_col not in data.columns:
            train_ratio = get_config_value(self.config, 'data', 'train_ratio', default=0.8)
            random_state = get_config_value(self.config, 'general', 'random_state', default=42)
            logger.warning(f"No '{sample_type_col}' column found. Creating {train_ratio:.0%}/{1-train_ratio:.0%} train/test split.")
            n = len(data)
            np.random.seed(random_state)
            data['sample_type'] = np.where(
                np.random.rand(n) < train_ratio, 0, 1
            )
            logger.info(f"Created sample_type: train={sum(data['sample_type']==0)}, test={sum(data['sample_type']==1)}")

        return data

    def run_cleaning(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Run data cleaning stage."""
        from stages.preprocessing import DataCleaner, DataCleanerConfig

        stage_name = "cleaning"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Data Cleaning")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get config values
        max_null = get_config_value(self.config, 'cleaning', 'max_null_ratio', default=0.97)
        max_same = get_config_value(self.config, 'cleaning', 'max_same_value_ratio', default=0.97)

        stage_config = DataCleanerConfig(
            max_null_ratio=max_null,
            max_same_value_ratio=max_same
        )
        cleaner = DataCleaner(stage_config)

        try:
            result = cleaner.fit_transform(data, target=target)
            data_out = result.data if hasattr(result, 'data') else result

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])
            dropped = features_in - features_out

            if dropped > 0:
                logger.info(f"Dropped {dropped} features (nulls/constants/non-numeric)")

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_type_detection(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Run type detection stage."""
        from stages.preprocessing import TypeDetector, TypeDetectorConfig

        stage_name = "type_detection"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Feature Type Detection")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        config = TypeDetectorConfig()
        detector = TypeDetector(config)

        try:
            result = detector.fit_transform(data, target=target)
            data_out = result.data if hasattr(result, 'data') else result

            # Log type distribution
            if hasattr(detector, 'get_feature_types'):
                types = detector.get_feature_types()
                type_counts = {}
                for info in types.values():
                    t = info.feature_type.value if hasattr(info, 'feature_type') else str(info)
                    type_counts[t] = type_counts.get(t, 0) + 1
                logger.info(f"Feature types: {type_counts}")

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_binning(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Run WoE binning stage."""
        from stages.transformation import WoEBinnerStage, WoEBinnerConfig

        stage_name = "binning"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "WoE Binning")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get config values
        min_iv = get_config_value(self.config, 'binning', 'min_iv', default=0.02)
        max_bins = get_config_value(self.config, 'binning', 'max_bins', default=10)
        mono_mode = get_config_value(self.config, 'binning', 'monotonicity_mode', default="warn")
        psi_mode = get_config_value(self.config, 'binning', 'psi_mode', default="warn")
        psi_threshold = get_config_value(self.config, 'binning', 'psi_threshold', default=0.25)

        stage_config = WoEBinnerConfig(
            min_iv=min_iv,
            max_bins=max_bins,
            monotonicity_mode=mono_mode,
            psi_mode=psi_mode,
            psi_threshold=psi_threshold
        )
        binner = WoEBinnerStage(stage_config)

        try:
            # Get sample_type for PSI calculation
            sample_type = data['sample_type'] if 'sample_type' in data.columns else None

            result = binner.fit_transform(data, target=target, sample_type=sample_type)
            data_out = result.data if hasattr(result, 'data') else result

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])
            dropped = features_in - features_out

            if dropped > 0:
                logger.info(f"Dropped {dropped} features (low IV < 0.02)")

            # Log IV stats
            if hasattr(binner, 'get_iv_values'):
                iv_values = binner.get_iv_values()
                if iv_values:
                    logger.info(f"IV range: {min(iv_values.values()):.3f} - {max(iv_values.values()):.3f}")

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_clustering(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Run feature clustering stage."""
        from stages.selection import ClusteringStage
        from core.base import ClusteringConfig

        stage_name = "clustering"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Feature Clustering")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get config values
        corr_thresh = get_config_value(self.config, 'clustering', 'correlation_threshold', default=0.7)
        sel_type = get_config_value(self.config, 'clustering', 'selection_type', default="max_test")
        pval_enabled = get_config_value(self.config, 'clustering', 'pvalue_filter_enabled', default=True)
        pval_thresh = get_config_value(self.config, 'clustering', 'pvalue_threshold', default=0.05)
        linkage = get_config_value(self.config, 'clustering', 'linkage_method', default="average")

        stage_config = ClusteringConfig(
            correlation_threshold=corr_thresh,
            selection_type=sel_type,
            pvalue_filter_enabled=pval_enabled,
            pvalue_threshold=pval_thresh,
            linkage_method=linkage
        )
        clustering = ClusteringStage(stage_config)

        try:
            result = clustering.fit_transform(data, target=target)
            data_out = result.data if hasattr(result, 'data') else result

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])

            # Log clustering info
            if hasattr(clustering, 'get_clustering_info'):
                info = clustering.get_clustering_info()
                if info:
                    logger.info(f"Found {info.n_clusters} clusters from {info.n_features_input} features")

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_stepwise(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Run stepwise selection stage."""
        from stages.selection import StepwiseSelectionStage
        from core.base import SelectionConfig

        stage_name = "stepwise"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Stepwise Feature Selection")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get config values
        alpha_enter = get_config_value(self.config, 'stepwise', 'alpha_enter', default=0.05)
        alpha_exit = get_config_value(self.config, 'stepwise', 'alpha_exit', default=0.10)
        max_features = get_config_value(self.config, 'stepwise', 'max_features', default=None)
        min_features = get_config_value(self.config, 'stepwise', 'min_features', default=3)

        # Early stopping config
        es_enabled = get_config_value(self.config, 'stepwise', 'early_stopping', 'enabled', default=True)
        es_patience = get_config_value(self.config, 'stepwise', 'early_stopping', 'patience', default=5)
        es_min_imp = get_config_value(self.config, 'stepwise', 'early_stopping', 'min_improvement', default=0.001)
        es_restore = get_config_value(self.config, 'stepwise', 'early_stopping', 'restore_best', default=True)
        es_metric = get_config_value(self.config, 'stepwise', 'early_stopping', 'monitor_metric', default="test_auc")

        stage_config = SelectionConfig(
            alpha_enter=alpha_enter,
            alpha_exit=alpha_exit,
            max_features=max_features,
            min_features=min_features,
            early_stopping_enabled=es_enabled,
            patience=es_patience,
            min_improvement=es_min_imp,
            restore_best=es_restore,
            monitor_metric=es_metric
        )
        stepwise = StepwiseSelectionStage(config)

        try:
            result = stepwise.fit_transform(data, target=target)
            data_out = result.data if hasattr(result, 'data') else result

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])

            # Log stepwise info
            if hasattr(stepwise, 'get_stepwise_info'):
                info = stepwise.get_stepwise_info()
                if info:
                    logger.info(f"Selected {info.n_selected_features} features in {info.n_steps} steps")
                    logger.info(f"Final AUC: train={info.final_train_auc:.4f}, test={info.final_test_auc:.4f}")
                    if info.early_stopping and info.early_stopping.triggered:
                        logger.info(f"Early stopping triggered at step {info.early_stopping.stopped_at_step}")

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_interactions(self, data: pd.DataFrame, target: str, enabled: bool = False) -> pd.DataFrame:
        """Run interaction detection stage (optional)."""
        # Check if enabled via config or CLI flag
        config_enabled = get_config_value(self.config, 'interactions', 'enabled', default=False)
        if not enabled and not config_enabled:
            return data

        from stages.selection import InteractionDetectorStage
        from core.base import InteractionConfig

        stage_name = "interactions"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Feature Interaction Detection")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get config values
        int_types = get_config_value(self.config, 'interactions', 'interaction_types', default=["multiplicative"])
        max_int = get_config_value(self.config, 'interactions', 'max_interactions', default=5)
        min_auc = get_config_value(self.config, 'interactions', 'min_auc_improvement', default=0.005)
        corr_thresh = get_config_value(self.config, 'interactions', 'correlation_threshold', default=0.5)
        pval_thresh = get_config_value(self.config, 'interactions', 'pvalue_threshold', default=0.05)

        stage_config = InteractionConfig(
            enabled=True,
            interaction_types=int_types,
            max_interactions=max_int,
            min_auc_improvement=min_auc,
            correlation_threshold=corr_thresh,
            pvalue_threshold=pval_thresh
        )
        detector = InteractionDetectorStage(stage_config)

        try:
            result = detector.fit_transform(data, target=target)
            data_out = result.data if hasattr(result, 'data') else result

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])

            # Log interaction info
            if hasattr(detector, 'get_interaction_info'):
                info = detector.get_interaction_info()
                if info:
                    logger.info(f"Tested {info.n_candidates_tested} interactions")
                    logger.info(f"Selected {info.n_interactions_selected} interactions")

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_final_filter(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """Run final filter stage."""
        from stages.selection import FinalFilterStage, FinalFilterConfig

        stage_name = "final_filter"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Final Filter (P-value & VIF)")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get config values
        max_pval = get_config_value(self.config, 'final_filter', 'max_pvalue', default=0.05)
        max_vif = get_config_value(self.config, 'final_filter', 'max_vif', default=5.0)
        check_sign = get_config_value(self.config, 'final_filter', 'check_coefficient_sign', default=True)

        stage_config = FinalFilterConfig(
            max_pvalue=max_pval,
            max_vif=max_vif,
            check_coefficient_sign=check_sign
        )
        final_filter = FinalFilterStage(stage_config)

        try:
            result = final_filter.fit_transform(data, target=target)
            data_out = result.data if hasattr(result, 'data') else result

            features_out = len([c for c in data_out.columns if c not in [target, 'sample_type']])

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_out)

            return data_out

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_training(self, data: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Run model training stage."""
        from stages.modeling import ModelTrainerStage
        from core.base import ModelConfig, BootstrapConfig

        stage_name = "training"
        logger = self._get_stage_logger(stage_name)
        self._log_stage_start(stage_name, "Model Training")

        start = time.time()
        features_in = len([c for c in data.columns if c not in [target, 'sample_type']])

        # Get model config values
        regularization = get_config_value(self.config, 'training', 'regularization', default=None)
        C = get_config_value(self.config, 'training', 'C', default=1.0)
        fit_intercept = get_config_value(self.config, 'training', 'fit_intercept', default=True)
        max_iter = get_config_value(self.config, 'training', 'max_iter', default=1000)
        solver = get_config_value(self.config, 'training', 'solver', default="lbfgs")

        model_config = ModelConfig(
            regularization=regularization,
            C=C,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            solver=solver
        )

        # Get bootstrap config values
        bs_enabled = get_config_value(self.config, 'bootstrap', 'enabled', default=True)
        bs_iterations = get_config_value(self.config, 'bootstrap', 'n_iterations', default=1000)
        bs_confidence = get_config_value(self.config, 'bootstrap', 'confidence_level', default=0.95)
        bs_metrics = get_config_value(self.config, 'bootstrap', 'metrics', default=["auc", "gini", "ks"])
        bs_stratified = get_config_value(self.config, 'bootstrap', 'stratified', default=True)
        random_state = get_config_value(self.config, 'general', 'random_state', default=42)

        bootstrap_config = BootstrapConfig(
            enabled=bs_enabled,
            n_iterations=bs_iterations,
            confidence_level=bs_confidence,
            metrics=bs_metrics,
            stratified=bs_stratified,
            random_state=random_state
        )
        trainer = ModelTrainerStage(config=model_config, bootstrap_config=bootstrap_config)

        try:
            result = trainer.fit_transform(data, target=target)

            # Log model info
            if hasattr(trainer, 'get_model_info'):
                info = trainer.get_model_info()
                if info:
                    logger.info(f"Train AUC: {info.train_auc:.4f}, Test AUC: {info.test_auc:.4f}")
                    logger.info(f"Train Gini: {info.train_gini:.4f}, Test Gini: {info.test_gini:.4f}")
                    logger.info(f"Model has {len(info.coefficients)} features")

                    if info.bootstrap_results:
                        br = info.bootstrap_results
                        if hasattr(br, 'test_auc_ci'):
                            ci = br.test_auc_ci
                            logger.info(f"Test AUC 95% CI: [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]")

            elapsed = time.time() - start
            self.stage_times[stage_name] = elapsed
            self._log_stage_end(stage_name, elapsed, features_in, features_in)

            return result.metadata if hasattr(result, 'metadata') else {}

        except Exception as e:
            self._log_stage_error(stage_name, e)
            raise

    def run_pipeline(
        self,
        csv_path: str,
        target: str,
        stages: Optional[List[str]] = None,
        enable_interactions: bool = False
    ) -> bool:
        """
        Run the complete pipeline or specific stages.

        Args:
            csv_path: Path to input CSV file
            target: Target column name
            stages: List of stages to run (None = all)
            enable_interactions: Whether to run interaction detection

        Returns:
            True if pipeline completed successfully
        """
        main_logger = self._get_stage_logger("pipeline")
        main_logger.info("=" * 70)
        main_logger.info("AURA PIPELINE STARTED")
        main_logger.info(f"Input: {csv_path}")
        main_logger.info(f"Target: {target}")
        main_logger.info("=" * 70)

        pipeline_start = time.time()

        try:
            # Load data
            data = self.load_data(csv_path, target)

            # Determine which stages to run
            stages_to_run = stages if stages else self.STAGES

            # Run stages
            if "cleaning" in stages_to_run:
                data = self.run_cleaning(data, target)

            if "type_detection" in stages_to_run:
                data = self.run_type_detection(data, target)

            if "binning" in stages_to_run:
                data = self.run_binning(data, target)

            if "clustering" in stages_to_run:
                data = self.run_clustering(data, target)

            if "stepwise" in stages_to_run:
                data = self.run_stepwise(data, target)

            if "interactions" in stages_to_run:
                data = self.run_interactions(data, target, enabled=enable_interactions)

            if "final_filter" in stages_to_run:
                data = self.run_final_filter(data, target)

            if "training" in stages_to_run:
                self.run_training(data, target)

            # Summary
            pipeline_elapsed = time.time() - pipeline_start
            main_logger.info("=" * 70)
            main_logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            main_logger.info(f"Total time: {pipeline_elapsed:.2f}s")
            main_logger.info("Stage times:")
            for stage, elapsed in self.stage_times.items():
                main_logger.info(f"  {stage}: {elapsed:.2f}s")
            main_logger.info("=" * 70)

            print("\nok")
            return True

        except Exception as e:
            main_logger.error("=" * 70)
            main_logger.error("PIPELINE FAILED")
            main_logger.error(f"Error: {str(e)}")
            main_logger.error("=" * 70)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="AURA Pipeline Runner - Reasoning Model Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python runner.py data.csv --target default_flag

  # Run with config file
  python runner.py data.csv --target default_flag --config config.yaml

  # Run with verbose logging
  python runner.py data.csv --target default_flag --verbose

  # Run specific stage only
  python runner.py data.csv --target default_flag --stage binning

  # Run up to a specific stage
  python runner.py data.csv --target default_flag --until clustering

  # Enable interaction detection
  python runner.py data.csv --target default_flag --interactions

  # Save logs to file
  python runner.py data.csv --target default_flag --log-file pipeline.log

Available stages: cleaning, type_detection, binning, clustering,
                  stepwise, interactions, final_filter, training
        """
    )

    parser.add_argument(
        "csv_file",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Name of target column (e.g., default_flag)"
    )
    parser.add_argument(
        "--stage", "-s",
        choices=PipelineRunner.STAGES,
        help="Run only this specific stage"
    )
    parser.add_argument(
        "--until", "-u",
        choices=PipelineRunner.STAGES,
        help="Run pipeline up to and including this stage"
    )
    parser.add_argument(
        "--interactions", "-i",
        action="store_true",
        help="Enable feature interaction detection"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging"
    )
    parser.add_argument(
        "--log-file", "-l",
        help="Save logs to this file"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file"
    )

    args = parser.parse_args()

    # Validate file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File not found: {args.csv_file}")
        sys.exit(1)

    # Load config file if provided
    config = {}
    if args.config:
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        try:
            config = load_config(args.config)
            print(f"Loaded config from: {args.config}")
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)

    # Determine stages to run
    stages = None
    if args.stage:
        stages = [args.stage]
    elif args.until:
        until_idx = PipelineRunner.STAGES.index(args.until)
        stages = PipelineRunner.STAGES[:until_idx + 1]

    # Get log file from config if not specified via CLI
    log_file = args.log_file or get_config_value(config, 'logging', 'file', default=None)

    # Run pipeline
    runner = PipelineRunner(verbose=args.verbose, log_file=log_file, config=config)
    success = runner.run_pipeline(
        csv_path=args.csv_file,
        target=args.target,
        stages=stages,
        enable_interactions=args.interactions
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
