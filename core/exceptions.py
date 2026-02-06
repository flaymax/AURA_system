"""
Pipeline exceptions and trigger mechanism.

This module provides custom exceptions for pipeline failures and
a trigger system that saves error details to JSON for later processing
by web frontend or API.

When a critical error occurs (like no features remaining), the pipeline
raises a PipelineTrigger exception which is caught and logged to JSON file.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum


logger = logging.getLogger(__name__)


# =============================================================================
# Trigger Severity Levels
# =============================================================================

class TriggerSeverity(Enum):
    """Severity level of pipeline trigger.

    INFO - informational, pipeline continues
    WARNING - potential issue, pipeline continues but logs warning
    ERROR - critical error, pipeline stops
    FATAL - unrecoverable error, pipeline stops immediately
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class TriggerCategory(Enum):
    """Category of trigger for grouping and filtering.

    Helps frontend display appropriate messages and icons.
    """
    DATA_QUALITY = "data_quality"
    FEATURE_SELECTION = "feature_selection"
    MODEL_TRAINING = "model_training"
    VALIDATION = "validation"
    SYSTEM = "system"


# =============================================================================
# Trigger Data Container
# =============================================================================

@dataclass
class TriggerDetails:
    """Container for trigger information.

    This gets serialized to JSON when trigger is raised.

    Attributes:
        code: unique error code (e.g. "ERR_NO_FEATURES")
        message: human-readable error message
        severity: how critical is this error
        category: what type of error is this
        stage_name: which pipeline stage raised the trigger
        details: additional context and diagnostic info
        suggestions: list of possible fixes user can try
        timestamp: when trigger was raised
        feature_list: affected features (if applicable)
    """
    code: str
    message: str
    severity: TriggerSeverity = TriggerSeverity.ERROR
    category: TriggerCategory = TriggerCategory.DATA_QUALITY
    stage_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    feature_list: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "stage_name": self.stage_name,
            "details": self.details,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp.isoformat(),
            "feature_list": self.feature_list,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# =============================================================================
# Custom Exceptions
# =============================================================================

class PipelineException(Exception):
    """Base exception for all pipeline errors.

    All custom pipeline exceptions should inherit from this class.
    """

    def __init__(self, message: str, trigger: Optional[TriggerDetails] = None):
        super().__init__(message)
        self.trigger = trigger
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        if self.trigger:
            return f"[{self.trigger.code}] {self.message}"
        return str(self.message)


class PipelineTrigger(PipelineException):
    """Exception raised when pipeline encounters a critical issue.

    This exception includes detailed trigger information that gets
    saved to JSON file for frontend consumption.

    Example:
        >>> trigger = TriggerDetails(
        ...     code="ERR_NO_FEATURES",
        ...     message="No features remaining after cleaning",
        ...     severity=TriggerSeverity.ERROR,
        ... )
        >>> raise PipelineTrigger("Pipeline failed", trigger)
    """

    def __init__(self, message: str, trigger: TriggerDetails):
        super().__init__(message, trigger)
        self.trigger = trigger


class DataQualityError(PipelineTrigger):
    """Raised when data quality issues prevent pipeline from continuing."""
    pass


class FeatureSelectionError(PipelineTrigger):
    """Raised when feature selection fails (e.g. no features pass criteria)."""
    pass


class ModelTrainingError(PipelineTrigger):
    """Raised when model training fails."""
    pass


class ValidationError(PipelineTrigger):
    """Raised when validation checks fail."""
    pass


# =============================================================================
# Trigger Manager
# =============================================================================

class TriggerManager:
    """Manages trigger logging and persistence.

    This class handles saving triggers to JSON files and maintaining
    a history of all triggers raised during pipeline execution.

    The trigger files are saved to a configurable output directory
    and can be picked up by the web frontend.

    Example:
        >>> manager = TriggerManager(output_dir="./triggers")
        >>> manager.raise_trigger(trigger_details)  # raises and saves
    """

    # Default output directory for trigger files
    DEFAULT_OUTPUT_DIR = "./pipeline_triggers"

    def __init__(self, output_dir: Optional[str] = None, session_id: Optional[str] = None):
        """Initialize trigger manager.

        Args:
            output_dir: directory to save trigger JSON files
            session_id: unique identifier for this pipeline run
        """
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.triggers: List[TriggerDetails] = []

        # Create output directory if it does not exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_trigger(self, trigger: TriggerDetails, save_to_file: bool = True) -> None:
        """Log trigger without raising exception.

        Use this for warnings and informational triggers that
        should not stop the pipeline.

        Args:
            trigger: trigger details to log
            save_to_file: whether to save to JSON file
        """
        self.triggers.append(trigger)

        # Log based on severity
        log_message = f"[{trigger.code}] {trigger.message}"
        if trigger.severity == TriggerSeverity.INFO:
            logger.info(log_message)
        elif trigger.severity == TriggerSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.error(log_message)

        if save_to_file:
            self._save_trigger(trigger)

    def raise_trigger(self, trigger: TriggerDetails, save_to_file: bool = True) -> None:
        """Log trigger and raise appropriate exception.

        This method saves the trigger to JSON and then raises
        an exception to stop pipeline execution.

        Args:
            trigger: trigger details
            save_to_file: whether to save to JSON file

        Raises:
            PipelineTrigger or subclass based on category
        """
        self.log_trigger(trigger, save_to_file)

        # Choose exception class based on category
        exception_map = {
            TriggerCategory.DATA_QUALITY: DataQualityError,
            TriggerCategory.FEATURE_SELECTION: FeatureSelectionError,
            TriggerCategory.MODEL_TRAINING: ModelTrainingError,
            TriggerCategory.VALIDATION: ValidationError,
            TriggerCategory.SYSTEM: PipelineTrigger,
        }

        exception_class = exception_map.get(trigger.category, PipelineTrigger)
        raise exception_class(trigger.message, trigger)

    def _save_trigger(self, trigger: TriggerDetails) -> Path:
        """Save trigger to JSON file.

        File naming convention: {session_id}_{timestamp}_{code}.json

        Args:
            trigger: trigger to save

        Returns:
            Path to saved file
        """
        timestamp = trigger.timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.session_id}_{timestamp}_{trigger.code}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(trigger.to_json())

        logger.debug(f"Trigger saved to {filepath}")
        return filepath

    def get_all_triggers(self) -> List[TriggerDetails]:
        """Get all triggers logged in this session."""
        return self.triggers.copy()

    def get_errors(self) -> List[TriggerDetails]:
        """Get only ERROR and FATAL triggers."""
        return [
            t for t in self.triggers
            if t.severity in (TriggerSeverity.ERROR, TriggerSeverity.FATAL)
        ]

    def get_warnings(self) -> List[TriggerDetails]:
        """Get only WARNING triggers."""
        return [t for t in self.triggers if t.severity == TriggerSeverity.WARNING]

    def has_errors(self) -> bool:
        """Check if any error triggers were raised."""
        return len(self.get_errors()) > 0

    def clear(self) -> None:
        """Clear all logged triggers."""
        self.triggers = []

    def export_session_summary(self) -> Path:
        """Export summary of all triggers in session to single JSON file.

        Returns:
            Path to summary file
        """
        summary = {
            "session_id": self.session_id,
            "total_triggers": len(self.triggers),
            "error_count": len(self.get_errors()),
            "warning_count": len(self.get_warnings()),
            "triggers": [t.to_dict() for t in self.triggers],
            "exported_at": datetime.now().isoformat(),
        }

        filepath = self.output_dir / f"{self.session_id}_summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        return filepath


# =============================================================================
# Pre-defined Trigger Codes
# =============================================================================

class TriggerCodes:
    """Standard trigger codes used across pipeline.

    Using standardized codes makes it easier for frontend
    to handle specific error types.
    """

    # Data Quality triggers
    NO_FEATURES_REMAINING = "ERR_NO_FEATURES"
    ALL_FEATURES_NULL = "ERR_ALL_NULL"
    NO_TRAINING_DATA = "ERR_NO_TRAIN_DATA"
    TARGET_INVALID = "ERR_TARGET_INVALID"
    SAMPLE_TYPE_MISSING = "ERR_SAMPLE_TYPE"

    # Feature Selection triggers
    NO_SIGNIFICANT_FEATURES = "ERR_NO_SIGNIFICANT"
    HIGH_MULTICOLLINEARITY = "WARN_HIGH_VIF"
    LOW_IV_ALL = "ERR_LOW_IV"

    # Model Training triggers
    CONVERGENCE_FAILED = "ERR_NO_CONVERGENCE"
    SINGULAR_MATRIX = "ERR_SINGULAR"
    COEFFICIENT_SIGN = "WARN_COEF_SIGN"

    # Validation triggers
    MONOTONICITY_VIOLATED = "WARN_MONOTONIC"
    PSI_THRESHOLD_EXCEEDED = "WARN_PSI_HIGH"
    GINI_DEGRADATION = "WARN_GINI_DROP"


# =============================================================================
# Convenience Functions
# =============================================================================

def create_no_features_trigger(
    stage_name: str,
    original_count: int,
    dropped_features: Dict[str, str],
) -> TriggerDetails:
    """Create standard trigger for when no features remain.

    Args:
        stage_name: which stage caused the issue
        original_count: how many features we started with
        dropped_features: dict of feature -> reason for dropping

    Returns:
        TriggerDetails ready to raise
    """
    return TriggerDetails(
        code=TriggerCodes.NO_FEATURES_REMAINING,
        message=f"No features remaining after {stage_name}. "
                f"Started with {original_count} features, all were dropped.",
        severity=TriggerSeverity.ERROR,
        category=TriggerCategory.DATA_QUALITY,
        stage_name=stage_name,
        details={
            "original_feature_count": original_count,
            "dropped_features": dropped_features,
        },
        suggestions=[
            "Check data quality - too many missing values?",
            "Lower the null threshold in preprocessing config",
            "Verify that numeric features are properly formatted",
            "Make sure dataset contains actual features, not just IDs",
        ],
        feature_list=list(dropped_features.keys()),
    )


def create_data_quality_trigger(
    code: str,
    message: str,
    stage_name: str,
    details: Dict[str, Any],
    suggestions: Optional[List[str]] = None,
) -> TriggerDetails:
    """Create generic data quality trigger.

    Args:
        code: error code
        message: error message
        stage_name: which stage
        details: additional info
        suggestions: possible fixes

    Returns:
        TriggerDetails ready to raise
    """
    return TriggerDetails(
        code=code,
        message=message,
        severity=TriggerSeverity.ERROR,
        category=TriggerCategory.DATA_QUALITY,
        stage_name=stage_name,
        details=details,
        suggestions=suggestions or [],
    )


# Global trigger manager instance (can be overridden)
_global_trigger_manager: Optional[TriggerManager] = None


def get_trigger_manager() -> TriggerManager:
    """Get global trigger manager instance.

    Creates one if it does not exist yet.
    """
    global _global_trigger_manager
    if _global_trigger_manager is None:
        _global_trigger_manager = TriggerManager()
    return _global_trigger_manager


def set_trigger_manager(manager: TriggerManager) -> None:
    """Set global trigger manager instance."""
    global _global_trigger_manager
    _global_trigger_manager = manager
