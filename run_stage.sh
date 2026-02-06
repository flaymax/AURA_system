#!/bin/bash
#
# AURA Pipeline Runner - Single Stage
#
# Usage:
#   ./run_stage.sh <stage> <csv_file> <target_column> [options]
#
# Stages:
#   cleaning       - Data cleaning (remove nulls, constants)
#   type_detection - Feature type detection
#   binning        - WoE binning
#   clustering     - Feature clustering
#   stepwise       - Stepwise selection
#   interactions   - Interaction detection
#   final_filter   - P-value and VIF filtering
#   training       - Model training
#
# Examples:
#   ./run_stage.sh binning data.csv default_flag
#   ./run_stage.sh stepwise data.csv default_flag --verbose
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VALID_STAGES="cleaning type_detection binning clustering stepwise interactions final_filter training"

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <stage> <csv_file> <target_column> [options]"
    echo ""
    echo "Available stages:"
    echo "  cleaning       - Data cleaning (remove nulls, constants)"
    echo "  type_detection - Feature type detection"
    echo "  binning        - WoE binning"
    echo "  clustering     - Feature clustering"
    echo "  stepwise       - Stepwise selection"
    echo "  interactions   - Interaction detection"
    echo "  final_filter   - P-value and VIF filtering"
    echo "  training       - Model training"
    echo ""
    echo "Options:"
    echo "  --verbose, -v    Enable verbose logging"
    echo ""
    echo "Example:"
    echo "  $0 binning data.csv default_flag --verbose"
    exit 1
fi

STAGE="$1"
CSV_FILE="$2"
TARGET="$3"
shift 3

# Validate stage
if [[ ! " $VALID_STAGES " =~ " $STAGE " ]]; then
    echo "Error: Invalid stage '$STAGE'"
    echo "Valid stages: $VALID_STAGES"
    exit 1
fi

# Check if file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File not found: $CSV_FILE"
    exit 1
fi

echo "========================================"
echo "AURA Pipeline - Stage: $STAGE"
echo "========================================"
echo "Input:  $CSV_FILE"
echo "Target: $TARGET"
echo "========================================"

# Run the specific stage
python "$SCRIPT_DIR/runner.py" "$CSV_FILE" --target "$TARGET" --stage "$STAGE" "$@"
