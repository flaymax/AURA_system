#!/bin/bash
#
# AURA Pipeline Runner - Run Until Stage
#
# Runs the pipeline from start up to and including the specified stage.
#
# Usage:
#   ./run_until.sh <stage> <csv_file> <target_column> [options]
#
# Stages (in order):
#   1. cleaning       - Data cleaning
#   2. type_detection - Feature type detection
#   3. binning        - WoE binning
#   4. clustering     - Feature clustering
#   5. stepwise       - Stepwise selection
#   6. interactions   - Interaction detection
#   7. final_filter   - P-value and VIF filtering
#   8. training       - Model training
#
# Examples:
#   ./run_until.sh clustering data.csv default_flag
#   ./run_until.sh stepwise data.csv default_flag --verbose
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VALID_STAGES="cleaning type_detection binning clustering stepwise interactions final_filter training"

# Check arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <stage> <csv_file> <target_column> [options]"
    echo ""
    echo "Runs pipeline from start until (and including) the specified stage."
    echo ""
    echo "Stages (in order):"
    echo "  1. cleaning       - Data cleaning"
    echo "  2. type_detection - Feature type detection"
    echo "  3. binning        - WoE binning"
    echo "  4. clustering     - Feature clustering"
    echo "  5. stepwise       - Stepwise selection"
    echo "  6. interactions   - Interaction detection"
    echo "  7. final_filter   - P-value and VIF filtering"
    echo "  8. training       - Model training"
    echo ""
    echo "Example:"
    echo "  $0 clustering data.csv default_flag"
    echo "  (This runs: cleaning -> type_detection -> binning -> clustering)"
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
echo "AURA Pipeline - Run Until: $STAGE"
echo "========================================"
echo "Input:  $CSV_FILE"
echo "Target: $TARGET"
echo "========================================"

# Run pipeline until the specified stage
python "$SCRIPT_DIR/runner.py" "$CSV_FILE" --target "$TARGET" --until "$STAGE" "$@"
