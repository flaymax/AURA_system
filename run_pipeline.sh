#!/bin/bash
#
# AURA Pipeline Runner - Full Pipeline
#
# Usage:
#   ./run_pipeline.sh data.csv target_column
#   ./run_pipeline.sh data.csv target_column --config config.yaml
#   ./run_pipeline.sh data.csv target_column --verbose
#   ./run_pipeline.sh data.csv target_column --interactions
#
# Examples:
#   ./run_pipeline.sh data.csv target_column
#   ./run_pipeline.sh data.csv target_column --config config.yaml
#   ./run_pipeline.sh data.csv target_column --verbose --log-file output.log
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <csv_file> <target_column> [options]"
    echo ""
    echo "Options:"
    echo "  --verbose, -v        Enable verbose logging"
    echo "  --interactions, -i   Enable interaction detection"
    echo "  --log-file FILE      Save logs to file"
    echo ""
    echo "Example:"
    echo "  $0 data.csv default_flag --verbose"
    exit 1
fi

CSV_FILE="$1"
TARGET="$2"
shift 2

# Check if file exists
if [ ! -f "$CSV_FILE" ]; then
    echo "Error: File not found: $CSV_FILE"
    exit 1
fi

echo "========================================"
echo "AURA Pipeline - Full Run"
echo "========================================"
echo "Input:  $CSV_FILE"
echo "Target: $TARGET"
echo "========================================"

# Run the pipeline
python "$SCRIPT_DIR/runner.py" "$CSV_FILE" --target "$TARGET" "$@"
