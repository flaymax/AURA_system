#!/usr/bin/env bash
set -e

echo "evaluating model..."
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

python src/models/evaluate.py \
  --model models/latest.pkl \
  --data data/validation.csv

echo "Evaluation done"
