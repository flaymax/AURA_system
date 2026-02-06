#!/usr/bin/env python3
"""
Generate synthetic test data for AURA pipeline testing.

Creates a CSV file with realistic features and a binary target for testing.

Usage:
    python generate_test_data.py                    # Creates test_data.csv
    python generate_test_data.py -o custom.csv      # Custom output file
    python generate_test_data.py -n 10000           # 10,000 rows
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_test_data(
    n_samples: int = 5000,
    n_numeric_features: int = 30,
    n_categorical_features: int = 5,
    bad_rate: float = 0.15,
    train_ratio: float = 0.8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic test data for reasoning pipeline.

    Args:
        n_samples: Number of samples to generate
        n_numeric_features: Number of continuous numeric features
        n_categorical_features: Number of categorical features
        bad_rate: Proportion of bad outcomes (target=1)
        train_ratio: Proportion of data for training
        random_state: Random seed for reproducibility

    Returns:
        DataFrame with features, target, and sample_type
    """
    np.random.seed(random_state)

    data = {}

    # Generate numeric features with varying predictive power
    for i in range(n_numeric_features):
        # Some features are strongly predictive
        if i < 5:
            # Strong predictors
            base = np.random.normal(50, 15, n_samples)
            noise = np.random.normal(0, 5, n_samples)
            data[f"feature_{i+1:02d}"] = base + noise
        elif i < 15:
            # Medium predictors
            base = np.random.exponential(10, n_samples)
            noise = np.random.normal(0, 3, n_samples)
            data[f"feature_{i+1:02d}"] = np.abs(base + noise)
        else:
            # Weak/noise features
            data[f"feature_{i+1:02d}"] = np.random.normal(0, 1, n_samples)

    # Add some correlated features (will be clustered together)
    data["income"] = np.random.lognormal(10, 0.5, n_samples)
    data["income_log"] = np.log(data["income"]) + np.random.normal(0, 0.1, n_samples)
    data["debt"] = data["income"] * np.random.uniform(0.1, 0.8, n_samples)
    data["debt_ratio"] = data["debt"] / data["income"]

    # Age-related features
    data["age"] = np.random.normal(40, 12, n_samples).clip(18, 80)
    data["years_employed"] = (data["age"] - 18) * np.random.uniform(0, 0.8, n_samples)
    data["years_employed"] = data["years_employed"].clip(0, None)

    # History features
    data["history_months"] = np.random.exponential(60, n_samples).clip(0, 360)
    data["num_accounts"] = np.random.poisson(3, n_samples)
    data["num_delinquencies"] = np.random.poisson(0.5, n_samples)

    # Categorical features
    for i in range(n_categorical_features):
        n_categories = np.random.choice([2, 3, 4, 5])
        data[f"category_{i+1}"] = np.random.randint(0, n_categories, n_samples)

    # Generate target based on feature values
    # Higher income, lower debt ratio, more history = lower bad rate
    linear_score = (
        -0.5 * (data["income"] - np.mean(data["income"])) / np.std(data["income"])
        + 1.0 * (data["debt_ratio"] - np.mean(data["debt_ratio"])) / np.std(data["debt_ratio"])
        + 0.3 * (data["num_delinquencies"])
        - 0.2 * (data["age"] - np.mean(data["age"])) / np.std(data["age"])
        - 0.1 * data["feature_01"] / np.std(data["feature_01"])
        - 0.1 * data["feature_02"] / np.std(data["feature_02"])
        + np.random.normal(0, 1, n_samples)  # noise
    )

    # Convert to probability and sample
    prob_bad = 1 / (1 + np.exp(-linear_score))
    # Adjust to match target bad rate
    threshold = np.percentile(prob_bad, 100 * (1 - bad_rate))
    target = (prob_bad > threshold).astype(int)

    data["target"] = target

    # Create sample_type (0=train, 1=test)
    train_mask = np.random.rand(n_samples) < train_ratio
    data["sample_type"] = (~train_mask).astype(int)

    # Add some missing values
    df = pd.DataFrame(data)
    for col in df.columns:
        if col not in ["target", "sample_type"]:
            # Randomly set some values to NaN (1-5% per feature)
            missing_rate = np.random.uniform(0.01, 0.05)
            missing_mask = np.random.rand(n_samples) < missing_rate
            df.loc[missing_mask, col] = np.nan

    # Add a constant column (should be dropped by cleaner)
    df["constant_col"] = 1

    # Add a high-null column (should be dropped by cleaner)
    df["high_null_col"] = np.nan
    df.loc[np.random.rand(n_samples) < 0.02, "high_null_col"] = 1

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic test data for AURA pipeline"
    )
    parser.add_argument(
        "-o", "--output",
        default="test_data.csv",
        help="Output CSV file path (default: test_data.csv)"
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)"
    )
    parser.add_argument(
        "--bad-rate",
        type=float,
        default=0.15,
        help="Target bad rate (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    print(f"Generating {args.n_samples:,} samples with {args.bad_rate:.1%} bad rate...")

    df = generate_test_data(
        n_samples=args.n_samples,
        bad_rate=args.bad_rate,
        random_state=args.seed
    )

    df.to_csv(args.output, index=False)

    print(f"Saved to: {args.output}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Bad rate: {df['target'].mean():.2%}")
    print(f"Train/Test: {sum(df['sample_type']==0):,} / {sum(df['sample_type']==1):,}")


if __name__ == "__main__":
    main()
