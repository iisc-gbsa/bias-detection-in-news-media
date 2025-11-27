"""
Utility functions for bias visualization.
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to config relative to this script
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "viz_config.yaml")

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def ensure_directories(base_dir):
    """Create output directory structure."""
    # Get absolute path if relative
    if not os.path.isabs(base_dir):
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = os.path.join(script_dir, base_dir)

    directories = [
        os.path.join(base_dir, "time_series"),
        os.path.join(base_dir, "heatmaps"),
        os.path.join(base_dir, "distributions"),
        os.path.join(base_dir, "comparisons"),
        os.path.join(base_dir, "media"),
    ]

    # Also create data directories
    script_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directories.extend([
        os.path.join(script_base, "data", "raw"),
        os.path.join(script_base, "data", "aggregated"),
    ])

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print(f"Output directories created under: {base_dir}")
    return base_dir


def validate_bias_scores(df, bias_columns):
    """Validate bias scores are within 0-1 range."""
    for col in bias_columns:
        if col in df.columns:
            invalid_count = ((df[col] < 0) | (df[col] > 1)).sum()
            if invalid_count > 0:
                print(f"Warning: {invalid_count} invalid scores in {col}")
    return df


def get_color_palette(config):
    """Get color palette for bias types."""
    return [config['colors'].get(bias_type, '#333333')
            for bias_type in config['bias_types']]


def format_number(num, decimal_places=2):
    """Format number for display."""
    if pd.isna(num):
        return 'N/A'
    if abs(num) >= 1e6:
        return f"{num / 1e6:.{decimal_places}f}M"
    if abs(num) >= 1e3:
        return f"{num / 1e3:.{decimal_places}f}K"
    return f"{num:.{decimal_places}f}"


def calculate_year_over_year_change(df, column, year_col='year'):
    """Calculate year-over-year percentage change."""
    df = df.sort_values(year_col)
    df[f'{column}_yoy_change'] = df[column].pct_change() * 100
    return df


def get_bias_severity(score, thresholds=None):
    """Categorize bias score into severity levels."""
    if thresholds is None:
        thresholds = {'low': 0.3, 'medium': 0.6}

    if pd.isna(score):
        return 'unknown'
    if score <= thresholds['low']:
        return 'low'
    if score <= thresholds['medium']:
        return 'medium'
    return 'high'


def print_summary_stats(df, bias_types):
    """Print summary statistics for bias data."""
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)

    if 'year' in df.columns:
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"Total years: {df['year'].nunique()}")

    if 'article_count' in df.columns:
        print(f"Total articles: {df['article_count'].sum():,}")

    print("\nBias Score Averages:")
    for bias_type in bias_types:
        col = f'{bias_type}_mean'
        if col in df.columns:
            avg = df[col].mean()
            print(f"  {bias_type}: {avg:.3f}")

    print("=" * 50)
