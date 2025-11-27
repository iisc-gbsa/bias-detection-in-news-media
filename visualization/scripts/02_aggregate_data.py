"""
Aggregate article data for bias visualization.
Creates yearly aggregations with 5-year moving averages.
Supports both overall and media-specific analysis.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config  # noqa: E402


class DataAggregator:
    """Aggregate article bias data at yearly level with 5-year MA."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "viz_config.yaml"
            )
        self.config = load_config(config_path)
        self.bias_types = self.config['bias_types']
        self.ma_window = self.config['moving_averages']['windows'][0]  # 5 years

    def load_raw_data(self, filepath=None):
        """Load raw data from CSV."""
        if filepath is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, "data", "raw", "raw_articles.csv")

        print(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        df['published_date'] = pd.to_datetime(df['published_date'])
        df['year'] = df['published_date'].dt.year
        print(f"Loaded {len(df)} articles from {df['year'].min()} to {df['year'].max()}")
        return df

    def aggregate_yearly(self, df):
        """Create yearly aggregations with comprehensive metrics."""
        print("Creating yearly aggregations...")

        yearly_agg = []

        for year, group in df.groupby('year'):
            row = {
                'year': int(year),
                'article_count': len(group)
            }

            # Calculate metrics for each bias type
            for bias_type in self.bias_types:
                if bias_type in group.columns:
                    scores = group[bias_type].dropna()
                    if len(scores) > 0:
                        row[f'{bias_type}_mean'] = scores.mean()
                        row[f'{bias_type}_median'] = scores.median()
                        row[f'{bias_type}_std'] = scores.std()
                        row[f'{bias_type}_min'] = scores.min()
                        row[f'{bias_type}_max'] = scores.max()
                        row[f'{bias_type}_q25'] = scores.quantile(0.25)
                        row[f'{bias_type}_q75'] = scores.quantile(0.75)
                        # High bias percentage
                        filters_cfg = self.config.get('filters', {}) if isinstance(self.config, dict) else {}
                        # Default 0.0 means any non-zero bias is treated as "high" unless overridden in config
                        threshold = filters_cfg.get('high_bias_threshold', 0.0)
                        row[f'{bias_type}_high_count'] = int((scores > threshold).sum())
                        row[f'{bias_type}_high_pct'] = (scores > threshold).mean() * 100
                    else:
                        for metric in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']:
                            row[f'{bias_type}_{metric}'] = np.nan
                        row[f'{bias_type}_high_count'] = 0
                        row[f'{bias_type}_high_pct'] = 0

            # Overall bias
            if 'overall_bias' in group.columns:
                scores = group['overall_bias'].dropna()
                if len(scores) > 0:
                    row['overall_bias_mean'] = scores.mean()
                    row['overall_bias_median'] = scores.median()

            yearly_agg.append(row)

        yearly_df = pd.DataFrame(yearly_agg)
        yearly_df = yearly_df.sort_values('year').reset_index(drop=True)

        # Add 5-year moving averages
        yearly_df = self._add_moving_averages(yearly_df)

        return yearly_df

    def _add_moving_averages(self, yearly_df):
        """Add 5-year moving averages to yearly data."""
        print(f"Calculating {self.ma_window}-year moving averages...")

        for bias_type in self.bias_types:
            col_name = f'{bias_type}_mean'
            if col_name in yearly_df.columns:
                yearly_df[f'{bias_type}_ma_5y'] = yearly_df[col_name].rolling(
                    window=self.ma_window, min_periods=1, center=False
                ).mean()

        # Overall bias MA
        if 'overall_bias_mean' in yearly_df.columns:
            yearly_df['overall_bias_ma_5y'] = yearly_df['overall_bias_mean'].rolling(
                window=self.ma_window, min_periods=1, center=False
            ).mean()

        return yearly_df

    def aggregate_by_media(self, df):
        """Create media source aggregations - overall."""
        print("Creating media source aggregations...")

        media_agg = []

        # Use min_articles from config if available, else default to 1 (include all media)
        filters_cfg = self.config.get('filters', {}) if isinstance(self.config, dict) else {}
        min_articles = filters_cfg.get('min_articles', 1)

        for media, group in df.groupby('media_name'):
            if len(group) >= min_articles:
                row = {
                    'media_name': media,
                    'article_count': len(group),
                    'year_start': int(group['year'].min()),
                    'year_end': int(group['year'].max()),
                    'years_active': group['year'].nunique()
                }

                for bias_type in self.bias_types:
                    if bias_type in group.columns:
                        scores = group[bias_type].dropna()
                        if len(scores) > 0:
                            row[f'{bias_type}_mean'] = scores.mean()
                            row[f'{bias_type}_median'] = scores.median()
                            row[f'{bias_type}_std'] = scores.std()
                            threshold = self.config['filters']['high_bias_threshold']
                            row[f'{bias_type}_high_pct'] = (scores > threshold).mean() * 100

                if 'overall_bias' in group.columns:
                    scores = group['overall_bias'].dropna()
                    if len(scores) > 0:
                        row['overall_bias_mean'] = scores.mean()

                media_agg.append(row)

        media_df = pd.DataFrame(media_agg)
        media_df = media_df.sort_values('article_count', ascending=False).reset_index(drop=True)
        return media_df

    def aggregate_by_media_yearly(self, df):
        """Create media-year level aggregations for detailed trends."""
        print("Creating media-yearly aggregations...")

        media_year_agg = []

        for (media, year), group in df.groupby(['media_name', 'year']):
            if len(group) >= 1:  # Include all years with at least 1 article
                row = {
                    'media_name': media,
                    'year': int(year),
                    'article_count': len(group)
                }

                for bias_type in self.bias_types:
                    if bias_type in group.columns:
                        scores = group[bias_type].dropna()
                        if len(scores) > 0:
                            row[f'{bias_type}_mean'] = scores.mean()
                            row[f'{bias_type}_median'] = scores.median()
                            threshold = self.config['filters']['high_bias_threshold']
                            row[f'{bias_type}_high_pct'] = (scores > threshold).mean() * 100

                if 'overall_bias' in group.columns:
                    scores = group['overall_bias'].dropna()
                    if len(scores) > 0:
                        row['overall_bias_mean'] = scores.mean()

                media_year_agg.append(row)

        media_year_df = pd.DataFrame(media_year_agg)
        media_year_df = media_year_df.sort_values(['media_name', 'year']).reset_index(drop=True)

        # Add 5-year MA per media
        media_year_df = self._add_media_moving_averages(media_year_df)

        return media_year_df

    def _add_media_moving_averages(self, df):
        """Add 5-year moving averages per media source."""
        print("Calculating per-media 5-year moving averages...")

        result_dfs = []

        for media in df['media_name'].unique():
            media_df = df[df['media_name'] == media].copy()
            media_df = media_df.sort_values('year')

            for bias_type in self.bias_types:
                col_name = f'{bias_type}_mean'
                if col_name in media_df.columns:
                    media_df[f'{bias_type}_ma_5y'] = media_df[col_name].rolling(
                        window=self.ma_window, min_periods=1, center=False
                    ).mean()

            if 'overall_bias_mean' in media_df.columns:
                media_df['overall_bias_ma_5y'] = media_df['overall_bias_mean'].rolling(
                    window=self.ma_window, min_periods=1, center=False
                ).mean()

            result_dfs.append(media_df)

        return pd.concat(result_dfs, ignore_index=True)

    def aggregate_bias_types(self, df):
        """Aggregate by bias type classifications (political_type, gender_type, etc.)."""
        print("Creating bias type classification aggregations...")

        type_agg = {}

        # Political type distribution by year
        if 'political_type' in df.columns:
            type_agg['political_type'] = df.groupby(['year', 'political_type']).size().unstack(fill_value=0)
            type_agg['political_type']['total'] = type_agg['political_type'].sum(axis=1)
            for col in type_agg['political_type'].columns:
                if col != 'total':
                    type_agg['political_type'][f'{col}_pct'] = (
                        type_agg['political_type'][col] / type_agg['political_type']['total'] * 100
                    )

        # Gender type distribution by year
        if 'gender_type' in df.columns:
            type_agg['gender_type'] = df.groupby(['year', 'gender_type']).size().unstack(fill_value=0)
            type_agg['gender_type']['total'] = type_agg['gender_type'].sum(axis=1)
            for col in type_agg['gender_type'].columns:
                if col != 'total':
                    type_agg['gender_type'][f'{col}_pct'] = (
                        type_agg['gender_type'][col] / type_agg['gender_type']['total'] * 100
                    )

        # Religious type distribution
        if 'religious_type' in df.columns:
            type_agg['religious_type'] = df.groupby(['year', 'religious_type']).size().unstack(fill_value=0)

        # Caste type distribution
        if 'caste_type' in df.columns:
            type_agg['caste_type'] = df.groupby(['year', 'caste_type']).size().unstack(fill_value=0)

        # Region type distribution
        if 'region_type' in df.columns:
            type_agg['region_type'] = df.groupby(['year', 'region_type']).size().unstack(fill_value=0)

        return type_agg

    def save_aggregations(self, aggregations, base_dir=None):
        """Save all aggregations to CSV files."""
        if base_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        agg_dir = os.path.join(base_dir, "data", "aggregated")
        os.makedirs(agg_dir, exist_ok=True)

        for name, data in aggregations.items():
            filepath = os.path.join(agg_dir, f"{name}.csv")
            if isinstance(data, pd.DataFrame):
                data.to_csv(filepath, index=False)
                print(f"Saved {name} to {filepath} ({len(data)} rows)")
            elif isinstance(data, dict):
                # For nested dicts (like bias type aggregations)
                for subname, subdf in data.items():
                    subpath = os.path.join(agg_dir, f"{name}_{subname}.csv")
                    if isinstance(subdf, pd.DataFrame):
                        subdf.to_csv(subpath)
                        print(f"Saved {name}_{subname} to {subpath}")


def main():
    """Main entry point for data aggregation."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    aggregator = DataAggregator()

    # Load raw data
    df = aggregator.load_raw_data()
    print(f"Loaded {len(df)} articles for aggregation")

    # Create aggregations
    aggregations = {
        'yearly_aggregations': aggregator.aggregate_yearly(df),
        'media_aggregations': aggregator.aggregate_by_media(df),
        'media_yearly_aggregations': aggregator.aggregate_by_media_yearly(df),
    }

    # Bias type distributions
    type_agg = aggregator.aggregate_bias_types(df)
    aggregations['bias_type_distributions'] = type_agg

    # Save aggregations
    aggregator.save_aggregations(aggregations, base_dir)

    print("\n" + "=" * 50)
    print("AGGREGATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"Years covered: {df['year'].min()} - {df['year'].max()}")
    print(f"Total years: {df['year'].nunique()}")
    print(f"Media sources: {df['media_name'].nunique()}")


if __name__ == "__main__":
    main()
