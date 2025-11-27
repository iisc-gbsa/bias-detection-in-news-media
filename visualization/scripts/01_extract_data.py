"""
Extract data from MongoDB for bias visualization.
Pulls articles with bias scores from realtime_news collection.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pymongo
from dateutil import parser as dateparser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config, ensure_directories  # noqa: E402)


class DataExtractor:
    """Extract and clean article data from MongoDB."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "viz_config.yaml"
            )
        self.config = load_config(config_path)
        self.client = None
        self.db = None
        self.collection = None
        self.bias_types = self.config['bias_types']

    def connect_mongodb(self):
        """Establish MongoDB connection."""
        try:
            self.client = pymongo.MongoClient(self.config['mongodb']['uri'])
            self.db = self.client[self.config['mongodb']['database']]
            self.collection = self.db[self.config['mongodb']['collection']]
            print(f"Connected to MongoDB: {self.config['mongodb']['database']}.{self.config['mongodb']['collection']}")
            return True
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return False

    def extract_articles(self, batch_size=10000):
        """Extract articles from MongoDB with bias scores."""
        if not self.connect_mongodb():
            return None

        # Define the fields we need (matching MongoDB schema from screenshot)
        projection = {
            '_id': 0,
            'url': 1,  # Added url for joining with test.articles
            'published_date': 1,
            'title': 1,
            'media_name': 1,
            'topic': 1,
            'source': 1,
            # Bias scores (numerical)
            'political_bias': 1,
            'gender_bias': 1,
            'religious_bias': 1,
            'caste_bias': 1,
            'region_bias': 1,
            # Bias classifications
            'political_type': 1,
            'gender_type': 1,
            'religious_type': 1,
            'caste_type': 1,
            'region_type': 1,
        }

        print("Extracting articles from MongoDB...")
        articles = []

        try:
            # Debug: List all collections in the database
            collections = self.db.list_collection_names()
            print(f"Available collections: {collections}")

            # Get count
            total_count = self.collection.count_documents({})
            print(f"Total articles to process: {total_count}")

            if total_count == 0:
                # Try to get a sample document to understand the structure
                sample = self.collection.find_one()
                if sample:
                    print(f"Sample document keys: {list(sample.keys())}")
                else:
                    print("Collection is empty or doesn't exist")
                    return pd.DataFrame()

            # Get all articles
            cursor = self.collection.find({}, projection).batch_size(batch_size)

            for i, doc in enumerate(cursor):
                if i % 10000 == 0 and i > 0:
                    print(f"Processed {i}/{total_count} articles...")

                # Extract article data
                article = {
                    'url': doc.get('url', ''),
                    'published_date': doc.get('published_date'),
                    'title': doc.get('title', ''),
                    'media_name': doc.get('media_name', 'Unknown'),
                    'topic': doc.get('topic', 'General'),
                    'source': doc.get('source', 'Unknown'),
                    # Bias scores
                    'political_bias': doc.get('political_bias', 0),
                    'gender_bias': doc.get('gender_bias', 0),
                    'religious_bias': doc.get('religious_bias', 0),
                    'caste_bias': doc.get('caste_bias', 0),
                    'region_bias': doc.get('region_bias', 0),
                    # Bias types
                    'political_type': doc.get('political_type', 'neutral'),
                    'gender_type': doc.get('gender_type', 'neutral'),
                    'religious_type': doc.get('religious_type', 'neutral'),
                    'caste_type': doc.get('caste_type', 'neutral'),
                    'region_type': doc.get('region_type', 'neutral'),
                }
                articles.append(article)

        except Exception as e:
            print(f"Error extracting data: {e}")
            import traceback
            traceback.print_exc()
            if self.client:
                self.client.close()
            return None

        # Convert to DataFrame
        df = pd.DataFrame(articles)
        print(f"Extracted {len(df)} articles")

        # Backfill missing published_dates from test.articles collection
        # (connection still open at this point)
        df = self.backfill_dates_from_articles(df)

        # Close connection after backfill
        if self.client:
            self.client.close()

        # Handle empty dataframe
        if len(df) == 0:
            print("No articles found in the collection!")
            return df

        # Data cleaning and validation
        df = self.clean_data(df)

        print(f"After cleaning: {len(df)} articles")
        return df

    def backfill_dates_from_articles(self, df):
        """
        Backfill missing published_date from test.articles collection using URL matching.
        Articles in realtime_news may have published_date as "None" string,
        but the actual date exists as scrape_date in test.articles.
        """
        print("\nBackfilling missing dates from test.articles collection...")

        # Identify rows with missing or invalid dates
        # Check for None, "None" string, NaT, or empty strings
        missing_mask = (
            df['published_date'].isna()
            | (df['published_date'] == 'None')
            | (df['published_date'] == '')
            | (df['published_date'].astype(str) == 'None')
        )

        missing_count = missing_mask.sum()
        print(f"Found {missing_count} articles with missing/invalid dates")

        # Debug: Show missing dates per media
        if missing_count > 0:
            print("\n--- Missing dates breakdown by media ---")
            for media in df['media_name'].unique():
                media_missing = (df['media_name'] == media) & missing_mask
                count = media_missing.sum()
                total = (df['media_name'] == media).sum()
                if count > 0:
                    print(f"  {media}: {count}/{total} missing ({count / total * 100:.1f}%)")

        if missing_count == 0:
            return df

        # Get URLs of articles with missing dates
        missing_urls = df.loc[missing_mask, 'url'].tolist()
        print(f"\nLooking up dates for {len(missing_urls)} URLs in test.articles...")

        try:
            # Access test.articles collection
            articles_collection = self.client['test']['articles']

            # Fetch scrape_date for matching URLs in batches
            date_map = {}
            batch_size = 1000

            for i in range(0, len(missing_urls), batch_size):
                batch_urls = missing_urls[i:i + batch_size]

                # Query test.articles for matching URLs
                cursor = articles_collection.find(
                    {'url': {'$in': batch_urls}},
                    {'url': 1, 'scrape_date': 1, '_id': 0}
                )

                for doc in cursor:
                    if doc.get('scrape_date'):
                        date_map[doc['url']] = doc['scrape_date']

                if (i + batch_size) % 10000 == 0:
                    print(f"  Processed {i + batch_size}/{len(missing_urls)} URLs...")

            print(f"Found {len(date_map)} matching dates in test.articles")

            # Update DataFrame with backfilled dates
            if len(date_map) > 0:
                df.loc[missing_mask, 'published_date'] = df.loc[missing_mask, 'url'].map(date_map)

                # Count how many were successfully backfilled
                still_missing = (
                    df['published_date'].isna()
                    | (df['published_date'] == 'None')
                    | (df['published_date'] == '')
                    | (df['published_date'].astype(str) == 'None')
                ).sum()

                backfilled = missing_count - still_missing
                print(f"Successfully backfilled {backfilled} dates from test.articles")
                print(f"Still missing: {still_missing} dates")
            else:
                print("Warning: No matching dates found in test.articles collection")

        except Exception as e:
            print(f"Error during date backfill: {e}")
            print("Continuing with original dates...")

        return df

    def clean_data(self, df):
        """Clean and validate the extracted data."""
        print("Cleaning extracted data...")

        # Debug: Show article counts per media BEFORE datetime parsing
        print("\n--- Articles per media BEFORE datetime parsing ---")
        for media, count in df['media_name'].value_counts().items():
            print(f"  {media}: {count} articles")

        # Debug: Show sample dates per media before parsing
        print("\n--- Sample dates per media (before parsing) ---")
        for media in df['media_name'].unique():
            media_df = df[df['media_name'] == media]
            sample_dates = media_df['published_date'].dropna().head(5).tolist()
            print(f"  {media}: {sample_dates}")
            # Check types
            if len(sample_dates) > 0:
                print(f"    Type: {type(sample_dates[0])}")

        # Store original dates before parsing to debug failures
        df['_original_date'] = df['published_date']

        # Parse published_date (handle various formats including ISO 8601 with timezone)
        # Convert to string first to handle mixed types (datetime objects and strings)
        df['published_date'] = df['published_date'].astype(str)

        # Use dateutil.parser for robust parsing of timezone-aware and simple dates
        def parse_date_flexible(date_str):
            """Parse date string handling timezones, returning only the date part"""
            if not date_str or date_str == 'nan' or date_str == 'None' or date_str == '':
                return pd.NaT
            try:
                # dateutil.parser handles various formats including ISO 8601 with timezone
                dt = dateparser.parse(date_str)
                if dt:
                    # Return only the date part (YYYY-MM-DD), ignore time and timezone
                    return dt.date()
                return pd.NaT
            except (ValueError, TypeError, AttributeError):
                return pd.NaT

        print("Parsing dates with timezone support...")
        df['published_date'] = df['published_date'].apply(parse_date_flexible)
        df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')

        # Debug: Show which media have invalid dates after parsing
        print("\n--- Invalid dates per media AFTER datetime parsing ---")
        invalid_mask = df['published_date'].isna()
        if invalid_mask.sum() > 0:
            for media in df['media_name'].unique():
                media_df = df[df['media_name'] == media]
                invalid_count = media_df['published_date'].isna().sum()
                total = len(media_df)
                if invalid_count > 0:
                    print(f"  {media}: {invalid_count}/{total} invalid ({invalid_count / total * 100:.1f}%)")
                    # Show sample of failed dates for this media
                    failed_dates = media_df[media_df['published_date'].isna()]['_original_date'].head(10).tolist()
                    print(f"    Sample failed dates: {failed_dates}")

        # Remove rows with missing dates
        initial_count = len(df)
        df = df.dropna(subset=['published_date']).copy()
        print(f"\nRemoved {initial_count - len(df)} rows with missing dates")

        # Drop temporary debug column
        if '_original_date' in df.columns:
            df = df.drop(columns=['_original_date'])

        # Extract year from published_date
        df['year'] = df['published_date'].dt.year

        # Filter by date range from config
        start_year = int(self.config['date_range']['start'][:4])
        end_year = int(self.config['date_range']['end'][:4])
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
        print(f"Filtered to years {start_year}-{end_year}: {len(df)} articles")

        # Validate and clip bias scores (should be 0-1)
        for bias_type in self.bias_types:
            if bias_type in df.columns:
                df[bias_type] = pd.to_numeric(df[bias_type], errors='coerce').fillna(0)
                df[bias_type] = np.clip(df[bias_type], 0, 1)

        # Clean string columns
        df['media_name'] = df['media_name'].fillna('Unknown').astype(str)
        df['topic'] = df['topic'].fillna('General').astype(str)

        # Calculate overall bias score (average of all bias types)
        bias_cols = [c for c in self.bias_types if c in df.columns]
        df['overall_bias'] = df[bias_cols].mean(axis=1)

        # Add high bias indicators
        # Use value from config if available, otherwise default to 0.0 (any non-zero bias counts as high)
        filters_cfg = self.config.get('filters', {}) if isinstance(self.config, dict) else {}
        threshold = filters_cfg.get('high_bias_threshold', 0.0)
        for bias_type in bias_cols:
            df[f'{bias_type}_high'] = (df[bias_type] > threshold).astype(int)

        return df

    def save_raw_data(self, df, filename="raw_articles.csv"):
        """Save raw extracted data to CSV."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(data_dir, exist_ok=True)

        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Raw data saved to {filepath}")

        # Save extraction statistics
        bias_cols = [c for c in self.bias_types if c in df.columns]
        stats = {
            'total_articles': len(df),
            'date_range': f"{df['year'].min()} to {df['year'].max()}",
            'years_covered': sorted(df['year'].unique().tolist()),
            'unique_sources': df['media_name'].nunique(),
            'sources': df['media_name'].value_counts().to_dict(),
            'articles_per_year': df.groupby('year').size().to_dict(),
            'bias_stats': {
                col: {
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                } for col in bias_cols
            }
        }

        stats_path = os.path.join(data_dir, "extraction_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"Extraction stats saved to {stats_path}")

        return filepath


def main():
    """Main entry point for data extraction."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure directories exist
    config = load_config(os.path.join(base_dir, "config", "viz_config.yaml"))
    ensure_directories(os.path.join(base_dir, config['output']['base_dir']))

    # Extract data
    extractor = DataExtractor()
    df = extractor.extract_articles()

    if df is not None and len(df) > 0:
        extractor.save_raw_data(df)
        print("\n" + "=" * 50)
        print("DATA EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Total articles: {len(df)}")
        print(f"Year range: {df['year'].min()} - {df['year'].max()}")
        print(f"Media sources: {df['media_name'].nunique()}")
    else:
        print("Data extraction failed or no data found!")


if __name__ == "__main__":
    main()
