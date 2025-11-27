"""
Word Count vs Article Count Distribution Analysis.
Creates visualizations showing the distribution of articles by word count.
"""

import pymongo
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config  # noqa: E402


class WordArticleAnalyzer:
    """Analyze and visualize word count distribution across articles."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "viz_config.yaml"
            )
        self.config = load_config(config_path)
        self.client = None
        self.db = None

        # Output directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, self.config['output']['base_dir'], 'word_distribution')
        os.makedirs(self.output_dir, exist_ok=True)

        # Data directory
        self.data_dir = os.path.join(self.base_dir, "data", "aggregated")
        os.makedirs(self.data_dir, exist_ok=True)

        # Set up plotting style
        self._setup_style()

    def _setup_style(self):
        """Set up matplotlib/seaborn styling."""
        sns.set_theme(style="whitegrid", palette="deep")
        style = self.config.get('plot_style', {})
        plt.rcParams['figure.figsize'] = style.get('figure_size', [14, 8])
        plt.rcParams['font.size'] = style.get('font_size', 12)
        plt.rcParams['axes.titlesize'] = style.get('title_size', 16)
        plt.rcParams['axes.labelsize'] = style.get('label_size', 14)
        plt.rcParams['lines.linewidth'] = style.get('line_width', 2)
        plt.rcParams['lines.markersize'] = style.get('marker_size', 8)

    def connect_mongodb(self):
        """Establish MongoDB connection."""
        try:
            self.client = pymongo.MongoClient(self.config['mongodb']['uri'])
            self.db = self.client[self.config['mongodb']['database']]
            print(f"Connected to MongoDB: {self.config['mongodb']['database']}")
            return True
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            return False

    def extract_word_counts(self, batch_size=10000):
        """
        Extract word counts from articles.
        Tries to get word_count field, or calculates from content/text field.
        """
        if not self.connect_mongodb():
            return None

        collection = self.db[self.config['mongodb']['collection']]

        # First, check what fields are available
        sample = collection.find_one()
        if sample:
            print(f"Available fields: {list(sample.keys())}")

        # Try to find word count field or content field
        projection = {
            '_id': 0,
            'url': 1,
            'title': 1,
            'media_name': 1,
            'published_date': 1,
            'word_count': 1,
            'content': 1,
            'text': 1,
            'article_text': 1,
            'body': 1,
        }

        print("Extracting word counts from MongoDB...")
        data = []

        try:
            total_count = collection.count_documents({})
            print(f"Total articles to process: {total_count}")

            cursor = collection.find({}, projection).batch_size(batch_size)

            for i, doc in enumerate(cursor):
                if i % 10000 == 0 and i > 0:
                    print(f"Processed {i}/{total_count} articles...")

                # Try to get word count
                word_count = doc.get('word_count')

                # If no word_count field, calculate from content
                if word_count is None:
                    content = (
                        doc.get('content')
                        or doc.get('text')
                        or doc.get('article_text')
                        or doc.get('body')
                        or ''
                    )
                    if content:
                        word_count = len(str(content).split())
                    else:
                        # Use title word count as fallback
                        title = doc.get('title', '')
                        word_count = len(str(title).split()) if title else 0

                data.append({
                    'url': doc.get('url', ''),
                    'title': doc.get('title', ''),
                    'media_name': doc.get('media_name', 'Unknown'),
                    'published_date': doc.get('published_date'),
                    'word_count': word_count
                })

        except Exception as e:
            print(f"Error extracting data: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.client:
                self.client.close()

        df = pd.DataFrame(data)
        print(f"Extracted {len(df)} articles with word counts")

        # Filter out zero word counts
        df = df[df['word_count'] > 0].copy()
        print(f"After filtering zero word counts: {len(df)} articles")

        return df

    def load_from_csv(self, filepath=None):
        """Load word count data from CSV if already extracted."""
        if filepath is None:
            filepath = os.path.join(self.data_dir, "word_counts.csv")

        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"Loaded {len(df)} articles from {filepath}")
            return df
        return None

    def save_data(self, df, filename="word_counts.csv"):
        """Save extracted data to CSV."""
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved word count data to {filepath}")
        return filepath

    def aggregate_word_distribution(self, df, bin_size=100):
        """
        Aggregate articles by word count bins.

        Args:
            df: DataFrame with word_count column
            bin_size: Size of each word count bin (default 100 words)

        Returns:
            DataFrame with word count bins and article counts
        """
        # Create word count bins
        max_words = df['word_count'].max()
        bins = list(range(0, int(max_words) + bin_size, bin_size))

        df['word_bin'] = pd.cut(df['word_count'], bins=bins, labels=bins[:-1])
        df['word_bin'] = df['word_bin'].astype(float)

        # Aggregate by bin
        agg_df = df.groupby('word_bin').agg(
            article_count=('word_count', 'count'),
            avg_word_count=('word_count', 'mean'),
            min_word_count=('word_count', 'min'),
            max_word_count=('word_count', 'max')
        ).reset_index()

        return agg_df

    def _save_figure(self, fig, name):
        """Save figure in configured formats."""
        for fmt in self.config['output'].get('formats', ['png']):
            filepath = os.path.join(self.output_dir, f"{name}.{fmt}")
            fig.savefig(filepath, dpi=self.config['output'].get('dpi', 300),
                        bbox_inches='tight', facecolor='white')
        print(f"Saved: {name}")
        plt.close(fig)

    def plot_word_article_histogram(self, df, max_words=5000):
        """
        Create histogram showing article count by word count.

        Args:
            df: DataFrame with word_count column
            max_words: Maximum word count to display (for better visualization)
        """
        fig, ax = plt.subplots(figsize=(16, 9))

        # Filter to max_words for better visualization
        plot_df = df[df['word_count'] <= max_words].copy()

        # Create histogram
        bins = np.arange(0, max_words + 100, 100)
        counts, edges, patches = ax.hist(
            plot_df['word_count'],
            bins=bins,
            color='steelblue',
            alpha=0.7,
            edgecolor='navy',
            linewidth=0.5
        )

        # Add statistics
        mean_words = df['word_count'].mean()
        median_words = df['word_count'].median()

        ax.axvline(mean_words, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_words:.0f} words')
        ax.axvline(median_words, color='green', linestyle='--', linewidth=2, label=f'Median: {median_words:.0f} words')

        ax.set_xlabel('Word Count', fontsize=14)
        ax.set_ylabel('Number of Articles', fontsize=14)
        ax.set_title('Distribution of Articles by Word Count', fontsize=16)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        self._save_figure(fig, 'word_article_histogram')

    def plot_word_article_line(self, df, bin_size=100, max_words=5000):
        """
        Create line chart showing article count by word count bins.

        Args:
            df: DataFrame with word_count column
            bin_size: Size of word count bins
            max_words: Maximum word count to display
        """
        # Filter and aggregate
        plot_df = df[df['word_count'] <= max_words].copy()
        agg_df = self.aggregate_word_distribution(plot_df, bin_size=bin_size)

        fig, ax = plt.subplots(figsize=(16, 9))

        ax.plot(
            agg_df['word_bin'],
            agg_df['article_count'],
            color='steelblue',
            linewidth=2.5,
            marker='o',
            markersize=6,
            markerfacecolor='white',
            markeredgecolor='steelblue',
            markeredgewidth=2
        )

        # Fill under the curve
        ax.fill_between(
            agg_df['word_bin'],
            agg_df['article_count'],
            alpha=0.3,
            color='steelblue'
        )

        ax.set_xlabel('Word Count (bin start)', fontsize=14)
        ax.set_ylabel('Number of Articles', fontsize=14)
        ax.set_title(f'Articles by Word Count ({bin_size}-word bins)', fontsize=16)
        ax.grid(True, alpha=0.3)

        # Format y-axis with thousands separator
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))

        self._save_figure(fig, 'word_article_line')

    def plot_word_article_cumulative(self, df, max_words=5000):
        """
        Create cumulative distribution chart.

        Args:
            df: DataFrame with word_count column
            max_words: Maximum word count to display
        """
        plot_df = df[df['word_count'] <= max_words].copy()

        fig, ax = plt.subplots(figsize=(16, 9))

        # Sort and calculate cumulative percentage
        sorted_counts = np.sort(plot_df['word_count'])
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100

        ax.plot(sorted_counts, cumulative, color='steelblue', linewidth=2)

        # Add reference lines
        percentiles = [25, 50, 75, 90]
        colors = ['green', 'orange', 'red', 'purple']
        for pct, color in zip(percentiles, colors):
            word_at_pct = np.percentile(plot_df['word_count'], pct)
            ax.axhline(pct, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(word_at_pct, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax.annotate(f'{pct}%: {word_at_pct:.0f} words',
                        xy=(word_at_pct, pct),
                        xytext=(word_at_pct + 100, pct + 3),
                        fontsize=10, color=color)

        ax.set_xlabel('Word Count', fontsize=14)
        ax.set_ylabel('Cumulative Percentage of Articles (%)', fontsize=14)
        ax.set_title('Cumulative Distribution of Articles by Word Count', fontsize=16)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        self._save_figure(fig, 'word_article_cumulative')

    def plot_word_distribution_by_media(self, df, top_n=10):
        """
        Create box plot showing word count distribution by media source.

        Args:
            df: DataFrame with word_count and media_name columns
            top_n: Number of top media sources to include
        """
        # Get top N media by article count
        top_media = df['media_name'].value_counts().head(top_n).index.tolist()
        plot_df = df[df['media_name'].isin(top_media)].copy()

        fig, ax = plt.subplots(figsize=(16, 10))

        # Create box plot
        media_order = plot_df.groupby('media_name')['word_count'].median().sort_values(ascending=False).index
        sns.boxplot(
            data=plot_df,
            x='media_name',
            y='word_count',
            order=media_order,
            palette='Set2',
            ax=ax,
            showfliers=False  # Hide outliers for cleaner visualization
        )

        ax.set_xlabel('Media Source', fontsize=14)
        ax.set_ylabel('Word Count', fontsize=14)
        ax.set_title(f'Word Count Distribution by Media Source (Top {top_n})', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        self._save_figure(fig, 'word_distribution_by_media')

    def plot_word_count_summary(self, df):
        """
        Create a summary dashboard with multiple views.

        Args:
            df: DataFrame with word_count column
        """
        fig = plt.figure(figsize=(20, 14))

        # 1. Histogram (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        max_words = min(5000, df['word_count'].quantile(0.99))
        plot_df = df[df['word_count'] <= max_words]
        ax1.hist(plot_df['word_count'], bins=50, color='steelblue', alpha=0.7, edgecolor='navy')
        ax1.axvline(df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {df['word_count'].mean():.0f}")
        ax1.axvline(df['word_count'].median(), color='green', linestyle='--', label=f"Median: {df['word_count'].median():.0f}")
        ax1.set_title('Word Count Distribution', fontsize=14)
        ax1.set_xlabel('Word Count')
        ax1.set_ylabel('Number of Articles')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Log-scale histogram (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.hist(df['word_count'], bins=100, color='coral', alpha=0.7, edgecolor='darkred')
        ax2.set_yscale('log')
        ax2.set_title('Word Count Distribution (Log Scale)', fontsize=14)
        ax2.set_xlabel('Word Count')
        ax2.set_ylabel('Number of Articles (log)')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Statistics table (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.axis('off')

        stats = {
            'Total Articles': f"{len(df):,}",
            'Mean Word Count': f"{df['word_count'].mean():.1f}",
            'Median Word Count': f"{df['word_count'].median():.1f}",
            'Std Dev': f"{df['word_count'].std():.1f}",
            'Min Word Count': f"{df['word_count'].min():,}",
            'Max Word Count': f"{df['word_count'].max():,}",
            '25th Percentile': f"{df['word_count'].quantile(0.25):.0f}",
            '75th Percentile': f"{df['word_count'].quantile(0.75):.0f}",
            '90th Percentile': f"{df['word_count'].quantile(0.90):.0f}",
            '99th Percentile': f"{df['word_count'].quantile(0.99):.0f}",
        }

        table_data = [[k, v] for k, v in stats.items()]
        table = ax3.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            loc='center',
            cellLoc='left',
            colWidths=[0.5, 0.3]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        ax3.set_title('Word Count Statistics', fontsize=14, pad=20)

        # 4. Word count ranges pie chart (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        ranges = [
            ('0-100', (df['word_count'] <= 100).sum()),
            ('101-300', ((df['word_count'] > 100) & (df['word_count'] <= 300)).sum()),
            ('301-500', ((df['word_count'] > 300) & (df['word_count'] <= 500)).sum()),
            ('501-1000', ((df['word_count'] > 500) & (df['word_count'] <= 1000)).sum()),
            ('1001-2000', ((df['word_count'] > 1000) & (df['word_count'] <= 2000)).sum()),
            ('2000+', (df['word_count'] > 2000).sum()),
        ]
        labels, sizes = zip(*ranges)
        colors = plt.cm.Set3(np.linspace(0, 1, len(ranges)))
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Articles by Word Count Range', fontsize=14)

        fig.suptitle('Word Count Analysis Summary', fontsize=18, y=1.02)
        fig.tight_layout()

        self._save_figure(fig, 'word_count_summary_dashboard')

    def generate_all_visualizations(self, df):
        """Generate all word count visualizations."""
        print("\nGenerating word count visualizations...")

        self.plot_word_article_histogram(df)
        self.plot_word_article_line(df)
        self.plot_word_article_cumulative(df)

        if 'media_name' in df.columns:
            self.plot_word_distribution_by_media(df)

        self.plot_word_count_summary(df)

        print(f"\nAll visualizations saved to: {self.output_dir}")

    def save_statistics(self, df, filename="word_count_stats.json"):
        """Save word count statistics to JSON."""
        stats = {
            'total_articles': int(len(df)),
            'mean_word_count': float(df['word_count'].mean()),
            'median_word_count': float(df['word_count'].median()),
            'std_word_count': float(df['word_count'].std()),
            'min_word_count': int(df['word_count'].min()),
            'max_word_count': int(df['word_count'].max()),
            'percentiles': {
                '10': float(df['word_count'].quantile(0.10)),
                '25': float(df['word_count'].quantile(0.25)),
                '50': float(df['word_count'].quantile(0.50)),
                '75': float(df['word_count'].quantile(0.75)),
                '90': float(df['word_count'].quantile(0.90)),
                '95': float(df['word_count'].quantile(0.95)),
                '99': float(df['word_count'].quantile(0.99)),
            },
            'word_count_ranges': {
                '0-100': int((df['word_count'] <= 100).sum()),
                '101-300': int(((df['word_count'] > 100) & (df['word_count'] <= 300)).sum()),
                '301-500': int(((df['word_count'] > 300) & (df['word_count'] <= 500)).sum()),
                '501-1000': int(((df['word_count'] > 500) & (df['word_count'] <= 1000)).sum()),
                '1001-2000': int(((df['word_count'] > 1000) & (df['word_count'] <= 2000)).sum()),
                '2000+': int((df['word_count'] > 2000).sum()),
            }
        }

        if 'media_name' in df.columns:
            stats['by_media'] = df.groupby('media_name')['word_count'].agg(
                ['count', 'mean', 'median', 'std', 'min', 'max']
            ).round(2).to_dict('index')

        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {filepath}")

        return stats


def main():
    """Main entry point for word count analysis."""
    analyzer = WordArticleAnalyzer()

    # Try to load existing data first
    df = analyzer.load_from_csv()

    if df is None:
        # Extract from MongoDB
        df = analyzer.extract_word_counts()

        if df is not None and len(df) > 0:
            analyzer.save_data(df)

    if df is not None and len(df) > 0:
        # Generate visualizations
        analyzer.generate_all_visualizations(df)

        # Save statistics
        analyzer.save_statistics(df)

        print("\n" + "=" * 50)
        print("WORD COUNT ANALYSIS COMPLETED!")
        print("=" * 50)
        print(f"Total articles analyzed: {len(df):,}")
        print(f"Mean word count: {df['word_count'].mean():.1f}")
        print(f"Median word count: {df['word_count'].median():.1f}")
        print(f"Output directory: {analyzer.output_dir}")
    else:
        print("No data available for analysis!")


if __name__ == "__main__":
    main()
