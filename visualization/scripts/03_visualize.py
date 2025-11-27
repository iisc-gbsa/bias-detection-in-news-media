"""
Visualize bias trends using Matplotlib and Seaborn.
Creates yearly trend charts with 5-year moving averages.
Supports both overall and media-specific visualizations.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config  # noqa: E402


class BiasVisualizer:
    """Create bias trend visualizations using matplotlib and seaborn."""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config", "viz_config.yaml"
            )
        self.config = load_config(config_path)
        self.bias_types = self.config['bias_types']
        self.colors = self.config['colors']
        self.labels = self.config['bias_type_labels']
        self.style = self.config['plot_style']

        # Set up plotting style
        self._setup_style()

        # Output directory
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, self.config['output']['base_dir'])
        self._ensure_output_dirs()

    def _setup_style(self):
        """Set up matplotlib/seaborn styling."""
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams['figure.figsize'] = self.style['figure_size']
        plt.rcParams['font.size'] = self.style['font_size']
        plt.rcParams['axes.titlesize'] = self.style['title_size']
        plt.rcParams['axes.labelsize'] = self.style['label_size']
        plt.rcParams['lines.linewidth'] = self.style['line_width']
        plt.rcParams['lines.markersize'] = self.style['marker_size']

    def _ensure_output_dirs(self):
        """Create output directories."""
        subdirs = ['time_series', 'heatmaps', 'distributions', 'comparisons', 'media']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def _save_figure(self, fig, name, subdir=''):
        """Save figure in configured formats."""
        for fmt in self.config['output']['formats']:
            filepath = os.path.join(self.output_dir, subdir, f"{name}.{fmt}")
            fig.savefig(filepath, dpi=self.config['output']['dpi'],
                        bbox_inches='tight', facecolor='white')
        print(f"Saved: {name}")
        plt.close(fig)

    def load_data(self):
        """Load all aggregated data."""
        data_dir = os.path.join(self.base_dir, "data", "aggregated")

        self.yearly_df = pd.read_csv(os.path.join(data_dir, "yearly_aggregations.csv"))
        self.media_df = pd.read_csv(os.path.join(data_dir, "media_aggregations.csv"))
        self.media_yearly_df = pd.read_csv(os.path.join(data_dir, "media_yearly_aggregations.csv"))

        # Load bias type distributions (categorical: left_leaning, right_leaning, etc.)
        self.bias_type_dfs = {}
        type_files = {
            'political_type': 'bias_type_distributions_political_type.csv',
            'gender_type': 'bias_type_distributions_gender_type.csv',
            'religious_type': 'bias_type_distributions_religious_type.csv',
            'caste_type': 'bias_type_distributions_caste_type.csv',
            'region_type': 'bias_type_distributions_region_type.csv',
        }

        for type_name, filename in type_files.items():
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                self.bias_type_dfs[type_name] = pd.read_csv(filepath, index_col=0)
                print(f"Loaded {type_name} distribution data")

        print(f"Loaded yearly data: {len(self.yearly_df)} years")
        print(f"Loaded media data: {len(self.media_df)} sources")
        print(f"Loaded media-yearly data: {len(self.media_yearly_df)} rows")
        print(f"Loaded {len(self.bias_type_dfs)} bias type distributions")

    # ========== TIME SERIES PLOTS ==========

    def plot_all_bias_trends(self):
        """Plot all bias types on a single chart with 5-year MA."""
        fig, ax = plt.subplots(figsize=(16, 9))

        for bias_type in self.bias_types:
            color = self.colors.get(bias_type, '#333333')
            label = self.labels.get(bias_type, bias_type)

            # Raw yearly values (lighter, thinner)
            if f'{bias_type}_mean' in self.yearly_df.columns:
                ax.plot(self.yearly_df['year'], self.yearly_df[f'{bias_type}_mean'],
                        color=color, alpha=0.3, linewidth=1, linestyle='--')

            # 5-year MA (bold)
            if f'{bias_type}_ma_5y' in self.yearly_df.columns:
                ax.plot(self.yearly_df['year'], self.yearly_df[f'{bias_type}_ma_5y'],
                        color=color, linewidth=2.5, marker='o', markersize=4,
                        label=f'{label} (5-yr MA)')

        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Bias Score (0-1)', fontsize=14)
        ax.set_title('Bias Trends Over Time (1997-2025)\nWith 5-Year Moving Average', fontsize=16)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        self._save_figure(fig, 'all_bias_trends_5y_ma', 'time_series')

    def plot_individual_bias_trends(self):
        """Plot each bias type separately with raw + 5-year MA."""
        for bias_type in self.bias_types:
            if f'{bias_type}_mean' not in self.yearly_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(14, 8))
            color = self.colors.get(bias_type, '#333333')
            label = self.labels.get(bias_type, bias_type)

            # Raw values
            ax.plot(self.yearly_df['year'], self.yearly_df[f'{bias_type}_mean'],
                    color=color, alpha=0.5, linewidth=1.5, linestyle='--',
                    marker='o', markersize=4, label='Yearly Average')

            # 5-year MA
            if f'{bias_type}_ma_5y' in self.yearly_df.columns:
                ax.plot(self.yearly_df['year'], self.yearly_df[f'{bias_type}_ma_5y'],
                        color=color, linewidth=3, marker='s', markersize=6,
                        label='5-Year Moving Average')

            # Fill between for confidence
            if f'{bias_type}_q25' in self.yearly_df.columns:
                ax.fill_between(self.yearly_df['year'],
                                self.yearly_df[f'{bias_type}_q25'],
                                self.yearly_df[f'{bias_type}_q75'],
                                color=color, alpha=0.1, label='IQR (25-75%)')

            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('Bias Score (0-1)', fontsize=14)
            ax.set_title(f'{label} Bias Trend (1997-2025)\nWith 5-Year Moving Average', fontsize=16)
            ax.legend(loc='upper left', fontsize=11)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

            self._save_figure(fig, f'{bias_type}_trend_5y_ma', 'time_series')

    def plot_bias_comparison_subplots(self):
        """Create a 2x3 subplot grid comparing all bias types."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, bias_type in enumerate(self.bias_types):
            if idx >= len(axes):
                break

            ax = axes[idx]
            color = self.colors.get(bias_type, '#333333')
            label = self.labels.get(bias_type, bias_type)

            if f'{bias_type}_mean' in self.yearly_df.columns:
                ax.plot(self.yearly_df['year'], self.yearly_df[f'{bias_type}_mean'],
                        color=color, alpha=0.4, linewidth=1, linestyle='--')

            if f'{bias_type}_ma_5y' in self.yearly_df.columns:
                ax.plot(self.yearly_df['year'], self.yearly_df[f'{bias_type}_ma_5y'],
                        color=color, linewidth=2.5, marker='o', markersize=3)

            ax.set_title(f'{label} Bias', fontsize=13)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

        # Hide unused subplot
        if len(self.bias_types) < len(axes):
            axes[-1].axis('off')

        fig.suptitle('Bias Trends by Category (5-Year Moving Average)', fontsize=16, y=1.02)
        fig.tight_layout()

        self._save_figure(fig, 'bias_comparison_grid', 'time_series')

    def plot_article_volume(self):
        """Plot article count over years."""
        fig, ax = plt.subplots(figsize=(14, 8))

        ax.bar(self.yearly_df['year'], self.yearly_df['article_count'],
               color='steelblue', alpha=0.7, edgecolor='navy')

        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Number of Articles', fontsize=14)
        ax.set_title('Article Volume by Year (1997-2025)', fontsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3, axis='y')

        # Add trend line
        z = np.polyfit(self.yearly_df['year'], self.yearly_df['article_count'], 1)
        p = np.poly1d(z)
        ax.plot(self.yearly_df['year'], p(self.yearly_df['year']),
                'r--', linewidth=2, label='Trend')
        ax.legend()

        self._save_figure(fig, 'article_volume_yearly', 'time_series')

    # ========== HEATMAPS ==========

    def plot_bias_heatmap(self):
        """Create a heatmap of bias scores over years."""
        # Prepare data for heatmap
        heatmap_data = []
        for bias_type in self.bias_types:
            if f'{bias_type}_mean' in self.yearly_df.columns:
                heatmap_data.append({
                    'bias_type': self.labels.get(bias_type, bias_type),
                    **{str(year): val for year, val in
                       zip(self.yearly_df['year'], self.yearly_df[f'{bias_type}_mean'])}
                })

        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('bias_type')

        fig, ax = plt.subplots(figsize=(20, 6))

        sns.heatmap(heatmap_df, cmap='RdYlGn_r', center=0.5, vmin=0, vmax=1,
                    annot=False, fmt='.2f', linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'Bias Score'})

        ax.set_title('Bias Scores Heatmap by Year (1997-2025)', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Bias Type', fontsize=12)

        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')

        self._save_figure(fig, 'bias_heatmap_yearly', 'heatmaps')

    def plot_high_bias_heatmap(self):
        """Heatmap showing percentage of high-bias articles."""
        heatmap_data = []
        for bias_type in self.bias_types:
            if f'{bias_type}_high_pct' in self.yearly_df.columns:
                heatmap_data.append({
                    'bias_type': self.labels.get(bias_type, bias_type),
                    **{str(year): val for year, val in
                       zip(self.yearly_df['year'], self.yearly_df[f'{bias_type}_high_pct'])}
                })

        if not heatmap_data:
            return

        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('bias_type')

        fig, ax = plt.subplots(figsize=(20, 6))

        sns.heatmap(heatmap_df, cmap='Reds', vmin=0, annot=False,
                    fmt='.1f', linewidths=0.5, ax=ax,
                    cbar_kws={'label': '% High Bias Articles'})

        ax.set_title('Percentage of High-Bias Articles by Year', fontsize=16)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Bias Type', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        self._save_figure(fig, 'high_bias_heatmap_yearly', 'heatmaps')

    # ========== DISTRIBUTION PLOTS ==========

    def plot_bias_distributions(self):
        """Box plots showing bias score distributions by year range."""
        # Group years into 5-year periods
        year_ranges = []
        for year in self.yearly_df['year']:
            decade = (year // 5) * 5
            year_ranges.append(f"{decade}-{decade + 4}")

        self.yearly_df['year_range'] = year_ranges

        for bias_type in self.bias_types:
            if f'{bias_type}_mean' not in self.yearly_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(14, 8))
            color = self.colors.get(bias_type, '#333333')
            label = self.labels.get(bias_type, bias_type)

            # Create box plot data
            data_by_range = []
            labels_list = []
            for yr_range in sorted(self.yearly_df['year_range'].unique()):
                subset = self.yearly_df[self.yearly_df['year_range'] == yr_range]
                data_by_range.append(subset[f'{bias_type}_mean'].values)
                labels_list.append(yr_range)

            bp = ax.boxplot(data_by_range, labels=labels_list, patch_artist=True)

            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            ax.set_xlabel('Year Range', fontsize=14)
            ax.set_ylabel('Bias Score', fontsize=14)
            ax.set_title(f'{label} Bias Distribution by 5-Year Periods', fontsize=16)
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45, ha='right')

            self._save_figure(fig, f'{bias_type}_distribution_boxplot', 'distributions')

    def plot_overall_distribution_violin(self):
        """Violin plot comparing all bias types."""
        # Melt data for violin plot
        melt_data = []
        for bias_type in self.bias_types:
            if f'{bias_type}_mean' in self.yearly_df.columns:
                for _, row in self.yearly_df.iterrows():
                    melt_data.append({
                        'Bias Type': self.labels.get(bias_type, bias_type),
                        'Score': row[f'{bias_type}_mean'],
                        'Year': row['year']
                    })

        melt_df = pd.DataFrame(melt_data)

        fig, ax = plt.subplots(figsize=(14, 8))

        palette = [self.colors.get(bt, '#333333') for bt in self.bias_types]
        sns.violinplot(data=melt_df, x='Bias Type', y='Score', ax=ax,
                       palette=palette, inner='box')

        ax.set_xlabel('Bias Type', fontsize=14)
        ax.set_ylabel('Bias Score', fontsize=14)
        ax.set_title('Bias Score Distribution by Type (All Years)', fontsize=16)
        ax.set_ylim(0, 1)

        self._save_figure(fig, 'bias_distribution_violin', 'distributions')

    # ========== MEDIA COMPARISON PLOTS ==========

    def plot_media_comparison_bar(self, top_n=10):
        """Bar chart comparing top media sources by bias."""
        # Get top N media by article count
        top_media = self.media_df.head(top_n)

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(top_media))
        width = 0.15

        for i, bias_type in enumerate(self.bias_types):
            if f'{bias_type}_mean' in top_media.columns:
                offset = (i - len(self.bias_types) / 2) * width
                color = self.colors.get(bias_type, '#333333')
                label = self.labels.get(bias_type, bias_type)
                ax.bar(x + offset, top_media[f'{bias_type}_mean'],
                       width, label=label, color=color, alpha=0.8)

        ax.set_xlabel('Media Source', fontsize=14)
        ax.set_ylabel('Average Bias Score', fontsize=14)
        ax.set_title(f'Bias Comparison: Top {top_n} Media Sources by Volume', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(top_media['media_name'], rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        self._save_figure(fig, 'media_comparison_bar', 'comparisons')

    def plot_media_radar(self, top_n=5):
        """Radar chart for top media sources."""
        from math import pi

        top_media = self.media_df.head(top_n)
        categories = [self.labels.get(bt, bt) for bt in self.bias_types]
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors_list = plt.cm.Set2(np.linspace(0, 1, len(top_media)))

        for idx, (_, row) in enumerate(top_media.iterrows()):
            values = [row.get(f'{bt}_mean', 0) for bt in self.bias_types]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=row['media_name'], color=colors_list[idx])
            ax.fill(angles, values, alpha=0.1, color=colors_list[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'Bias Profile: Top {top_n} Media Sources', fontsize=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        self._save_figure(fig, 'media_radar_chart', 'comparisons')

    def plot_media_heatmap(self, top_n=15):
        """Heatmap comparing media sources across bias types."""
        top_media = self.media_df.head(top_n)

        # Prepare heatmap data
        heatmap_data = []
        for _, row in top_media.iterrows():
            entry = {'Media': row['media_name']}
            for bias_type in self.bias_types:
                entry[self.labels.get(bias_type, bias_type)] = row.get(f'{bias_type}_mean', 0)
            heatmap_data.append(entry)

        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('Media')

        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(heatmap_df, cmap='RdYlGn_r', center=0.5, vmin=0, vmax=1,
                    annot=True, fmt='.2f', linewidths=0.5, ax=ax,
                    cbar_kws={'label': 'Bias Score'})

        ax.set_title(f'Bias Scores: Top {top_n} Media Sources', fontsize=16)
        plt.xticks(rotation=45, ha='right')

        self._save_figure(fig, 'media_bias_heatmap', 'media')

    def plot_media_trends(self, top_n=5):
        """Line plots showing bias trends for top media sources."""
        # Get top N media
        top_media_names = self.media_df.head(top_n)['media_name'].tolist()

        for bias_type in self.bias_types:
            if f'{bias_type}_mean' not in self.media_yearly_df.columns:
                continue

            fig, ax = plt.subplots(figsize=(14, 8))
            label = self.labels.get(bias_type, bias_type)

            colors_list = plt.cm.tab10(np.linspace(0, 1, len(top_media_names)))

            for idx, media in enumerate(top_media_names):
                media_data = self.media_yearly_df[self.media_yearly_df['media_name'] == media]
                media_data = media_data.sort_values('year')

                # Raw values (light)
                ax.plot(media_data['year'], media_data[f'{bias_type}_mean'],
                        color=colors_list[idx], alpha=0.3, linewidth=1)

                # 5-year MA (bold)
                if f'{bias_type}_ma_5y' in media_data.columns:
                    ax.plot(media_data['year'], media_data[f'{bias_type}_ma_5y'],
                            color=colors_list[idx], linewidth=2.5, marker='o',
                            markersize=3, label=media)

            ax.set_xlabel('Year', fontsize=14)
            ax.set_ylabel('Bias Score', fontsize=14)
            ax.set_title(f'{label} Bias Trend by Media Source (5-Year MA)', fontsize=16)
            ax.legend(loc='upper left', fontsize=10)
            ax.set_ylim(0, 1)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.3)

            self._save_figure(fig, f'{bias_type}_media_trends', 'media')

    # ========== CORRELATION PLOTS ==========

    def plot_bias_correlation(self):
        """Correlation matrix between bias types."""
        # Get mean columns for correlation
        corr_cols = [f'{bt}_mean' for bt in self.bias_types if f'{bt}_mean' in self.yearly_df.columns]

        if len(corr_cols) < 2:
            return

        corr_matrix = self.yearly_df[corr_cols].corr()

        # Rename columns for display
        rename_map = {f'{bt}_mean': self.labels.get(bt, bt) for bt in self.bias_types}
        corr_matrix = corr_matrix.rename(columns=rename_map, index=rename_map)

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    vmin=-1, vmax=1, annot=True, fmt='.2f', linewidths=0.5,
                    ax=ax, cbar_kws={'label': 'Correlation'})

        ax.set_title('Correlation Between Bias Types', fontsize=16)

        self._save_figure(fig, 'bias_correlation_matrix', 'comparisons')

    def plot_bias_scatter_matrix(self):
        """Scatter matrix showing relationships between bias types."""
        cols = [f'{bt}_mean' for bt in self.bias_types if f'{bt}_mean' in self.yearly_df.columns]

        if len(cols) < 2:
            return

        plot_df = self.yearly_df[cols].copy()
        rename_map = {f'{bt}_mean': self.labels.get(bt, bt) for bt in self.bias_types}
        plot_df = plot_df.rename(columns=rename_map)

        fig = plt.figure(figsize=(14, 14))
        axes = pd.plotting.scatter_matrix(plot_df, alpha=0.6, figsize=(14, 14),
                                          diagonal='hist', color='steelblue')

        for ax in axes.flatten():
            ax.xaxis.label.set_rotation(45)
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')

        fig.suptitle('Bias Types Scatter Matrix', fontsize=16, y=1.02)

        self._save_figure(fig, 'bias_scatter_matrix', 'comparisons')

    # ========== SUMMARY STATISTICS ==========

    def plot_summary_stats(self):
        """Create a summary visualization with key metrics."""
        fig = plt.figure(figsize=(18, 12))

        # 1. Overall bias trend (top left)
        ax1 = fig.add_subplot(2, 2, 1)
        if 'overall_bias_mean' in self.yearly_df.columns:
            ax1.plot(self.yearly_df['year'], self.yearly_df['overall_bias_mean'],
                     'b-', alpha=0.4, linewidth=1)
            if 'overall_bias_ma_5y' in self.yearly_df.columns:
                ax1.plot(self.yearly_df['year'], self.yearly_df['overall_bias_ma_5y'],
                         'b-', linewidth=3, label='5-Year MA')
        ax1.set_title('Overall Bias Trend', fontsize=14)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Bias Score')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 2. Article volume (top right)
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.bar(self.yearly_df['year'], self.yearly_df['article_count'],
                color='steelblue', alpha=0.7)
        ax2.set_title('Article Volume by Year', fontsize=14)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Bias comparison (bottom left)
        ax3 = fig.add_subplot(2, 2, 3)
        avg_biases = []
        labels_list = []
        colors_list = []
        for bias_type in self.bias_types:
            if f'{bias_type}_mean' in self.yearly_df.columns:
                avg_biases.append(self.yearly_df[f'{bias_type}_mean'].mean())
                labels_list.append(self.labels.get(bias_type, bias_type))
                colors_list.append(self.colors.get(bias_type, '#333333'))

        ax3.barh(labels_list, avg_biases, color=colors_list, alpha=0.8)
        ax3.set_title('Average Bias by Type (All Years)', fontsize=14)
        ax3.set_xlabel('Average Bias Score')
        ax3.set_xlim(0, 1)
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Top media sources (bottom right)
        ax4 = fig.add_subplot(2, 2, 4)
        top_5 = self.media_df.head(5)
        if 'overall_bias_mean' in top_5.columns:
            y_pos = np.arange(len(top_5))
            ax4.barh(y_pos, top_5['overall_bias_mean'], color='coral', alpha=0.8)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(top_5['media_name'])
            ax4.set_title('Overall Bias: Top 5 Media Sources', fontsize=14)
            ax4.set_xlabel('Overall Bias Score')
            ax4.set_xlim(0, 1)
            ax4.grid(True, alpha=0.3, axis='x')

        fig.suptitle('Bias Detection Summary Dashboard', fontsize=18, y=1.02)
        fig.tight_layout()

        self._save_figure(fig, 'summary_dashboard', '')

    # ========== BIAS TYPE DISTRIBUTION PLOTS ==========
    # (left_leaning, right_leaning, neutral, etc. over time)

    def plot_political_type_trends(self):
        """Plot political type distribution over years (left/right/neutral)."""
        if 'political_type' not in self.bias_type_dfs:
            print("Political type data not found, skipping...")
            return

        df = self.bias_type_dfs['political_type'].copy()

        # Get only the count columns (exclude percentage columns and total)
        count_cols = [c for c in df.columns if not c.endswith('_pct') and c != 'total']

        if len(count_cols) == 0:
            return

        # Calculate percentages for stacked area
        df_pct = df[count_cols].div(df[count_cols].sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        # Stacked Area Chart (Counts)
        ax1 = axes[0]
        colors = plt.cm.RdYlBu(np.linspace(0.1, 0.9, len(count_cols)))
        df[count_cols].plot.area(ax=ax1, stacked=True, alpha=0.7, color=colors)
        ax1.set_title('Political Orientation Distribution Over Time (Article Counts)', fontsize=14)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Articles')
        ax1.legend(title='Political Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.grid(True, alpha=0.3)

        # Stacked Area Chart (Percentage)
        ax2 = axes[1]
        df_pct.plot.area(ax=ax2, stacked=True, alpha=0.7, color=colors)
        ax2.set_title('Political Orientation Distribution Over Time (Percentage)', fontsize=14)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Percentage of Articles (%)')
        ax2.set_ylim(0, 100)
        ax2.legend(title='Political Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Political Bias Type: Left-Leaning vs Right-Leaning vs Neutral', fontsize=16, y=1.02)
        fig.tight_layout()

        self._save_figure(fig, 'political_type_trends', 'distributions')

        # Also create line chart with 5-year MA for each political type
        self._plot_type_with_ma(df_pct, 'political_type', 'Political Orientation')

    def plot_gender_type_trends(self):
        """Plot gender type distribution over years (male_bias/female_bias/neutral)."""
        if 'gender_type' not in self.bias_type_dfs:
            print("Gender type data not found, skipping...")
            return

        df = self.bias_type_dfs['gender_type'].copy()
        count_cols = [c for c in df.columns if not c.endswith('_pct') and c != 'total']

        if len(count_cols) == 0:
            return

        df_pct = df[count_cols].div(df[count_cols].sum(axis=1), axis=0) * 100

        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        colors = plt.cm.PiYG(np.linspace(0.1, 0.9, len(count_cols)))

        # Stacked Area (Counts)
        ax1 = axes[0]
        df[count_cols].plot.area(ax=ax1, stacked=True, alpha=0.7, color=colors)
        ax1.set_title('Gender Bias Type Distribution Over Time (Article Counts)', fontsize=14)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Articles')
        ax1.legend(title='Gender Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.grid(True, alpha=0.3)

        # Stacked Area (Percentage)
        ax2 = axes[1]
        df_pct.plot.area(ax=ax2, stacked=True, alpha=0.7, color=colors)
        ax2.set_title('Gender Bias Type Distribution Over Time (Percentage)', fontsize=14)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Percentage of Articles (%)')
        ax2.set_ylim(0, 100)
        ax2.legend(title='Gender Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Gender Bias Type: Male-Centric vs Female-Centric vs Neutral', fontsize=16, y=1.02)
        fig.tight_layout()

        self._save_figure(fig, 'gender_type_trends', 'distributions')
        self._plot_type_with_ma(df_pct, 'gender_type', 'Gender Orientation')

    def plot_religious_type_trends(self):
        """Plot religious type distribution over years."""
        if 'religious_type' not in self.bias_type_dfs:
            print("Religious type data not found, skipping...")
            return

        df = self.bias_type_dfs['religious_type'].copy()
        count_cols = [c for c in df.columns if not c.endswith('_pct') and c != 'total']

        if len(count_cols) == 0:
            return

        df_pct = df[count_cols].div(df[count_cols].sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(16, 8))
        colors = plt.cm.Set2(np.linspace(0, 1, len(count_cols)))
        df_pct.plot.area(ax=ax, stacked=True, alpha=0.7, color=colors)
        ax.set_title('Religious Bias Type Distribution Over Time', fontsize=16)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Articles (%)')
        ax.set_ylim(0, 100)
        ax.legend(title='Religious Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_figure(fig, 'religious_type_trends', 'distributions')
        self._plot_type_with_ma(df_pct, 'religious_type', 'Religious Orientation')

    def plot_caste_type_trends(self):
        """Plot caste type distribution over years."""
        if 'caste_type' not in self.bias_type_dfs:
            print("Caste type data not found, skipping...")
            return

        df = self.bias_type_dfs['caste_type'].copy()
        count_cols = [c for c in df.columns if not c.endswith('_pct') and c != 'total']

        if len(count_cols) == 0:
            return

        df_pct = df[count_cols].div(df[count_cols].sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(16, 8))
        colors = plt.cm.Accent(np.linspace(0, 1, len(count_cols)))
        df_pct.plot.area(ax=ax, stacked=True, alpha=0.7, color=colors)
        ax.set_title('Caste Bias Type Distribution Over Time', fontsize=16)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Articles (%)')
        ax.set_ylim(0, 100)
        ax.legend(title='Caste Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_figure(fig, 'caste_type_trends', 'distributions')
        self._plot_type_with_ma(df_pct, 'caste_type', 'Caste Focus')

    def plot_region_type_trends(self):
        """Plot region type distribution over years."""
        if 'region_type' not in self.bias_type_dfs:
            print("Region type data not found, skipping...")
            return

        df = self.bias_type_dfs['region_type'].copy()
        count_cols = [c for c in df.columns if not c.endswith('_pct') and c != 'total']

        if len(count_cols) == 0:
            return

        df_pct = df[count_cols].div(df[count_cols].sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(16, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(count_cols)))
        df_pct.plot.area(ax=ax, stacked=True, alpha=0.7, color=colors)
        ax.set_title('Regional Bias Type Distribution Over Time', fontsize=16)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Articles (%)')
        ax.set_ylim(0, 100)
        ax.legend(title='Region Type', loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_figure(fig, 'region_type_trends', 'distributions')
        self._plot_type_with_ma(df_pct, 'region_type', 'Regional Focus')

    def _plot_type_with_ma(self, df_pct, type_name, title_prefix):
        """Plot each bias type category with 5-year moving average."""
        fig, ax = plt.subplots(figsize=(16, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(df_pct.columns)))

        for i, col in enumerate(df_pct.columns):
            # Raw values (light, dashed)
            ax.plot(df_pct.index, df_pct[col], color=colors[i],
                    alpha=0.3, linewidth=1, linestyle='--')

            # 5-year moving average (bold)
            ma_5y = df_pct[col].rolling(window=5, min_periods=1).mean()
            ax.plot(df_pct.index, ma_5y, color=colors[i],
                    linewidth=2.5, marker='o', markersize=3, label=f'{col} (5-yr MA)')

        ax.set_title(f'{title_prefix}: Percentage Over Time with 5-Year Moving Average', fontsize=16)
        ax.set_xlabel('Year')
        ax.set_ylabel('Percentage of Articles (%)')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._save_figure(fig, f'{type_name}_5y_ma', 'distributions')

    def plot_all_type_summary(self):
        """Create a summary grid of all bias type distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()

        type_configs = [
            ('political_type', 'Political Orientation', plt.cm.RdYlBu),
            ('gender_type', 'Gender Bias', plt.cm.PiYG),
            ('religious_type', 'Religious Bias', plt.cm.Set2),
            ('caste_type', 'Caste Focus', plt.cm.Accent),
            ('region_type', 'Regional Focus', plt.cm.tab10),
        ]

        for idx, (type_name, title, cmap) in enumerate(type_configs):
            ax = axes[idx]

            if type_name not in self.bias_type_dfs:
                ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center')
                ax.set_title(title)
                continue

            df = self.bias_type_dfs[type_name].copy()
            count_cols = [c for c in df.columns if not c.endswith('_pct') and c != 'total']

            if len(count_cols) == 0:
                continue

            df_pct = df[count_cols].div(df[count_cols].sum(axis=1), axis=0) * 100
            colors = cmap(np.linspace(0.1, 0.9, len(count_cols)))

            df_pct.plot.area(ax=ax, stacked=True, alpha=0.7, color=colors, legend=False)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Year')
            ax.set_ylabel('%')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

        # Hide unused subplot
        axes[-1].axis('off')

        # Add common legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right', ncol=3, fontsize=10)

        fig.suptitle('Bias Type Distributions Over Time (All Categories)', fontsize=18, y=1.02)
        fig.tight_layout()

        self._save_figure(fig, 'all_bias_types_summary', 'distributions')

    def generate_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)

        # Load data
        self.load_data()

        print("\n--- Time Series Plots ---")
        self.plot_all_bias_trends()
        self.plot_individual_bias_trends()
        self.plot_bias_comparison_subplots()
        self.plot_article_volume()

        print("\n--- Heatmaps ---")
        self.plot_bias_heatmap()
        self.plot_high_bias_heatmap()

        print("\n--- Distribution Plots ---")
        self.plot_bias_distributions()
        self.plot_overall_distribution_violin()

        print("\n--- Bias Type Distributions (Left/Right/Neutral over time) ---")
        self.plot_political_type_trends()
        self.plot_gender_type_trends()
        self.plot_religious_type_trends()
        self.plot_caste_type_trends()
        self.plot_region_type_trends()
        self.plot_all_type_summary()

        print("\n--- Media Comparisons ---")
        self.plot_media_comparison_bar()
        self.plot_media_radar()
        self.plot_media_heatmap()
        self.plot_media_trends()

        print("\n--- Correlation Analysis ---")
        self.plot_bias_correlation()
        self.plot_bias_scatter_matrix()

        print("\n--- Summary Dashboard ---")
        self.plot_summary_stats()

        print("\n" + "=" * 50)
        print("ALL VISUALIZATIONS COMPLETED!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 50)


def main():
    """Main entry point."""
    visualizer = BiasVisualizer()
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
