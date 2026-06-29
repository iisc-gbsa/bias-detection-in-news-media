#!/usr/bin/env python3
"""
Main pipeline runner for bias visualization.
Executes: Extract -> Aggregate -> Visualize
"""

import os
import sys
import argparse
import importlib.util
from datetime import datetime

# Add scripts directory to path
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, script_dir)


def run_extraction():
    """Run data extraction from MongoDB."""
    print("\n" + "=" * 60)
    print("STEP 1: DATA EXTRACTION")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load utils
    utils_path = os.path.join(base_dir, "scripts", "utils.py")
    spec = importlib.util.spec_from_file_location("utils", utils_path)
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    # Ensure directories exist
    config = utils.load_config(os.path.join(base_dir, "config", "viz_config.yaml"))
    utils.ensure_directories(os.path.join(base_dir, config['output']['base_dir']))

    # Load extractor module
    extract_path = os.path.join(base_dir, "scripts", "01_extract_data.py")
    spec = importlib.util.spec_from_file_location("extract_data", extract_path)
    extract_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract_module)

    # Extract data
    extractor = extract_module.DataExtractor()
    df = extractor.extract_articles()

    if df is not None and len(df) > 0:
        extractor.save_raw_data(df)
        print(f"Extracted {len(df)} articles successfully")
        return True
    elif df is not None and len(df) == 0:
        print("No articles found in MongoDB collection!")
        print("Please check:")
        print("  1. MongoDB is running on the correct port")
        print("  2. The database and collection names are correct")
        print("  3. The collection contains data")
        return False
    else:
        print("Extraction failed!")
        return False


def run_aggregation():
    """Run data aggregation."""
    print("\n" + "=" * 60)
    print("STEP 2: DATA AGGREGATION")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load aggregator module
    agg_path = os.path.join(base_dir, "scripts", "02_aggregate_data.py")
    spec = importlib.util.spec_from_file_location("aggregate_data", agg_path)
    agg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agg_module)

    aggregator = agg_module.DataAggregator()

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

    print("Aggregation completed successfully")
    return True


def run_visualization():
    """Run visualization generation."""
    print("\n" + "=" * 60)
    print("STEP 3: VISUALIZATION GENERATION")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load visualizer module
    viz_path = os.path.join(base_dir, "scripts", "03_visualize.py")
    spec = importlib.util.spec_from_file_location("visualize", viz_path)
    viz_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(viz_module)

    visualizer = viz_module.BiasVisualizer()
    visualizer.generate_all_visualizations()

    print("Visualization completed successfully")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bias Detection Visualization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. extract    - Extract data from MongoDB
  2. aggregate  - Create yearly aggregations with 5-year MA
  3. visualize  - Generate matplotlib/seaborn charts
  
Examples:
  python run_pipeline.py --all           # Run complete pipeline
  python run_pipeline.py --extract       # Extract only
  python run_pipeline.py --aggregate     # Aggregate only (requires extracted data)
  python run_pipeline.py --visualize     # Visualize only (requires aggregated data)
        """
    )

    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (extract -> aggregate -> visualize)')
    parser.add_argument('--extract', action='store_true',
                        help='Run data extraction from MongoDB')
    parser.add_argument('--aggregate', action='store_true',
                        help='Run data aggregation')
    parser.add_argument('--visualize', action='store_true',
                        help='Run visualization generation')

    args = parser.parse_args()

    # If no args provided, show help
    if not any([args.all, args.extract, args.aggregate, args.visualize]):
        parser.print_help()
        return

    start_time = datetime.now()
    print(f"\nPipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    success = True

    if args.all or args.extract:
        success = run_extraction() and success

    if args.all or args.aggregate:
        if success:
            success = run_aggregation() and success
        else:
            print("Skipping aggregation due to previous failure")

    if args.all or args.visualize:
        if success:
            success = run_visualization() and success
        else:
            print("Skipping visualization due to previous failure")

    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    if success:
        print("PIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("PIPELINE COMPLETED WITH ERRORS")
    print(f"Duration: {duration}")
    print("=" * 60)


if __name__ == "__main__":
    main()
