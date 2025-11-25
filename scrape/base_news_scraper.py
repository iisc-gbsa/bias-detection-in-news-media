"""
Base News Scraper Template

This is a reusable template/base class for creating news scrapers with caching support.
Each news source needs to implement its own URL construction and article extraction logic.

Usage:
1. Copy this structure to create a new scraper (e.g., scrape_economic_times.py)
2. Update the configuration constants (BASE_URL, MEDIA_NAME, CACHE_DIR, etc.)
3. Implement the site-specific logic in scrape_articles_for_date()
4. Adjust the main() function date ranges as needed
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import os
from datetime import datetime


# ============================================================================
# CONFIGURATION - UPDATE THESE FOR EACH NEWS SOURCE
# ============================================================================

BASE_URL = "https://example-news-site.com"
MEDIA_NAME = "EXAMPLE NEWS"
CACHE_DIR = "cache_example"

# Additional site-specific constants (if needed)
# For example, TOI uses INITIAL_STARTTIME
# Add any constants specific to your news source here


# ============================================================================
# CACHE MANAGEMENT (REUSABLE - NO CHANGES NEEDED)
# ============================================================================

PROGRESS_FILE = os.path.join(CACHE_DIR, "scraping_progress.json")
DATA_CACHE_FILE = os.path.join(CACHE_DIR, "scraped_data_cache.csv")


def load_progress():
    """Load scraping progress from cache."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_dates": [], "last_date": None}


def save_progress(completed_dates, last_date):
    """Save scraping progress to cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    progress = {
        "completed_dates": completed_dates,
        "last_date": last_date,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def append_to_cache(articles):
    """Append articles to cache CSV file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = pd.DataFrame(articles)

    if os.path.exists(DATA_CACHE_FILE):
        df.to_csv(DATA_CACHE_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(DATA_CACHE_FILE, mode="w", header=True, index=False)


def load_cached_data():
    """Load all cached data."""
    if os.path.exists(DATA_CACHE_FILE):
        return pd.read_csv(DATA_CACHE_FILE)
    return pd.DataFrame()


# ============================================================================
# SITE-SPECIFIC SCRAPING LOGIC - CUSTOMIZE THIS SECTION
# ============================================================================


def scrape_articles_for_date(year, month, day):
    """
    Scrape article links for a specific date.

    **THIS FUNCTION MUST BE CUSTOMIZED FOR EACH NEWS SOURCE**

    Each website has different:
    - Archive URL patterns
    - HTML structure
    - Article link identification logic

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day (1-31)

    Returns:
        list: List of dictionaries with keys: 'Media Name', 'Article Link', 'Date'
    """
    date_str = f"{year}-{month:02d}-{day:02d}"

    # ========================================================================
    # STEP 1: Construct the archive URL for this specific news source
    # ========================================================================
    # Examples:
    # Indian Express: f'{BASE_URL}/archive/{year}/{month:02d}/{day:02d}/'
    # Times of India: f'{BASE_URL}/{year}/{month}/{day}/archivelist/...'
    # Economic Times: (implement based on their archive structure)
    # Hindu: (implement based on their archive structure)

    url = f"{BASE_URL}/archive/{year}/{month:02d}/{day:02d}/"  # CUSTOMIZE THIS
    print(f"Scraping URL: {url}")

    # ========================================================================
    # STEP 2: Set up headers
    # ========================================================================
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": BASE_URL,
        "Connection": "keep-alive",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(
                f"Failed to retrieve data from {url} - Status code: {response.status_code}"
            )
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        # ====================================================================
        # STEP 3: Extract article links - CUSTOMIZE THIS SECTION
        # ====================================================================
        # This is the most important part to customize for each site
        # Different websites have different HTML structures and patterns

        article_links = []

        # Example 1: Indian Express approach (look for /article/ in href)
        # for link in soup.find_all('a', href=True):
        #     href = link['href']
        #     if '/article/' in href:
        #         full_link = BASE_URL + href if href.startswith('/') else href
        #         full_link = full_link.split('?')[0]  # Remove query params
        #         if full_link not in [item['Article Link'] for item in article_links]:
        #             article_links.append({...})

        # Example 2: Times of India approach (filter by keywords)
        # for link in soup.find_all('a', href=True):
        #     href = link['href']
        #     if href.startswith('/') or href.startswith('http'):
        #         full_link = BASE_URL + href if href.startswith('/') else href
        #         if "articles" in full_link or "news" in full_link:
        #             if full_link not in [item['Article Link'] for item in article_links]:
        #                 article_links.append({...})

        # YOUR IMPLEMENTATION HERE:
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Implement your site-specific filtering logic
            # Example placeholder logic:
            if href.startswith("/"):
                full_link = BASE_URL + href
            elif href.startswith("http"):
                full_link = href
            else:
                continue

            # Add your filtering conditions (e.g., URL patterns, keywords, etc.)
            if "YOUR_FILTER_CONDITION_HERE":  # CUSTOMIZE THIS
                if full_link not in [item["Article Link"] for item in article_links]:
                    article_links.append(
                        {
                            "Media Name": MEDIA_NAME,
                            "Article Link": full_link,
                            "Date": date_str,
                        }
                    )

        print(f"Found {len(article_links)} articles for {date_str}")
        return article_links

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return []


# ============================================================================
# MAIN SCRAPING LOOP (REUSABLE - MINIMAL CHANGES NEEDED)
# ============================================================================


def scrape_articles(start_year=2020, end_year=2024, use_cache=True):
    """
    Scrape articles for a date range with caching support.

    This function is reusable across all scrapers.

    Args:
        start_year (int): Starting year
        end_year (int): Ending year (inclusive)
        use_cache (bool): Whether to use cache and resume from last position

    Returns:
        list: List of all articles scraped
    """
    progress = (
        load_progress() if use_cache else {"completed_dates": [], "last_date": None}
    )
    completed_dates = set(progress["completed_dates"])

    print(f"\nCache status: {len(completed_dates)} dates already scraped")
    if progress["last_date"]:
        print(f"Last scraped date: {progress['last_date']}")

    all_articles = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Calculate days in month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                num_days = 31
            elif month in [4, 6, 9, 11]:
                num_days = 30
            else:  # February
                num_days = (
                    29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28
                )

            for day in range(1, num_days + 1):
                date_str = f"{year}-{month:02d}-{day:02d}"

                # Skip if already scraped
                if use_cache and date_str in completed_dates:
                    print(f"Skipping {date_str} (already scraped)")
                    continue

                try:
                    articles = scrape_articles_for_date(year, month, day)
                    if articles:
                        all_articles.extend(articles)

                        # Save to cache immediately
                        if use_cache:
                            append_to_cache(articles)

                    # Mark date as completed
                    if use_cache:
                        completed_dates.add(date_str)
                        save_progress(list(completed_dates), date_str)

                except Exception as e:
                    print(f"Error on {date_str}: {e}")
                    continue

                # Random delay between requests
                time.sleep(random.uniform(1, 3))

    return all_articles


# ============================================================================
# MAIN EXECUTION (UPDATE DATE RANGES AS NEEDED)
# ============================================================================


def main():
    """Main execution function"""
    print("=" * 80)
    print(f"{MEDIA_NAME} ARCHIVE SCRAPER")
    print("=" * 80)

    # Configure date range here
    START_YEAR = 2020
    END_YEAR = 2024

    print(f"\nScraping articles from {START_YEAR} to {END_YEAR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check for existing cache
    cached_df = load_cached_data()
    if len(cached_df) > 0:
        print(f"\nFound cached data: {len(cached_df)} articles")
        print("Resuming from last position...\n")

    # Scrape articles
    all_data = scrape_articles(start_year=START_YEAR, end_year=END_YEAR, use_cache=True)

    # Combine with cached data
    if len(cached_df) > 0:
        new_df = pd.DataFrame(all_data)
        df = pd.concat([cached_df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame(all_data)

    # Display results
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETED")
    print("=" * 80)
    print(f"Total articles scraped: {len(df)}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if len(df) > 0:
        print("\nFirst 5 articles:")
        print(df.head())

        # Remove duplicates
        df = df.drop_duplicates(subset=["Article Link"], keep="first")

        # Save final output
        output_file = f'{MEDIA_NAME.lower().replace(" ", "_")}_articles_{START_YEAR}_to_{END_YEAR}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nFinal data saved to: {output_file}")
        print(f"Cache saved in: {CACHE_DIR}/")

        # Display statistics
        print("\nStatistics:")
        print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"- Unique articles: {df['Article Link'].nunique()}")
    else:
        print("\nNo articles found. Please check the date range and try again.")


if __name__ == "__main__":
    main()
