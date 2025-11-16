"""
Indian Express Archive Article Scraper

This script scrapes article URLs from The Indian Express archive pages.
Archive URL format: https://indianexpress.com/archive/YYYY/MM/DD/
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime

# Base URL for Indian Express Archive
BASE_URL = 'https://indianexpress.com'


def scrape_indian_express_articles_for_date(year, month, day):
    """
    Scrape article links for a specific date from Indian Express archive.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day (1-31)

    Returns:
        list: List of dictionaries containing article information
    """
    # Format date string
    date_str = f'{year}-{month:02d}-{day:02d}'

    # Construct the archive URL for the specific date
    url = f'{BASE_URL}/archive/{year}/{month:02d}/{day:02d}/'
    print(f'Scraping URL: {url}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': BASE_URL,
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"Failed to retrieve data from {url} - Status code: {response.status_code}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all article links on the page
        # Indian Express article URLs follow pattern: /article/...
        article_links = []

        # Find all <a> tags
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Check if it's an article link (contains /article/)
            if '/article/' in href:
                # Make full URL if it's a relative path
                if href.startswith('/'):
                    full_link = BASE_URL + href
                elif href.startswith('http'):
                    full_link = href
                else:
                    continue

                # Remove query parameters like ?ref=archive_pg
                full_link = full_link.split('?')[0]

                # Add to list if not already present
                if full_link not in [item['Article Link'] for item in article_links]:
                    article_links.append({
                        'Media Name': 'THE INDIAN EXPRESS',
                        'Article Link': full_link,
                        'Date': date_str
                    })

        print(f"Found {len(article_links)} articles for {date_str}")
        return article_links

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return []


def scrape_indian_express_articles(start_year=2020, end_year=2024):
    """
    Scrape articles for a date range.

    Args:
        start_year (int): Starting year
        end_year (int): Ending year (inclusive)

    Returns:
        list: List of all articles scraped
    """
    all_articles = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get the number of days in the month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                num_days = 31
            elif month in [4, 6, 9, 11]:
                num_days = 30
            else:  # February
                num_days = 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28

            for day in range(1, num_days + 1):
                try:
                    articles = scrape_indian_express_articles_for_date(year, month, day)
                    if articles:
                        all_articles.extend(articles)
                except Exception as e:
                    print(f"Error on {year}-{month:02d}-{day:02d}: {e}")
                    continue

                # Random delay between requests to be respectful to the server
                time.sleep(random.uniform(1, 3))

    return all_articles


def main():
    """Main execution function"""
    print("=" * 80)
    print("INDIAN EXPRESS ARCHIVE SCRAPER")
    print("=" * 80)

    # Configure date range here
    START_YEAR = 2024
    END_YEAR = 2024

    print(f"\nScraping articles from {START_YEAR} to {END_YEAR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Scrape articles
    all_data = scrape_indian_express_articles(start_year=START_YEAR, end_year=END_YEAR)

    # Convert to DataFrame
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

        # Save to CSV
        output_file = f'indian_express_articles_{START_YEAR}_to_{END_YEAR}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nData saved to: {output_file}")

        # Display statistics
        print("\nStatistics:")
        print(f"- Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"- Unique articles: {df['Article Link'].nunique()}")
    else:
        print("\nNo articles found. Please check the date range and try again.")


if __name__ == "__main__":
    main()
