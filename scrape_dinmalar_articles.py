"""
Dinamalar Archive Article Scraper

This script scrapes article URLs from The Dinamalar archive pages.
Archive URL format: https://www.dinamalar.com/archive/2024-Jan/01
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime

from urllib.parse import urljoin


def get_month_abbr(month_num):
    """
    Convert month number to 3-letter month abbreviation.

    Args:
        month_num (int): Month number (1-12)

    Returns:
        str: 3-letter month abbreviation (e.g., 'Jan', 'Feb', etc.)
    """
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months[month_num - 1] if 1 <= month_num <= 12 else ""


# Base URL for Indian Express Archive
BASE_URL = 'https://www.dinamalar.com'


def scrape_dinamalar_links_for_date(year, month, day):
    try:
        # Format date string with month abbreviation (e.g., '2024-Jan-01')
        month_abbr = get_month_abbr(month)

        date_str = f'{year}-{month:02d}-{day:02d}'

        date_month_str = f'{year}-{month_abbr}-{day:02d}'

        # Construct the archive URL for the specific date
        url = f'{BASE_URL}/archive/{year}-{month_abbr}/{day:02d}/'
        print(f'Scraping URL: {url}')
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all anchor tags
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            # Convert relative URLs to absolute URLs
            full_url = urljoin(url, href)
            # Filter out empty links and anchor links
            if full_url and not full_url.startswith('#') and '/photos/today-photos/' in full_url:
                links.append({
                    'Media Name': 'DINA MALAR',
                    'Article Link': full_url,
                    'text': a_tag.get_text(strip=True),
                    'Date': date_str,
                    'url': full_url
                })

        print(f"Found {len(links)} articles for {date_month_str}")
        return links

    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []


def scrape_articles(start_year=2020, end_year=2024):
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
                    articles = scrape_dinamalar_links_for_date(year, month, day)
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
    all_data = scrape_articles(start_year=START_YEAR, end_year=END_YEAR)

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
        output_file = f'dinamalar_articles_{START_YEAR}_to_{END_YEAR}.csv'
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