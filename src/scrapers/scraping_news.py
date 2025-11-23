import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime, timedelta

# Initial starttime for January 1, 2020
INITIAL_STARTTIME = 43831


# Function to get article links for a specific date
def scrape_articles_for_date(media_name, year, month, day):
    # Calculate starttime based on the date
    date_str = f'{year}-{month:02}-{day:02}'
    date_object = datetime(year, month, day)

    # Calculate the number of days since January 1, 2020
    days_since_start = (date_object - datetime(2020, 1, 1)).days

    # Calculate starttime
    starttime = INITIAL_STARTTIME + days_since_start

    # Construct the URL for the specific date
    url = f'{BASE_URL}/archivelist/year-{year},month-{month},starttime-{starttime}.cms'
    print(f'Scraping URL: {url}')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': BASE_URL,
        'Connection': 'keep-alive'
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to retrieve data from {url} - Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all article links on the page and filter out ads
    articles = soup.find_all('a', href=True)

    # Extracting article links and returning them
    article_links = []
    for article in articles:
        link = article['href']

        # Check if the link is an article link and not an ad or unrelated link
        if link.startswith('/') or link.startswith('http'):
            full_link = BASE_URL + link if link.startswith('/') else link

            # Filter criteria to exclude ads or unrelated links
            if "article" in full_link or "news" in full_link:  # Adjust this condition based on actual URL patterns
                article_links.append({
                    'Media Name': media_name,
                    'Article Link': full_link,
                    'Date': date_str
                })

    return article_links


# Function to iterate through each month and day for a given year range
def scrape_articles(media_name, start_year=2020, end_year=2024):
    all_articles = []

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get the number of days in the month
            if month in [1, 3, 5, 7, 8, 10, 12]:
                num_days = 31
            elif month in [4, 6, 9, 11]:
                num_days = 30
            else:  # February
                num_days = 29 if year % 4 == 0 else 28

            for day in range(1, num_days + 1):
                try:
                    articles = scrape_articles_for_date(media_name, year, month, day)
                    if articles:
                        all_articles.extend(articles)
                except Exception as e:
                    print(f"Error on {year}-{month:02}-{day:02}: {e}")
                    continue

                time.sleep(random.uniform(1, 3))  # Random delay between requests

    return all_articles


# Base URL for Economic Times Archive
BASE_URL = 'https://economictimes.indiatimes.com'

# Main execution
all_data = scrape_articles('THE ECONOMIC TIMES', start_year=2024, end_year=2024)

# Convert the data into a DataFrame
df_et = pd.DataFrame(all_data)

print("Scraping completed.")
print(df_et.head(20))
print(len(df_et))

# Save the scraped data to a CSV file
df_et.to_csv('economic_times_articles_2024.csv', index=False)

print("Scraping completed and saved to CSV.")