"""
Indian Express Article Content Scraper

This script reads article URLs from a CSV file and scrapes the full content of each article.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from datetime import datetime
import json
import re


def scrape_article_content(url):
    """
    Scrape the full content of an article from Indian Express.

    Args:
        url (str): Article URL

    Returns:
        dict: Dictionary containing article content and metadata
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            print(f"Failed to retrieve {url} - Status code: {response.status_code}")
            return {
                'success': False,
                'error': f"HTTP {response.status_code}"
            }

        soup = BeautifulSoup(response.content, 'html.parser')

        # Initialize result dictionary
        article_data = {
            'success': True,
            'url': url,
            'title': None,
            'author': None,
            'published_date': None,
            'modified_date': None,
            'section': None,
            'tags': None,
            'article_text': None,
            'word_count': 0,
            'error': None
        }

        # Extract title
        title_tag = soup.find('h1', class_='native_story_title')
        if not title_tag:
            title_tag = soup.find('h1')
        if title_tag:
            article_data['title'] = title_tag.get_text(strip=True)

        # Extract author
        author_tag = soup.find('p', class_='editor')
        if not author_tag:
            author_tag = soup.find('div', class_='editor')
        if not author_tag:
            author_tag = soup.find('a', rel='author')
        if author_tag:
            author_text = author_tag.get_text(strip=True)
            # Clean up author text (remove "Written by", "By", etc.)
            author_text = re.sub(r'^(Written by|By|Author:)\s*', '', author_text, flags=re.IGNORECASE)
            article_data['author'] = author_text

        # Extract published date from meta tags or JSON-LD
        # Try meta tags first
        date_meta = soup.find('meta', property='article:published_time')
        if date_meta:
            article_data['published_date'] = date_meta.get('content')

        # Try modified date
        modified_meta = soup.find('meta', property='article:modified_time')
        if modified_meta:
            article_data['modified_date'] = modified_meta.get('content')

        # Extract section/category
        section_meta = soup.find('meta', property='article:section')
        if section_meta:
            article_data['section'] = section_meta.get('content')

        # Extract tags
        tag_meta = soup.find('meta', property='article:tag')
        if tag_meta:
            article_data['tags'] = tag_meta.get('content')
        else:
            # Try to find tags in the page
            tag_elements = soup.find_all('a', class_='tag')
            if tag_elements:
                article_data['tags'] = ', '.join([tag.get_text(strip=True) for tag in tag_elements])

        # Extract article text/body
        # Indian Express uses different structures for article content
        article_body = None

        # Method 1: Look for story content div
        article_body = soup.find('div', class_='story_details')
        if not article_body:
            article_body = soup.find('div', class_='full-details')
        if not article_body:
            article_body = soup.find('div', itemprop='articleBody')
        if not article_body:
            article_body = soup.find('article')

        if article_body:
            # Extract all paragraphs
            paragraphs = article_body.find_all('p')

            # Filter out unwanted paragraphs (ads, related articles, etc.)
            article_text_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                # Skip empty paragraphs or very short ones that are likely not content
                if len(text) > 20:
                    # Skip paragraphs that are likely ads or navigation
                    if not any(skip_word in text.lower() for skip_word in ['advertisement', 'also read', 'read more', 'subscribe now']):
                        article_text_parts.append(text)

            article_data['article_text'] = '\n\n'.join(article_text_parts)
            article_data['word_count'] = len(article_data['article_text'].split())

        # If no article text found, try alternative method
        if not article_data['article_text'] or article_data['word_count'] < 50:
            # Try JSON-LD data
            json_ld = soup.find('script', type='application/ld+json')
            if json_ld:
                try:
                    data = json.loads(json_ld.string)
                    if isinstance(data, list):
                        data = data[0]

                    if 'articleBody' in data:
                        article_data['article_text'] = data['articleBody']
                        article_data['word_count'] = len(article_data['article_text'].split())

                    # Get other metadata from JSON-LD if not already found
                    if not article_data['title'] and 'headline' in data:
                        article_data['title'] = data['headline']
                    if not article_data['author'] and 'author' in data:
                        if isinstance(data['author'], dict):
                            article_data['author'] = data['author'].get('name')
                        elif isinstance(data['author'], list):
                            article_data['author'] = ', '.join([a.get('name', '') for a in data['author']])
                    if not article_data['published_date'] and 'datePublished' in data:
                        article_data['published_date'] = data['datePublished']
                    if not article_data['modified_date'] and 'dateModified' in data:
                        article_data['modified_date'] = data['dateModified']

                except json.JSONDecodeError:
                    pass

        return article_data

    except requests.exceptions.Timeout:
        print(f"Timeout while fetching {url}")
        return {'success': False, 'url': url, 'error': 'Timeout'}
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {str(e)}")
        return {'success': False, 'url': url, 'error': str(e)}
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return {'success': False, 'url': url, 'error': str(e)}


def scrape_articles_from_csv(csv_file, output_file=None, start_index=0, limit=None):
    """
    Read article URLs from CSV and scrape content for each.

    Args:
        csv_file (str): Path to input CSV file with article URLs
        output_file (str): Path to output CSV file (optional)
        start_index (int): Index to start scraping from (for resuming)
        limit (int): Maximum number of articles to scrape (optional)

    Returns:
        pd.DataFrame: DataFrame with article content
    """
    print("=" * 80)
    print("ARTICLE CONTENT SCRAPER")
    print("=" * 80)

    # Read the CSV file
    print(f"\nReading URLs from: {csv_file}")
    df = pd.read_csv(csv_file)

    total_articles = len(df)
    print(f"Total articles in CSV: {total_articles}")

    # Determine range to scrape
    end_index = min(start_index + limit, total_articles) if limit else total_articles
    print(f"Scraping articles from index {start_index} to {end_index - 1}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Initialize results list
    scraped_content = []
    success_count = 0
    failure_count = 0

    # Scrape articles
    for idx in range(start_index, end_index):
        row = df.iloc[idx]
        article_url = row['Article Link']

        print(f"\n[{idx + 1}/{end_index}] Scraping: {article_url}")

        # Scrape article content
        content = scrape_article_content(article_url)

        # Add original metadata
        content['media_name'] = row['Media Name']
        content['scrape_date'] = row['Date']
        content['csv_index'] = idx

        scraped_content.append(content)

        if content['success']:
            success_count += 1
            print(f"✓ Success - Title: {content.get('title', 'N/A')[:60]}...")
            print(f"  Word count: {content.get('word_count', 0)}")
        else:
            failure_count += 1
            print(f"✗ Failed - Error: {content.get('error', 'Unknown')}")

        # Save progress periodically (every 50 articles)
        if (idx + 1) % 50 == 0 and output_file:
            temp_df = pd.DataFrame(scraped_content)
            temp_file = output_file.replace('.csv', f'_progress_{idx + 1}.csv')
            temp_df.to_csv(temp_file, index=False)
            print(f"\n💾 Progress saved to: {temp_file}")

        # Random delay between requests (2-5 seconds)
        delay = random.uniform(2, 5)
        time.sleep(delay)

    # Create DataFrame from results
    results_df = pd.DataFrame(scraped_content)

    # Display summary
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETED")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults:")
    print(f"  ✓ Successful: {success_count}")
    print(f"  ✗ Failed: {failure_count}")
    print(f"  Total: {len(scraped_content)}")

    if success_count > 0:
        avg_word_count = results_df[results_df['success']]['word_count'].mean()
        print(f"\nAverage word count: {avg_word_count:.0f} words")

    # Save to CSV
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Data saved to: {output_file}")

    return results_df


def main():
    """Main execution function"""

    # Configuration
    INPUT_CSV = 'indian_express_articles_2024_to_2024.csv'
    OUTPUT_CSV = 'indian_express_article_content_2024.csv'

    # Optional: Set start index and limit for testing or resuming
    START_INDEX = 0  # Change this to resume from a specific point
    LIMIT = None  # Set to a number to limit how many articles to scrape (e.g., 100 for testing)

    # Run the scraper
    results = scrape_articles_from_csv(
        csv_file=INPUT_CSV,
        output_file=OUTPUT_CSV,
        start_index=START_INDEX,
        limit=LIMIT
    )

    print("\n" + "=" * 80)
    print("Sample of scraped data:")
    print("=" * 80)
    print(results[['title', 'author', 'published_date', 'word_count']].head())


if __name__ == "__main__":
    main()
