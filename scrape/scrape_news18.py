"""
News18 Archive Article Scraper

This script scrapes article URLs from News18 archive pages.
Archive URL format: https://www.news18.com/archives/YYYY-MM-DD.html
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import os
import re
from datetime import datetime
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Base URL for News18 Archive
BASE_URL = "https://www.news18.com"

# Progress tracking
CACHE_DIR = "cache_news18"
PROGRESS_FILE = os.path.join(CACHE_DIR, "scraping_progress.json")

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "test"
MONGO_COLLECTION = "articles"

# Parallelization Configuration
MAX_WORKERS = 5  # Number of concurrent threads for article extraction


# Initialize MongoDB connection
def get_mongo_collection():
    """Get MongoDB collection instance."""
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]
    # Create index on URL to speed up duplicate checks
    collection.create_index("url", unique=True)
    return collection


def extract_article_content(url):
    """
    Extract the full content of an article from News18.

    Args:
        url (str): Article URL

    Returns:
        dict: Dictionary containing article content and metadata
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}"}

        soup = BeautifulSoup(response.content, "html.parser")

        # Initialize result dictionary
        article_data = {
            "success": True,
            "url": url,
            "title": None,
            "author": None,
            "published_date": None,
            "modified_date": None,
            "section": None,
            "tags": None,
            "article_text": None,
            "word_count": 0,
            "error": None,
        }

        # Extract title - News18 uses h1 for title
        title_tag = soup.find("h1", class_="jsx-ace90f60da59ed01")
        if not title_tag:
            title_tag = soup.find("h1")
        if title_tag:
            article_data["title"] = title_tag.get_text(strip=True)

        # Extract author from meta tag or author section
        author_meta = soup.find("meta", attrs={"name": "author"})
        if author_meta:
            article_data["author"] = author_meta.get("content")
        else:
            # Try to find author link
            author_link = soup.find("a", href=re.compile(r"/byline/"))
            if author_link:
                article_data["author"] = author_link.get_text(strip=True)

        # Extract published date from meta tags
        date_meta = soup.find("meta", property="article:published_time")
        if date_meta:
            article_data["published_date"] = date_meta.get("content")

        # Try modified date
        modified_meta = soup.find("meta", property="article:modified_time")
        if modified_meta:
            article_data["modified_date"] = modified_meta.get("content")

        # Extract section/category from URL or meta
        section_meta = soup.find("meta", property="article:section")
        if section_meta:
            article_data["section"] = section_meta.get("content")
        else:
            # Try to extract section from URL
            url_parts = url.replace(BASE_URL, "").split("/")
            if len(url_parts) > 1 and url_parts[1]:
                article_data["section"] = url_parts[1]

        # Extract tags
        tag_meta = soup.find("meta", property="article:tag")
        if tag_meta:
            article_data["tags"] = tag_meta.get("content")
        else:
            # Try to find keyword meta
            keywords_meta = soup.find("meta", attrs={"name": "keywords"})
            if keywords_meta:
                article_data["tags"] = keywords_meta.get("content")

        # Extract article text/body
        article_body = None

        # Try different selectors for News18 article content
        article_body = soup.find("div", class_="jsx-ace90f60da59ed01 content")
        if not article_body:
            article_body = soup.find("div", class_="article_content")
        if not article_body:
            article_body = soup.find("div", class_="story_content")
        if not article_body:
            article_body = soup.find("article")
        if not article_body:
            # Try finding the main content area by ID
            article_body = soup.find("div", id="article-body")
        if not article_body:
            # Fallback to finding paragraphs in main content
            article_body = soup.find(
                "div", class_=re.compile(r"content|article|story", re.I)
            )

        if article_body:
            paragraphs = article_body.find_all("p")
            article_text_parts = []
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 20:
                    if not any(
                        skip_word in text.lower()
                        for skip_word in [
                            "advertisement",
                            "also read",
                            "read more",
                            "subscribe now",
                            "follow us",
                            "download app",
                            "share your feedback",
                        ]
                    ):
                        article_text_parts.append(text)

            article_data["article_text"] = "\n\n".join(article_text_parts)
            article_data["word_count"] = len(article_data["article_text"].split())

        # If no article text found, try JSON-LD data
        if not article_data["article_text"] or article_data["word_count"] < 50:
            json_ld_scripts = soup.find_all("script", type="application/ld+json")
            for json_ld in json_ld_scripts:
                try:
                    data = json.loads(json_ld.string)
                    if isinstance(data, list):
                        data = data[0]

                    if "articleBody" in data:
                        article_data["article_text"] = data["articleBody"]
                        article_data["word_count"] = len(
                            article_data["article_text"].split()
                        )

                    if not article_data["title"] and "headline" in data:
                        article_data["title"] = data["headline"]
                    if not article_data["author"] and "author" in data:
                        if isinstance(data["author"], dict):
                            article_data["author"] = data["author"].get("name")
                        elif isinstance(data["author"], list):
                            article_data["author"] = ", ".join(
                                [
                                    a.get("name", "")
                                    for a in data["author"]
                                    if isinstance(a, dict)
                                ]
                            )
                    if not article_data["published_date"] and "datePublished" in data:
                        article_data["published_date"] = data["datePublished"]
                    if not article_data["modified_date"] and "dateModified" in data:
                        article_data["modified_date"] = data["dateModified"]

                    # Break if we found article body
                    if article_data["article_text"]:
                        break
                except json.JSONDecodeError:
                    pass
                except Exception:
                    pass

        return article_data

    except requests.exceptions.Timeout:
        return {"success": False, "url": url, "error": "Timeout"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "url": url, "error": str(e)}
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


def scrape_news18_articles_for_date(year, month, day):
    """
    Scrape article links for a specific date from News18 archive.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day (1-31)

    Returns:
        list: List of dictionaries containing article information
    """
    # Format date string
    date_str = f"{year}-{month:02d}-{day:02d}"

    # Construct the archive URL for the specific date
    # News18 archive format: https://www.news18.com/archives/YYYY-MM-DD.html
    url = f"{BASE_URL}/archives/{year}-{month:02d}-{day:02d}.html"
    print(f"Scraping URL: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
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

        # Find all article links on the page
        article_links = []
        seen_urls = set()

        # News18 article URLs typically follow patterns like:
        # /india/..., /world/..., /cricket/..., /movies/..., /business/..., etc.
        # They end with a numeric ID like -8724673.html
        article_pattern = re.compile(
            r"^https?://www\.news18\.com/[a-z0-9-]+/.*-\d+\.html$"
        )

        # Find all <a> tags with href
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Make full URL if it's a relative path
            if href.startswith("/"):
                full_link = BASE_URL + href
            elif href.startswith("http"):
                full_link = href
            else:
                continue

            # Skip non-article links
            if "/archives/" in full_link:
                continue
            if "/photogallery/" in full_link:
                continue
            if "/short-videos/" in full_link:
                continue
            if "/short-news/" in full_link:
                continue
            if "/web-stories/" in full_link:
                continue
            if "/livetv/" in full_link:
                continue
            if "/topics/" in full_link:
                continue
            if "/agency/" in full_link:
                continue
            if "/byline/" in full_link:
                continue

            # Check if it matches the article pattern
            if article_pattern.match(full_link):
                # Remove query parameters
                full_link = full_link.split("?")[0]

                # Add to list if not already present
                if full_link not in seen_urls:
                    seen_urls.add(full_link)
                    article_links.append(
                        {
                            "Media Name": "NEWS18",
                            "Article Link": full_link,
                            "Date": date_str,
                        }
                    )

        print(f"Found {len(article_links)} articles for {date_str}")
        return article_links

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return []


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


def process_single_article(article_info, collection, stats_lock):
    """
    Process a single article: check for duplicates, extract content, and save to MongoDB.

    Args:
        article_info (dict): Article metadata (URL, date, media name)
        collection: MongoDB collection instance
        stats_lock (Lock): Thread lock for updating statistics

    Returns:
        dict: Processing result with statistics
    """
    result = {
        "url": article_info["Article Link"],
        "duplicate": False,
        "success": False,
        "error": None,
    }

    article_url = article_info["Article Link"]

    try:
        # Check if URL already exists in MongoDB
        if collection.find_one({"url": article_url}):
            print(f"  Skipping duplicate: {article_url}")
            result["duplicate"] = True
            return result

        print(f"  Extracting content from: {article_url}")

        # Extract article content
        content = extract_article_content(article_url)

        if content["success"]:
            # Add metadata from URL scraping
            content["media_name"] = article_info["Media Name"]
            content["scrape_date"] = article_info["Date"]
            content["published_date"] = article_info["Date"]
            content["scraped_at"] = datetime.now().isoformat()

            # Insert into MongoDB
            try:
                collection.insert_one(content)
                result["success"] = True
                print(
                    f"    ✓ Added to MongoDB - Title: {content.get('title', 'N/A')[:60]}..."
                )
                print(f"      Word count: {content.get('word_count', 0)}")
            except Exception as e:
                result["error"] = f"MongoDB insert error: {str(e)}"
                print(f"    ✗ {result['error']}")
        else:
            result["error"] = f"Extraction failed: {content.get('error', 'Unknown')}"
            print(f"    ✗ {result['error']}")

        # Small delay to be respectful to the server
        time.sleep(random.uniform(1, 2))

    except Exception as e:
        result["error"] = str(e)
        print(f"    ✗ Error processing {article_url}: {str(e)}")

    return result


def scrape_news18_articles(
    start_year=2024, end_year=2025, use_cache=True, max_workers=MAX_WORKERS
):
    """
    Scrape articles for a date range with caching support and MongoDB storage.
    Uses parallel processing for faster extraction.

    Args:
        start_year (int): Starting year
        end_year (int): Ending year (inclusive)
        use_cache (bool): Whether to use cache and resume from last position
        max_workers (int): Number of concurrent threads for article extraction

    Returns:
        dict: Statistics of scraping operation
    """
    # Load progress if using cache
    progress = (
        load_progress() if use_cache else {"completed_dates": [], "last_date": None}
    )
    completed_dates = set(progress["completed_dates"])

    print(f"\nCache status: {len(completed_dates)} dates already scraped")
    if progress["last_date"]:
        print(f"Last scraped date: {progress['last_date']}")

    # Get MongoDB collection
    collection = get_mongo_collection()

    # Thread-safe lock for updating statistics
    stats_lock = Lock()

    stats = {
        "total_urls_found": 0,
        "new_articles_added": 0,
        "duplicates_skipped": 0,
        "extraction_failures": 0,
    }

    print(f"\nUsing {max_workers} concurrent threads for article extraction")

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get the number of days in the month
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

                # Skip future dates
                try:
                    current_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if current_date > datetime.now():
                        print(f"Skipping {date_str} (future date)")
                        continue
                except ValueError:
                    continue

                # Skip if already scraped
                if use_cache and date_str in completed_dates:
                    print(f"Skipping {date_str} (already scraped)")
                    continue

                try:
                    article_urls = scrape_news18_articles_for_date(year, month, day)
                    stats["total_urls_found"] += len(article_urls)

                    if not article_urls:
                        # No articles found for this date
                        if use_cache:
                            completed_dates.add(date_str)
                            save_progress(list(completed_dates), date_str)
                        continue

                    print(
                        f"  Processing {len(article_urls)} articles with {max_workers} threads..."
                    )

                    # Process articles in parallel using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all article processing tasks
                        future_to_article = {
                            executor.submit(
                                process_single_article,
                                article_info,
                                collection,
                                stats_lock,
                            ): article_info
                            for article_info in article_urls
                        }

                        # Process completed tasks
                        for future in as_completed(future_to_article):
                            try:
                                result = future.result()

                                # Update statistics (thread-safe)
                                with stats_lock:
                                    if result["duplicate"]:
                                        stats["duplicates_skipped"] += 1
                                    elif result["success"]:
                                        stats["new_articles_added"] += 1
                                    else:
                                        stats["extraction_failures"] += 1

                            except Exception as e:
                                print(f"    ✗ Thread error: {str(e)}")
                                with stats_lock:
                                    stats["extraction_failures"] += 1

                    # Mark date as completed
                    if use_cache:
                        completed_dates.add(date_str)
                        save_progress(list(completed_dates), date_str)

                    print(
                        f"  Completed {date_str} - Added: {stats['new_articles_added']}, Duplicates: {stats['duplicates_skipped']}, Failures: {stats['extraction_failures']}"
                    )

                except Exception as e:
                    print(f"Error on {date_str}: {e}")
                    continue

                # Random delay between dates
                time.sleep(random.uniform(1, 2))

    return stats


def main():
    """Main execution function"""
    print("=" * 80)
    print("NEWS18 ARCHIVE SCRAPER")
    print("=" * 80)

    # Configure date range here
    START_YEAR = 2024
    END_YEAR = 2025

    print(f"\nScraping articles from {START_YEAR} to {END_YEAR}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Scrape articles and extract content to MongoDB
    stats = scrape_news18_articles(
        start_year=START_YEAR, end_year=END_YEAR, use_cache=True
    )

    # Display results
    print("\n" + "=" * 80)
    print("SCRAPING COMPLETED")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nStatistics:")
    print(f"- Total URLs found: {stats['total_urls_found']}")
    print(f"- New articles added to MongoDB: {stats['new_articles_added']}")
    print(f"- Duplicates skipped: {stats['duplicates_skipped']}")
    print(f"- Extraction failures: {stats['extraction_failures']}")
    print(f"\nProgress tracking saved in: {CACHE_DIR}/")
    print(f"MongoDB: {MONGO_URI}{MONGO_DB}/{MONGO_COLLECTION}")


if __name__ == "__main__":
    main()
