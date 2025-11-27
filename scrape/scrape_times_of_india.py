"""
Times of India Archive Article Scraper

This script scrapes article URLs from The Times of India archive pages.
Archive URL format: https://timesofindia.indiatimes.com/YYYY/MM/DD/archivelist/...
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
import threading

# Base URL for Times of India Archive
BASE_URL = "https://timesofindia.indiatimes.com"


def extract_article_id(url):
    """Extract article ID from URL."""
    match = re.search(r"/articleshow/(\d+)", url)
    return match.group(1) if match else None


def clean_content(content):
    """Clean the content and title."""
    unwanted_phrases = [
        "click here",
        "Click Here",
        "advertisement",
        "Subscribe",
        "Read More",
        "Learn More",
        "Join Now",
        "Get Started",
        "Sign Up",
        "Buy Now",
        "Limited Time Offer",
        "Act Now",
        "Don't Miss Out",
        "Exclusive Deal",
        "Shop Now",
        "Download Now",
        "Try for Free",
        "Free Trial",
        "Register Now",
        "See More",
        "Follow Us",
        "Stay Updated",
        "Get Updates",
        "Explore More",
        "More Info",
        "This Just In",
        "Breaking News",
        "Today's Deals",
        "YOU MAY LIKE",
        "Post comment",
        "You can now subscribe to our",
        "WhatsApp channel",
        "Preview Sample",
        "Download the app",
        "Download app",
        "Join the community",
        "Also Read",
        "Recommended For You",
        "Times of India",
        "TOI Plus",
        "End of Article",
        "ON SOCIAL MEDIA",
        "Trending",
        "Box Office Collection",
        "News entertainment hindi bollywood",
        "News entertainment",
        "News India News",
        "India News",
        "Text Size Small Medium Large",
        "TOI TimesPoints",
        "HOW TO EARN",
        "MY ACTIVITY",
        "DAILY CHECK-INS",
        "REDEEM YOUR TIMES POINTS",
        "TODAYS ACTIVITY",
        "REDEEM POINTS",
        "Visit TOI Daily",
        "KNOW MORE",
        "Read ePaper",
        "Edit Profile",
        "My Subscription",
        "Redeem Benefits",
        "Epaper Preferences",
        "Notification Center",
        "Sign In",
        "LOGOUT",
        "Read All Comments",
        "Back to Top",
        "Continue without login",
        "Refrain from posting comments",
        "Help us delete comments",
        "HIDE",
        "All Comments",
        "Your Activity",
        "Sort UpVoted Newest Oldest",
        "Be the first one to review",
        "We have sent you a verification email",
        "more from Cities",
        "full coverage",
        "More From TOI",
        "Featured Today",
        "Successfully logged in",
        "Enjoy reading",
    ]

    # Remove unwanted phrases (case-insensitive)
    for phrase in unwanted_phrases:
        content = re.sub(re.escape(phrase), "", content, flags=re.IGNORECASE)

    # Remove repetitive navigation patterns (e.g., "Pushpa 2 Review Pushpa 2 Box Office...")
    # This removes patterns where the same short phrases repeat multiple times
    content = re.sub(
        r"\b(\w+\s+\w+)\s+\1\s+\1", r"\1", content
    )  # Remove triple repetitions
    content = re.sub(
        r"\b(\w+\s+\w+\s+\w+)\s+\1", r"\1", content
    )  # Remove double repetitions of 3-word phrases

    # Remove common UI patterns
    content = re.sub(
        r"News\s*/\s*City\s+News\s*/\s*\w+\s+News\s*/", "", content
    )  # Breadcrumb navigation
    content = re.sub(
        r"News\s+City\s+News\s+\w+\s+News\s+", "", content
    )  # Breadcrumb without slashes
    content = re.sub(r"Topics\s*:\s*[\w\s,]+(?=\w+\s*:)", "", content)  # Topics section
    content = re.sub(
        r"Get\s+an?\s+chance\s+to\s+win.*?(?:survey|here|now)",
        "",
        content,
        flags=re.IGNORECASE,
    )  # Promotional content

    # Remove HTML tags if any remain
    content = re.sub(r"<[^>]+>", "", content)

    # Remove invalid characters (non-ASCII and words containing unusual symbols)
    content = re.sub(r"[^\x00-\x7F]+", "", content)

    # Clean up formatting
    content = re.sub(
        r"\s+", " ", content
    )  # Replace multiple whitespace with a single space
    content = re.sub(r"(?<=[a-zA-Z])(?=\d)", " ", content)  # Add space before digits
    content = re.sub(
        r"(?<=[\w])(?=[\W])", " ", content
    )  # Add space before non-word characters

    return content.strip()


# Initial starttime value (base value for calculation)
# This value needs to be calibrated based on TOI's archive system
INITIAL_STARTTIME = 43829  # Adjust this based on actual TOI archive structure

# Progress tracking
CACHE_DIR = "cache_toi"
PROGRESS_FILE = os.path.join(CACHE_DIR, "scraping_progress.json")

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "test"
MONGO_COLLECTION = "articles"


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
    Extract the full content of an article from Times of India.
    Adapted from Indian Express extraction logic.

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

        # Try article ID-based extraction first
        article_id = extract_article_id(url)

        # Extract title (TOI specific)
        title_tag = soup.find("h1", class_="HNMDR")
        if not title_tag:
            title_tag = soup.find("h1", class_="_2yIBL")
        if not title_tag:
            title_tag = soup.find("h1")

        # Use document title as fallback
        if title_tag:
            article_data["title"] = clean_content(title_tag.get_text(strip=True))
        if not article_data["title"]:
            doc_title = soup.find("title")
            if doc_title:
                # Clean document title (remove site name, etc.)
                title_text = doc_title.get_text(strip=True)
                # Remove common suffixes like "- Times of India"
                title_text = re.sub(
                    r"\s*[-|–]\s*(The\s+)?Times of India.*$",
                    "",
                    title_text,
                    flags=re.IGNORECASE,
                )
                title_text = re.sub(
                    r"\s*[-|–]\s*TOI.*$", "", title_text, flags=re.IGNORECASE
                )
                article_data["title"] = clean_content(title_text)

        # Extract author (TOI specific)
        author_tag = soup.find("div", class_="byline")
        if not author_tag:
            author_tag = soup.find("a", rel="author")
        if author_tag:
            author_text = author_tag.get_text(strip=True)
            author_text = re.sub(
                r"^(Written by|By|Author:|TNN\s*\|)\s*",
                "",
                author_text,
                flags=re.IGNORECASE,
            )
            article_data["author"] = author_text

        # Extract published date from meta tags
        date_meta = soup.find("meta", property="article:published_time")
        if date_meta:
            article_data["published_date"] = date_meta.get("content")

        # Try modified date
        modified_meta = soup.find("meta", property="article:modified_time")
        if modified_meta:
            article_data["modified_date"] = modified_meta.get("content")

        # Extract section/category
        section_meta = soup.find("meta", property="article:section")
        if section_meta:
            article_data["section"] = section_meta.get("content")

        # Extract tags
        tag_meta = soup.find("meta", {"name": "keywords"})
        if tag_meta:
            article_data["tags"] = tag_meta.get("content")

        # Extract article text/body - Simplified approach
        # Find the article container using multiple fallback selectors
        article_body = None

        # Try various selectors in order of specificity
        selectors = [
            (
                ("div", {"id": f"toi-article-container-{article_id}"})
                if article_id
                else None
            ),
            ("div", {"class": "_s30J clearfix"}),
            ("div", {"class": "_s30J"}),
            ("div", {"class": "_3YYSt"}),
            ("div", {"class": "ga-headlines"}),
            ("div", {"itemprop": "articleBody"}),
            ("article", {}),
            ("main", {}),
        ]

        for selector in selectors:
            if selector is None:
                continue
            tag, attrs = selector
            article_body = soup.find(tag, attrs)
            if article_body:
                break

        # Last resort: find div with substantial content
        if not article_body:
            potential_divs = soup.find_all("div", class_=True)
            for div in potential_divs:
                classes = " ".join(div.get("class", []))
                if "article" in classes.lower() or "content" in classes.lower():
                    text_length = len(div.get_text(strip=True))
                    if text_length > 200:  # Has substantial text
                        article_body = div
                        break

        # Extract all text from article body
        raw_article_text = ""
        if article_body:
            # Remove unwanted elements
            for element in article_body.find_all(
                ["script", "style", "nav", "header", "footer", "aside", "iframe"]
            ):
                element.decompose()

            # Remove ads, navigation, and polling elements by class/id
            for element in article_body.find_all(
                class_=lambda x: x
                and any(
                    keyword in str(x).lower()
                    for keyword in [
                        "ad",
                        "advertisement",
                        "promo",
                        "social",
                        "share",
                        "poll",
                        "trending",
                        "recommend",
                    ]
                )
            ):
                element.decompose()

            # Remove specific TOI ad/poll blocks
            for element in article_body.find_all("div", {"data-type": "in_view"}):
                element.decompose()
            for element in article_body.find_all(
                "div", id=lambda x: x and "poll" in str(x).lower()
            ):
                element.decompose()

            # Remove navigation and trending sections
            for element in article_body.find_all(["nav", "ul", "ol"], class_=True):
                # Remove if it contains navigation-like content
                text = element.get_text(strip=True)
                if any(
                    keyword in text.lower()
                    for keyword in ["trending", "popular", "most read", "recommended"]
                ):
                    element.decompose()

            # Extract ALL remaining text
            raw_article_text = article_body.get_text(separator=" ", strip=True)

        # Clean the extracted text using existing clean_content function
        article_data["article_text"] = (
            clean_content(raw_article_text) if raw_article_text else ""
        )
        article_data["word_count"] = (
            len(article_data["article_text"].split())
            if article_data["article_text"]
            else 0
        )

        # If no article text found, try JSON-LD data
        if not article_data["article_text"] or article_data["word_count"] < 50:
            json_ld = soup.find("script", type="application/ld+json")
            if json_ld:
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
                                [a.get("name", "") for a in data["author"]]
                            )
                    if not article_data["published_date"] and "datePublished" in data:
                        article_data["published_date"] = data["datePublished"]
                    if not article_data["modified_date"] and "dateModified" in data:
                        article_data["modified_date"] = data["dateModified"]
                except json.JSONDecodeError:
                    pass

        return article_data

    except requests.exceptions.Timeout:
        return {"success": False, "url": url, "error": "Timeout"}
    except requests.exceptions.RequestException as e:
        return {"success": False, "url": url, "error": str(e)}
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


def scrape_toi_articles_for_date(year, month, day):
    """
    Scrape article links for a specific date from Times of India archive.

    TOI uses a unique archive URL structure with a 'starttime' parameter
    that increments daily from a base date.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day (1-31)

    Returns:
        list: List of dictionaries containing article information
    """
    # Format date string
    date_str = f"{year}-{month:02d}-{day:02d}"
    date_object = datetime(year, month, day)

    # Calculate the number of days since January 1, 2020
    days_since_start = (date_object - datetime(2020, 1, 1)).days

    # Calculate starttime parameter
    starttime = INITIAL_STARTTIME + days_since_start

    # Construct the URL for the specific date
    url = f"{BASE_URL}/{year}/{month}/{day}/archivelist/year-{year},month-{month},starttime-{starttime}.cms"
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

        # Find all article links on the page and filter out ads
        articles = soup.find_all("a", href=True)

        # Extracting article links and returning them
        article_links = []
        for article in articles:
            link = article["href"]

            # Check if the link is an article link and not an ad or unrelated link
            if link.startswith("/") or link.startswith("http"):
                full_link = BASE_URL + link if link.startswith("/") else link

                # Filter criteria to exclude ads or unrelated links
                # TOI-specific filtering logic
                if "articles" in full_link or "news" in full_link:
                    # Add to list if not already present
                    if full_link not in [
                        item["Article Link"] for item in article_links
                    ]:
                        article_links.append(
                            {
                                "Media Name": "THE TIMES OF INDIA",
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


def batch_check_existing_urls(collection, urls):
    """
    Check which URLs already exist in MongoDB using a single batch query.

    Args:
        collection: MongoDB collection instance
        urls (list): List of URLs to check

    Returns:
        set: Set of URLs that already exist in the database
    """
    if not urls:
        return set()

    # Use $in operator to find all matching URLs in a single query
    existing_docs = collection.find({"url": {"$in": urls}}, {"url": 1})
    existing_urls = {doc["url"] for doc in existing_docs}

    return existing_urls


def process_single_article(article_info, collection, stats_lock, stats):
    """
    Process a single article: extract content and save to MongoDB.
    Thread-safe function for parallel processing.
    Note: Duplicate check is now done in batch before threading.

    Args:
        article_info (dict): Article metadata (URL, date, etc.)
        collection: MongoDB collection instance
        stats_lock: Threading lock for updating stats
        stats (dict): Statistics dictionary

    Returns:
        tuple: (success, message)
    """
    article_url = article_info["Article Link"]

    try:
        print(f"  Extracting content from: {article_url}")

        # Extract article content
        content = extract_article_content(article_url)

        if content["success"]:
            # Skip if word count is zero
            if content.get("word_count", 0) == 0:
                with stats_lock:
                    stats["zero_word_count_skipped"] += 1
                return (
                    False,
                    f"    ⚠ Skipping - Zero word count (likely not an article)",
                )

            # Add metadata from URL scraping
            content["media_name"] = article_info["Media Name"]
            content["scrape_date"] = article_info["Date"]
            content["scraped_at"] = datetime.now().isoformat()

            # Insert into MongoDB
            try:
                collection.insert_one(content)
                with stats_lock:
                    stats["new_articles_added"] += 1

                title_preview = content.get("title", "N/A")[:60]
                word_count = content.get("word_count", 0)
                return (
                    True,
                    f"    ✓ Added to MongoDB - Title: {title_preview}...\n      Word count: {word_count}",
                )
            except Exception as e:
                with stats_lock:
                    stats["extraction_failures"] += 1
                return (False, f"    ✗ MongoDB insert error: {str(e)}")
        else:
            with stats_lock:
                stats["extraction_failures"] += 1
            return (
                False,
                f"    ✗ Extraction failed: {content.get('error', 'Unknown')}",
            )

    except Exception as e:
        with stats_lock:
            stats["extraction_failures"] += 1
        return (False, f"    ✗ Exception: {str(e)}")


def scrape_toi_articles(start_year=2020, end_year=2024, use_cache=True, max_workers=5):
    """
    Scrape articles for a date range with caching support and MongoDB storage.
    Uses parallel processing for article extraction.

    Args:
        start_year (int): Starting year
        end_year (int): Ending year (inclusive)
        use_cache (bool): Whether to use cache and resume from last position
        max_workers (int): Maximum number of parallel workers for article extraction

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
    print(f"Using {max_workers} parallel workers for article extraction\n")

    # Get MongoDB collection
    collection = get_mongo_collection()

    # Thread lock for stats updates
    stats_lock = threading.Lock()

    stats = {
        "total_urls_found": 0,
        "new_articles_added": 0,
        "duplicates_skipped": 0,
        "zero_word_count_skipped": 0,
        "extraction_failures": 0,
    }

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

                # Skip if already scraped
                if use_cache and date_str in completed_dates:
                    print(f"Skipping {date_str} (already scraped)")
                    continue

                try:
                    article_urls = scrape_toi_articles_for_date(year, month, day)
                    stats["total_urls_found"] += len(article_urls)

                    if not article_urls:
                        # Mark date as completed even if no articles found
                        if use_cache:
                            completed_dates.add(date_str)
                            save_progress(list(completed_dates), date_str)
                        continue

                    # Batch check for existing URLs to avoid duplicate processing
                    print(f"  Checking for duplicates in batch...")
                    all_urls = [article["Article Link"] for article in article_urls]
                    existing_urls = batch_check_existing_urls(collection, all_urls)

                    # Filter out articles that already exist
                    new_articles = [
                        article
                        for article in article_urls
                        if article["Article Link"] not in existing_urls
                    ]

                    duplicates_found = len(article_urls) - len(new_articles)
                    if duplicates_found > 0:
                        with stats_lock:
                            stats["duplicates_skipped"] += duplicates_found
                        print(
                            f"  Found {duplicates_found} duplicates, processing {len(new_articles)} new articles"
                        )

                    if not new_articles:
                        print(
                            f"  All articles for {date_str} already exist in database"
                        )
                        # Mark date as completed
                        if use_cache:
                            completed_dates.add(date_str)
                            save_progress(list(completed_dates), date_str)
                        continue
                    
                    filtered_article_urls = []
                    for article_url in new_articles:
                        if article_url['Article Link'] not in ['https://www.facebook.com/toiworldnews']:
                            filtered_article_urls.append(article_url)

                    # Process only new articles in parallel using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit all article processing tasks
                        future_to_article = {
                            executor.submit(
                                process_single_article,
                                article_info,
                                collection,
                                stats_lock,
                                stats,
                            ): article_info
                            for article_info in filtered_article_urls
                        }

                        # Process completed tasks as they finish
                        for future in as_completed(future_to_article):
                            article_info = future_to_article[future]
                            try:
                                success, message = future.result()
                                print(message)

                                # Small delay to avoid overwhelming the server
                                time.sleep(random.uniform(0.5, 1.5))
                            except Exception as e:
                                print(
                                    f"    ✗ Task exception for {article_info['Article Link']}: {str(e)}"
                                )
                                with stats_lock:
                                    stats["extraction_failures"] += 1

                    # Mark date as completed
                    if use_cache:
                        completed_dates.add(date_str)
                        save_progress(list(completed_dates), date_str)

                except Exception as e:
                    print(f"Error on {date_str}: {e}")
                    continue

                # Random delay between dates
                time.sleep(random.uniform(1, 2))

    return stats


def main():
    """Main execution function"""
    print("=" * 80)
    print("TIMES OF INDIA ARCHIVE SCRAPER (PARALLEL)")
    print("=" * 80)

    # Configure date range and parallelization here
    START_YEAR = 2020
    END_YEAR = 2025
    MAX_WORKERS = 20  # Number of parallel workers for article extraction

    print(f"\nScraping articles from {START_YEAR} to {END_YEAR}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Scrape articles and extract content to MongoDB
    stats = scrape_toi_articles(
        start_year=START_YEAR,
        end_year=END_YEAR,
        use_cache=True,
        max_workers=MAX_WORKERS,
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
    print(f"- Zero word count skipped: {stats['zero_word_count_skipped']}")
    print(f"- Extraction failures: {stats['extraction_failures']}")
    print(f"\nProgress tracking saved in: {CACHE_DIR}/")
    print(f"MongoDB: {MONGO_URI}{MONGO_DB}/{MONGO_COLLECTION}")


if __name__ == "__main__":
    main()
