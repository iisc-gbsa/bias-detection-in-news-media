"""
Fetch diverse sample articles from MongoDB for Streamlit demo
Shows articles with different bias types: gender, religious, caste, region
"""

import json
from pymongo import MongoClient
from datetime import datetime

# Configuration
MONGO_HOST = "localhost"
MONGO_PORT = 27017
DATABASE = "bias_detection"
COLLECTION = "realtime_news_DEverything_B10000_Topic6"
OUTPUT_FILE = "sample_articles.json"


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def connect_to_mongodb():
    """Connect to MongoDB and return collection"""
    log(f"Connecting to MongoDB at {MONGO_HOST}:{MONGO_PORT}...")
    client = MongoClient(MONGO_HOST, MONGO_PORT, serverSelectionTimeoutMS=5000)

    # Test connection
    client.server_info()
    log("✓ Connected to MongoDB successfully")

    db = client[DATABASE]
    collection = db[COLLECTION]

    # Get collection stats
    count = collection.count_documents({})
    log(f"✓ Collection '{COLLECTION}' has {count:,} documents")

    return collection


def fetch_sample(collection, query, fields, description):
    """Fetch a single sample matching the query"""
    log(f"  Searching for: {description}...")

    doc = collection.find_one(query, fields)

    if doc:
        title = doc.get("title", "No title")[:60]
        log(f'  ✓ Found: "{title}..."')
        return doc
    else:
        log(f"  ✗ No match found")
        return None


def fetch_diverse_samples(collection):
    """Fetch articles with diverse bias types"""
    samples = []
    fields = {
        "title": 1,
        "article_text": 1,
        "gender_bias": 1,
        "gender_type": 1,
        "religious_bias": 1,
        "religious_type": 1,
        "caste_bias": 1,
        "caste_type": 1,
        "region_bias": 1,
        "region_type": 1,
        "socioeconomic_bias": 1,
        "socioeconomic_type": 1,
        "political_bias": 1,
        "political_type": 1,
        "overall_bias_score": 1,
    }

    log("\n" + "=" * 60)
    log("FETCHING GENDER BIAS SAMPLES")
    log("=" * 60)

    # Gender - male bias
    doc = fetch_sample(
        collection,
        {"gender_type": "male_bias", "gender_bias": {"$gt": 0.3}},
        fields,
        "Male gender bias (score > 0.3)",
    )
    if doc:
        samples.append(
            {
                "category": "Gender Bias (Male)",
                "description": f"Gender bias: {doc.get('gender_bias', 0):.3f} - {doc.get('gender_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    # Gender - female bias
    doc = fetch_sample(
        collection,
        {"gender_type": "female_bias", "gender_bias": {"$gt": 0.3}},
        fields,
        "Female gender bias (score > 0.3)",
    )
    if doc:
        samples.append(
            {
                "category": "Gender Bias (Female)",
                "description": f"Gender bias: {doc.get('gender_bias', 0):.3f} - {doc.get('gender_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    log("\n" + "=" * 60)
    log("FETCHING RELIGIOUS BIAS SAMPLES")
    log("=" * 60)

    # Religious bias - hindu focus
    doc = fetch_sample(
        collection,
        {"religious_type": "hindu_focus", "religious_bias": {"$gt": 0.2}},
        fields,
        "Hindu religious focus (score > 0.2)",
    )
    if doc:
        samples.append(
            {
                "category": "Religious Bias (Hindu)",
                "description": f"Religious bias: {doc.get('religious_bias', 0):.3f} - {doc.get('religious_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    # Religious bias - muslim focus
    doc = fetch_sample(
        collection,
        {"religious_type": "muslim_focus", "religious_bias": {"$gt": 0.2}},
        fields,
        "Muslim religious focus (score > 0.2)",
    )
    if doc:
        samples.append(
            {
                "category": "Religious Bias (Muslim)",
                "description": f"Religious bias: {doc.get('religious_bias', 0):.3f} - {doc.get('religious_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    # Any non-neutral religious
    doc = fetch_sample(
        collection,
        {
            "religious_type": {"$nin": ["neutral", "hindu_focus", "muslim_focus"]},
            "religious_bias": {"$gt": 0.2},
        },
        fields,
        "Other religious bias (score > 0.2)",
    )
    if doc:
        samples.append(
            {
                "category": f"Religious Bias ({doc.get('religious_type', 'other')})",
                "description": f"Religious bias: {doc.get('religious_bias', 0):.3f} - {doc.get('religious_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    log("\n" + "=" * 60)
    log("FETCHING CASTE BIAS SAMPLES")
    log("=" * 60)

    # Caste - upper caste focus
    doc = fetch_sample(
        collection,
        {"caste_type": "upper_caste_focus", "caste_bias": {"$gt": 0.3}},
        fields,
        "Upper caste focus (score > 0.3)",
    )
    if doc:
        samples.append(
            {
                "category": "Caste Bias (Upper)",
                "description": f"Caste bias: {doc.get('caste_bias', 0):.3f} - {doc.get('caste_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    # Caste - lower caste focus
    doc = fetch_sample(
        collection,
        {"caste_type": "lower_caste_focus", "caste_bias": {"$gt": 0.3}},
        fields,
        "Lower caste focus (score > 0.3)",
    )
    if doc:
        samples.append(
            {
                "category": "Caste Bias (Lower)",
                "description": f"Caste bias: {doc.get('caste_bias', 0):.3f} - {doc.get('caste_type')}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    log("\n" + "=" * 60)
    log("FETCHING REGIONAL BIAS SAMPLES")
    log("=" * 60)

    # Region - various
    for region in [
        "north_focus",
        "south_focus",
        "east_focus",
        "west_focus",
        "northeast_focus",
    ]:
        doc = fetch_sample(
            collection,
            {"region_type": region, "region_bias": {"$gt": 0.2}},
            fields,
            f"{region.replace('_', ' ').title()} (score > 0.2)",
        )
        if doc:
            samples.append(
                {
                    "category": f"Regional Bias ({region.replace('_focus', '').title()})",
                    "description": f"Region bias: {doc.get('region_bias', 0):.3f} - {doc.get('region_type')}",
                    "title": doc.get("title", ""),
                    "article_text": doc.get("article_text", ""),
                }
            )

    log("\n" + "=" * 60)
    log("FETCHING OVERALL BIAS SAMPLES")
    log("=" * 60)

    # High overall bias
    doc = fetch_sample(
        collection,
        {
            "overall_bias_score": {"$gt": 0.4},
            "article_text": {"$exists": True, "$ne": ""},
        },
        fields,
        "High overall bias (score > 0.4)",
    )
    if doc:
        samples.append(
            {
                "category": "High Overall Bias",
                "description": f"Overall score: {doc.get('overall_bias_score', 0):.3f}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    # Low/neutral bias
    doc = fetch_sample(
        collection,
        {
            "overall_bias_score": {"$lt": 0.1},
            "article_text": {"$exists": True, "$ne": "", "$not": {"$type": "null"}},
        },
        fields,
        "Low/neutral bias (score < 0.1)",
    )
    if doc:
        samples.append(
            {
                "category": "Low Bias (Neutral)",
                "description": f"Overall score: {doc.get('overall_bias_score', 0):.3f}",
                "title": doc.get("title", ""),
                "article_text": doc.get("article_text", ""),
            }
        )

    return samples


def save_samples(samples):
    """Save samples to JSON file"""
    log(f"\nSaving {len(samples)} samples to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False, default=str)

    log(f"✓ Saved successfully")


def print_summary(samples):
    """Print summary of fetched samples"""
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    for i, sample in enumerate(samples, 1):
        title = (
            sample["title"][:50] + "..."
            if len(sample["title"]) > 50
            else sample["title"]
        )
        text_len = len(sample.get("article_text", ""))
        log(f"{i:2}. [{sample['category']}]")
        log(f"    Title: {title}")
        log(f"    {sample['description']} | {text_len} chars")

    log(f"\nTotal samples collected: {len(samples)}")


def main():
    log("=" * 60)
    log("SAMPLE ARTICLE FETCHER")
    log("=" * 60)
    log(f"Database: {DATABASE}")
    log(f"Collection: {COLLECTION}")
    log("=" * 60 + "\n")

    try:
        collection = connect_to_mongodb()
        samples = fetch_diverse_samples(collection)

        if samples:
            save_samples(samples)
            print_summary(samples)
            log(f"\n✓ Done! Samples saved to {OUTPUT_FILE}")
        else:
            log("\n✗ No samples found matching criteria")

    except Exception as e:
        log(f"\n✗ Error: {e}")
        raise


if __name__ == "__main__":
    main()
