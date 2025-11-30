from pymongo import MongoClient


def main():
    client = MongoClient("mongodb://localhost:27017")
    db = client["test"]
    articles = db["articles"]

    # Query for documents where published_date is missing/invalid
    missing_published_query = {
        "$or": [
            {"published_date": {"$exists": False}},
            {"published_date": None},
            {"published_date": ""},
            {"published_date": "None"},
        ]
    }

    # Require scrape_date to be present and non-empty
    scrape_query = {"scrape_date": {"$exists": True, "$nin": [None, "", "None"]}}

    bulk_query = {"$and": [missing_published_query, scrape_query]}

    print("Running bulk update_many to backfill published_date from scrape_date...")
    result = articles.update_many(
        bulk_query,
        [
            {"$set": {"published_date": "$scrape_date"}},
        ],
    )

    print(f"Matched docs:  {result.matched_count}")
    print(f"Updated docs:  {result.modified_count}")

    # Normalize ISO datetime-like published_date strings to plain YYYY-MM-DD
    # Example: "2025-11-23T13:01:41+05:30" -> "2025-11-23"
    normalize_query = {
        "published_date": {
            "$type": "string",
            "$regex": r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T",
        }
    }

    print("Normalizing published_date strings to YYYY-MM-DD format...")
    norm_result = articles.update_many(
        normalize_query,
        [{"$set": {"published_date": {"$substrCP": ["$published_date", 0, 10]}}}],
    )

    print(f"Normalized docs: {norm_result.modified_count}")

    client.close()


if __name__ == "__main__":
    main()
