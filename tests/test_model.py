import argparse
import glob
import os
from typing import List

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging
import time
import numpy as np

try:
    import torch
except Exception:
    torch = None

DEFAULT_INPUT_DIR = (
    "/Users/sngirish/personal_work/mtech-learning/data-science/project/Media-Bias-Analysis-in-Indian-News-Articles/articles/cat"
)
DEFAULT_OUTPUT_DIR = (
    "/Users/sngirish/personal_work/mtech-learning/data-science/project/Media-Bias-Analysis-in-Indian-News-Articles/articles/bias"
)


def load_csv(path: str, preferred_text_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    candidates: List[str] = [preferred_text_col] if preferred_text_col else []
    candidates += ["text", "article_text", "content", "Article Text", "article", "body", "description"]
    text_col = next((c for c in candidates if c and c in df.columns), None)
    if text_col is None:
        raise ValueError(
            f"No text column found in {path}. Available: {list(df.columns)}"
        )
    return df, text_col


def preprocess_text(text: str, max_length: int = 512) -> str:
    """Preprocess and truncate text to handle length issues"""
    if not isinstance(text, str):
        text = str(text)

    # Simple word-based truncation (approximate)
    words = text.split()
    # Ensure at least 1 word is kept, max_length is token count so divide by 2 for words
    # But ensure minimum of 1 word
    max_words = max(1, max_length // 2) if max_length > 0 else len(words)
    if len(words) > max_words:
        text = ' '.join(words[:max_words])

    return text


def init_bias_pipeline(device: int | None = None):
    """Initialize bias detection pipeline with better models"""

    # Updated model candidates with better working models
    model_candidates = [
        # Sentiment models that can proxy for bias
        {
            "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "type": "sentiment",
            "labels": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        },
        # Emotion detection (can indicate bias)
        {
            "name": "j-hartmann/emotion-english-distilroberta-base",
            "type": "emotion",
            "labels": None
        },
        # Zero-shot for custom bias detection
        {
            "name": "facebook/bart-large-mnli",
            "type": "zero-shot",
            "labels": None
        },
        # Toxicity detection
        {
            "name": "unitary/toxic-bert",
            "type": "toxicity",
            "labels": {"TOXIC": "biased", "NON_TOXIC": "neutral"}
        },
        # Fallback
        {
            "name": "distilbert-base-uncased-finetuned-sst-2-english",
            "type": "sentiment",
            "labels": {"NEGATIVE": "negative", "POSITIVE": "positive"}
        }
    ]

    last_err = None

    for model_info in model_candidates:
        try:
            model_name = model_info["name"]
            model_type = model_info["type"]

            chosen_device = device
            if chosen_device is None:
                if torch is not None and torch.cuda.is_available():
                    chosen_device = 0
                else:
                    chosen_device = -1

            if model_type == "zero-shot":
                clf = pipeline("zero-shot-classification", model=model_name, device=chosen_device)
            else:
                clf = pipeline("text-classification", model=model_name, device=chosen_device)

            logging.info(f"Successfully initialized {model_type} model: {model_name} on device={chosen_device}")
            return clf, model_info

        except Exception as e:
            last_err = e
            logging.warning(f"Failed initializing model {model_info['name']}: {e}")
            continue

    raise RuntimeError(f"Failed to initialize any bias detection pipeline. Last error: {last_err}")


def calculate_bias_score(prediction, model_info):
    """Calculate unified bias score from different model outputs"""

    model_type = model_info["type"]

    if model_type == "zero-shot":
        # For zero-shot classification
        bias_labels = ["biased", "partisan", "unfair", "prejudiced", "propaganda"]

        top_label = prediction["labels"][0].lower()
        top_score = prediction["scores"][0]

        if any(bias_word in top_label for bias_word in bias_labels):
            return top_score, "biased"
        else:
            return 1.0 - top_score, "neutral"

    elif model_type == "sentiment":
        # Use sentiment extremes as bias indicators
        label = prediction["label"]
        score = prediction["score"]

        # High confidence extreme sentiments might indicate bias
        if ("negative" in label.lower() or "positive" in label.lower()) and score > 0.8:
            return score * 0.7, "potential_bias"  # Scale down sentiment-based bias
        else:
            return score * 0.3, "neutral"

    elif model_type == "emotion":
        # Strong emotions might indicate bias
        label = prediction["label"].lower()
        score = prediction["score"]

        bias_emotions = ["anger", "disgust", "fear"]
        if any(emotion in label for emotion in bias_emotions) and score > 0.7:
            return score * 0.8, "emotional_bias"
        else:
            return score * 0.2, "neutral"

    elif model_type == "toxicity":
        # Direct toxicity detection
        label = prediction["label"]
        score = prediction["score"]

        if "TOXIC" in label:
            return score, "toxic_bias"
        else:
            return 1.0 - score, "neutral"

    else:
        # Default handling
        return prediction.get("score", 0.0), prediction.get("label", "unknown")


def score_bias_for_df(
    df: pd.DataFrame,
    text_col: str,
    clf,
    model_info: dict,
    batch_size: int = 8,
    max_length: int = 512,
) -> pd.DataFrame:

    # Preprocess texts to handle length issues
    texts = df[text_col].fillna("").astype(str).tolist()
    processed_texts = [preprocess_text(text, max_length) for text in texts]

    logging.info(f"Scoring {len(processed_texts)} texts with {model_info['type']} model")

    t0 = time.time()
    results: List[dict] = []

    # Define bias categories for zero-shot classification
    bias_categories = [
        "neutral and objective",
        "biased and partisan",
        "propaganda",
        "unfair reporting",
        "balanced journalism"
    ]

    # Process texts individually to avoid batching issues
    for i, text in enumerate(processed_texts):
        try:
            if model_info["type"] == "zero-shot":
                prediction = clf(text, bias_categories)
            else:
                prediction = clf(text, truncation=True, max_length=max_length)

            # Handle list responses
            if isinstance(prediction, list) and len(prediction) > 0:
                prediction = prediction[0]

            results.append(prediction)

            if (i + 1) % 20 == 0:
                logging.debug(f"Processed {i + 1}/{len(processed_texts)} texts")

        except Exception as e:
            logging.warning(f"Error processing text {i}: {e}")
            results.append({"label": "ERROR", "score": 0.0, "error": str(e)})

    logging.info(f"Finished scoring {len(processed_texts)} texts in {time.time() - t0:.2f}s")

    # Calculate unified bias scores
    bias_scores = []
    bias_labels = []
    raw_predictions = []

    for r in results:
        try:
            if isinstance(r, dict) and "error" not in r:
                bias_score, bias_label = calculate_bias_score(r, model_info)
                bias_scores.append(float(bias_score))
                bias_labels.append(bias_label)
                raw_predictions.append(str(r))
            else:
                bias_scores.append(0.0)
                bias_labels.append("error")
                raw_predictions.append(str(r))
        except Exception as e:
            logging.warning(f"Error processing result {r}: {e}")
            bias_scores.append(0.0)
            bias_labels.append("error")
            raw_predictions.append(str(r))

    # Create output dataframe
    df_out = df.copy()
    df_out["bias_score"] = bias_scores
    df_out["bias_label"] = bias_labels
    df_out["model_used"] = model_info["name"]
    df_out["model_type"] = model_info["type"]
    df_out["raw_prediction"] = raw_predictions
    df_out["processed_text_length"] = [len(text.split()) for text in processed_texts]

    # Adjusted bias categories with lower thresholds
    df_out["bias_category"] = df_out["bias_score"].apply(
        lambda x: "high_bias" if x > 0.5 else "medium_bias" if x > 0.3 else "low_bias"
    )

    return df_out


def process_all(
    input_dir: str,
    output_dir: str,
    preferred_text_col: str | None = None,
    batch_size: int = 8,
    max_length: int = 512,
    device: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    csvs = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

    if not csvs:
        print(f"No CSVs found in {input_dir}")
        logging.warning(f"No CSVs found in {input_dir}")
        return

    clf, model_info = init_bias_pipeline(device=device)
    print(f"Using bias detection model: {model_info['name']} (type: {model_info['type']})")
    logging.info(f"Using model {model_info['name']} of type {model_info['type']}")

    # Process only first 2 CSVs for testing
    csvs = csvs[:100]

    summary_stats = {
        "total_files": len(csvs),
        "successful": 0,
        "failed": 0,
        "total_articles": 0,
        "high_bias_articles": 0,
        "medium_bias_articles": 0
    }

    for idx, path in enumerate(csvs, 1):
        try:
            logging.info(f"[{idx}/{len(csvs)}] Reading {path}")
            df, text_col = load_csv(path, preferred_text_col)

            # Process only first 10 rows for testing
            df = df[:10]
            summary_stats["total_articles"] += len(df)

            logging.info(f"[{idx}/{len(csvs)}] Processing {len(df)} rows, text_col='{text_col}'")

            t_file = time.time()
            scored = score_bias_for_df(
                df,
                text_col,
                clf,
                model_info,
                batch_size=batch_size,
                max_length=max_length,
            )

            # Count bias articles
            high_bias_count = len(scored[scored["bias_category"] == "high_bias"])
            medium_bias_count = len(scored[scored["bias_category"] == "medium_bias"])
            summary_stats["high_bias_articles"] += high_bias_count
            summary_stats["medium_bias_articles"] += medium_bias_count

            # Save results
            base = os.path.basename(path)
            name, ext = os.path.splitext(base)
            out_path = os.path.join(output_dir, f"{name}_bias_analysis{ext}")
            scored.to_csv(out_path, index=False)

            print(f"[{idx}/{len(csvs)}] ✅ Processed {path}")
            print(f"    → Output: {out_path}")
            print(f"    → Articles: {len(df)}, High bias: {high_bias_count}, Medium bias: {medium_bias_count}")
            print(f"    → Avg bias score: {scored['bias_score'].mean():.3f}")
            print(f"    → Processing time: {time.time() - t_file:.2f}s")

            # Show sample results
            print("    → Sample results:")
            for i in range(min(3, len(scored))):
                row = scored.iloc[i]
                print(f"      Article {i + 1}: {row['bias_category']} (score: {row['bias_score']:.3f})")

            logging.info(f"[{idx}/{len(csvs)}] Successfully processed {path} in {time.time() - t_file:.2f}s")
            summary_stats["successful"] += 1

        except Exception as e:
            print(f"[{idx}/{len(csvs)}] ❌ Failed {path}: {e}")
            logging.exception(f"[{idx}/{len(csvs)}] Failed {path}: {e}")
            summary_stats["failed"] += 1

    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Files processed: {summary_stats['successful']}/{summary_stats['total_files']}")
    print(f"Total articles analyzed: {summary_stats['total_articles']}")
    print(f"High bias articles: {summary_stats['high_bias_articles']}")
    print(f"Medium bias articles: {summary_stats['medium_bias_articles']}")
    print(f"Model used: {model_info['name']} ({model_info['type']})")


def main():
    parser = argparse.ArgumentParser(description="Advanced bias detection for news articles.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=400)  # Reduced for better performance
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=== Enhanced News Article Bias Detection ===")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max length: {args.max_length}")
    print("Starting bias analysis...\n")

    process_all(
        args.input_dir,
        args.output_dir,
        args.text_column,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()
