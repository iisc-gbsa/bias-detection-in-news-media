import argparse
import os
import glob
import re
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ------------------------------
# IO directories (defaults to user's requested folders)
# ------------------------------
DEFAULT_INPUT_DIR = \
    "/Users/sngirish/personal_work/mtech-learning/data-science/project/Media-Bias-Analysis-in-Indian-News-Articles/articles/raw"
DEFAULT_OUTPUT_DIR = \
    "/Users/sngirish/personal_work/mtech-learning/data-science/project/Media-Bias-Analysis-in-Indian-News-Articles/articles/cat"


def load_text_df(input_csv: str, preferred_text_col: str | None = None) -> pd.DataFrame:
    """Load a CSV and return a DataFrame with a 'text' column and any useful ids."""
    src = pd.read_csv(input_csv)
    candidates = [preferred_text_col] if preferred_text_col else []
    candidates += ["article_text", "text", "Article Text", "article", "content"]
    text_col = next((c for c in candidates if c and c in src.columns), None)
    if text_col is None:
        raise ValueError(f"No text column found in {input_csv}. Available: {list(src.columns)}")

    out = pd.DataFrame({'text': src[text_col].fillna("")})
    # preserve common identifiers if present
    for keep_col in ["url", "title", "published_date", "media_name", "csv_index"]:
        if keep_col in src.columns:
            out[keep_col] = src[keep_col]
    return out

# ------------------------------
# Preprocessing
# ------------------------------


def preprocess_text(text: str) -> str:
    """Clean and preprocess text for topic modeling without NLTK downloads."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    stop_words = ENGLISH_STOP_WORDS
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


# ------------------------------
# LDA Topic Modeling
# ------------------------------
def perform_lda_topic_modeling(texts, n_topics=6):
    """
    Perform LDA topic modeling on texts.

    Args:
        texts: List of preprocessed texts
        n_topics: Number of topics to extract

    Returns:
        lda: Fitted LDA model
        vectorizer: CountVectorizer used
        doc_topic_dist: Document-topic distribution matrix
        topic_words: List of top words per topic
        feature_names: Feature names from vectorizer
    """
    # Vectorize the text
    vectorizer = CountVectorizer(
        max_df=0.8,
        min_df=2,
        max_features=100,
        ngram_range=(1, 2)
    )

    doc_term_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=100
    )

    lda.fit(doc_term_matrix)

    # Get topic-word distribution
    topic_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)

    # Get document-topic distribution
    doc_topic_dist = lda.transform(doc_term_matrix)

    return lda, vectorizer, doc_topic_dist, topic_words, feature_names


# ------------------------------
# Keyword-based classification (lightweight)
# ------------------------------
def keyword_based_topic_classification(df_in: pd.DataFrame) -> dict:
    topic_keywords = {
        'politics': ['government', 'policy', 'political', 'minister', 'election', 'parliament', 'ruling', 'opposition'],
        'sports': ['cricket', 'match', 'stadium', 'team', 'player', 'olympic', 'training', 'sports'],
        'technology': ['technology', 'software', 'digital', 'startup', 'innovation', 'tech', 'platform', 'data'],
        'entertainment': ['movie', 'bollywood', 'film', 'entertainment', 'box office', 'streaming', 'cultural'],
        'business': ['stock', 'market', 'economic', 'business', 'growth', 'budget', 'financial', 'economy'],
        'health': ['health', 'medical', 'healthcare', 'covid', 'vaccination', 'hospital', 'treatment'],
        'education': ['education', 'learning', 'school', 'student', 'university', 'skill', 'educational'],
        'environment': ['environment', 'climate', 'renewable', 'energy', 'clean', 'pollution', 'green'],
        'infrastructure': ['infrastructure', 'highway', 'railway', 'construction', 'development', 'project'],
        'social': ['social', 'festival', 'community', 'cultural', 'religious', 'harmony', 'celebration']
    }

    def classify_article(text: str):
        text_lower = text.lower()
        scores = {t: sum(1 for kw in kws if kw in text_lower) for t, kws in topic_keywords.items()}
        scores = {k: v for k, v in scores.items() if v > 0}
        if scores:
            best = max(scores, key=scores.get)
            return best, scores[best] / len(topic_keywords[best])
        return 'unknown', 0.0

    topics, confs = [], []
    for text in df_in['text']:
        t, c = classify_article(text)
        topics.append(t)
        confs.append(c)

    df_in['keyword_topic'] = topics
    df_in['keyword_confidence'] = confs
    return topic_keywords


# ------------------------------
# TF-IDF + K-means Clustering
# ------------------------------
def tfidf_clustering_topics(texts, n_clusters=5):
    """
    Use TF-IDF + K-means for topic discovery.

    Args:
        texts: List of preprocessed texts
        n_clusters: Number of clusters

    Returns:
        kmeans: Fitted KMeans model
        tfidf_vectorizer: TfidfVectorizer used
        cluster_labels: Cluster assignments
        feature_names: Feature names from vectorizer
    """
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.8,
        min_df=1,
        max_features=100,
        ngram_range=(1, 2),
        stop_words='english'
    )

    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)

    return kmeans, tfidf_vectorizer, cluster_labels, feature_names


# ------------------------------
# Pre-trained Model Classification
# ------------------------------
def use_pretrained_topic_models(texts, candidate_labels=None):
    """
    Use pre-trained zero-shot classification models.

    Args:
        texts: List of texts to classify
        candidate_labels: List of candidate topic labels

    Returns:
        results: List of classification results
        classifier: The classifier instance (or None if unavailable)
    """
    if candidate_labels is None:
        candidate_labels = [
            'politics', 'sports', 'technology', 'entertainment',
            'business', 'health', 'education', 'environment',
            'infrastructure', 'social issues'
        ]

    try:
        from transformers import pipeline

        classifier = pipeline("zero-shot-classification",
                              model="facebook/bart-large-mnli")

        results = []
        for text in texts:
            result = classifier(text, candidate_labels)
            results.append({
                'top_label': result['labels'][0],
                'top_score': result['scores'][0],
                'all_labels': result['labels'],
                'all_scores': result['scores']
            })

        return results, classifier

    except Exception as e:
        print(f"Warning: Zero-shot classifier unavailable: {e}")
        return None, None


# ------------------------------
# Ensemble Topic Classifier
# ------------------------------
class EnsembleTopicClassifier:
    """Combine multiple approaches for robust topic identification."""

    def __init__(self):
        self.methods = {}
        self.topic_mapping = {
            0: 'politics', 1: 'sports', 2: 'technology',
            3: 'entertainment', 4: 'business', 5: 'other'
        }
        self.topic_keywords = {
            'politics': ['government', 'policy', 'political', 'minister', 'election', 'parliament', 'ruling', 'opposition'],
            'sports': ['cricket', 'match', 'stadium', 'team', 'player', 'olympic', 'training', 'sports'],
            'technology': ['technology', 'software', 'digital', 'startup', 'innovation', 'tech', 'platform', 'data'],
            'entertainment': ['movie', 'bollywood', 'film', 'entertainment', 'box office', 'streaming', 'cultural'],
            'business': ['stock', 'market', 'economic', 'business', 'growth', 'budget', 'financial', 'economy'],
            'health': ['health', 'medical', 'healthcare', 'covid', 'vaccination', 'hospital', 'treatment'],
            'education': ['education', 'learning', 'school', 'student', 'university', 'skill', 'educational'],
            'environment': ['environment', 'climate', 'renewable', 'energy', 'clean', 'pollution', 'green'],
            'infrastructure': ['infrastructure', 'highway', 'railway', 'construction', 'development', 'project'],
            'social': ['social', 'festival', 'community', 'cultural', 'religious', 'harmony', 'celebration']
        }

    def classify_article(self, text):
        """Get topic predictions from multiple methods."""
        results = {}

        # Method 1: Keyword-based
        results['keywords'] = self.keyword_classification(text)

        # Method 2: TF-IDF similarity (simplified)
        results['tfidf'] = self.tfidf_classification(text)

        # Method 3: Simple rule-based
        results['rules'] = self.rule_based_classification(text)

        return results

    def keyword_classification(self, text):
        """Keyword-based classification."""
        text_lower = text.lower()

        for topic, keywords in self.topic_keywords.items():
            if any(word in text_lower for word in keywords):
                return topic
        return 'other'

    def tfidf_classification(self, text):
        """Simple TF-IDF based classification."""
        words = text.lower().split()

        if len([w for w in words if w in ['government', 'policy', 'political']]) > 0:
            return 'politics'
        elif len([w for w in words if w in ['sports', 'cricket', 'match']]) > 0:
            return 'sports'
        elif len([w for w in words if w in ['technology', 'software', 'digital', 'tech']]) > 0:
            return 'technology'
        elif len([w for w in words if w in ['business', 'market', 'economic']]) > 0:
            return 'business'
        else:
            return 'other'

    def rule_based_classification(self, text):
        """Rule-based classification."""
        text_lower = text.lower()

        # Count domain-specific terms
        scores = {
            'politics': sum(1 for term in ['minister', 'parliament', 'government', 'policy'] if term in text_lower),
            'sports': sum(1 for term in ['cricket', 'match', 'team', 'player'] if term in text_lower),
            'technology': sum(1 for term in ['software', 'digital', 'tech', 'startup'] if term in text_lower),
            'business': sum(1 for term in ['market', 'economic', 'business', 'financial'] if term in text_lower),
            'health': sum(1 for term in ['health', 'medical', 'healthcare', 'hospital'] if term in text_lower),
            'education': sum(1 for term in ['education', 'learning', 'school', 'university'] if term in text_lower)
        }

        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'other'

    def get_consensus(self, results):
        """Get consensus from multiple methods."""
        predictions = list(results.values())

        # Simple majority voting
        vote_counts = Counter(predictions)
        consensus = vote_counts.most_common(1)[0][0]
        confidence = vote_counts[consensus] / len(predictions)

        return consensus, confidence


def create_ensemble_topic_classifier():
    """Factory function to create an EnsembleTopicClassifier instance."""
    return EnsembleTopicClassifier()


# ------------------------------
# Visualization
# ------------------------------
def visualize_topics(df, output_path=None):
    """
    Visualize topic distributions and relationships.

    Args:
        df: DataFrame with topic classification results
        output_path: Optional path to save the visualization
    """
    plt.figure(figsize=(12, 8))

    # Subplot 1: Keyword-based topics
    if 'keyword_topic' in df.columns:
        plt.subplot(2, 2, 1)
        topic_counts = df['keyword_topic'].value_counts()
        plt.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.1f%%')
        plt.title('Keyword-Based Topic Distribution')

    # Subplot 2: LDA topics
    if 'lda_topic' in df.columns:
        plt.subplot(2, 2, 2)
        lda_topic_counts = df['lda_topic'].value_counts()
        plt.bar(range(len(lda_topic_counts)), lda_topic_counts.values)
        plt.title('LDA Topic Distribution')
        plt.xlabel('Topic ID')
        plt.ylabel('Count')

    # Subplot 3: Clustering results
    if 'cluster_topic' in df.columns:
        plt.subplot(2, 2, 3)
        cluster_counts = df['cluster_topic'].value_counts()
        plt.bar(range(len(cluster_counts)), cluster_counts.values)
        plt.title('K-means Cluster Distribution')
        plt.xlabel('Cluster ID')
        plt.ylabel('Count')

    # Subplot 4: Ensemble results
    if 'ensemble_topic' in df.columns:
        plt.subplot(2, 2, 4)
        ensemble_counts = df['ensemble_topic'].value_counts()
        plt.pie(ensemble_counts.values, labels=ensemble_counts.index, autopct='%1.1f%%')
        plt.title('Ensemble Topic Distribution')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def categorize_file(input_csv: str, output_dir: str, preferred_text_col: str | None = None,
                    use_all_methods: bool = False, visualize: bool = False) -> str:
    """
    Categorize articles in a CSV file using various classification methods.

    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save output files
        preferred_text_col: Preferred text column name
        use_all_methods: If True, apply all classification methods (LDA, clustering, ensemble)
        visualize: If True, generate and save visualizations

    Returns:
        out_path: Path to the output CSV file
    """
    # Load and preprocess data
    df = load_text_df(input_csv, preferred_text_col)
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Always apply keyword-based classification
    print("Applying keyword-based classification...")
    _ = keyword_based_topic_classification(df)

    if use_all_methods:
        print("Applying all classification methods...")

        # LDA Topic Modeling
        try:
            print("  - Running LDA topic modeling...")
            lda_model, lda_vectorizer, doc_topics, topic_words, _ = perform_lda_topic_modeling(
                df['processed_text'], n_topics=6
            )
            df['lda_topic'] = np.argmax(doc_topics, axis=1)
            df['lda_confidence'] = np.max(doc_topics, axis=1)
            print("    ✓ LDA completed")
        except Exception as e:
            print(f"    LDA failed: {e}")

        # TF-IDF + K-means Clustering
        try:
            print("  - Running TF-IDF clustering...")
            kmeans_model, tfidf_vec, cluster_labels, _ = tfidf_clustering_topics(
                df['processed_text'], n_clusters=5
            )
            df['cluster_topic'] = cluster_labels
            print("    ✓ Clustering completed")
        except Exception as e:
            print(f"    Clustering failed: {e}")

        # Ensemble Classification
        try:
            print("  - Running ensemble classification...")
            ensemble = create_ensemble_topic_classifier()
            ensemble_results = []
            ensemble_confidences = []

            for text in df['text']:
                results = ensemble.classify_article(text)
                consensus, confidence = ensemble.get_consensus(results)
                ensemble_results.append(consensus)
                ensemble_confidences.append(confidence)

            df['ensemble_topic'] = ensemble_results
            df['ensemble_confidence'] = ensemble_confidences
            print("    ✓ Ensemble classification completed")
        except Exception as e:
            print(f"    Ensemble classification failed: {e}")

        # Zero-shot Classification (optional, may be slow)
        # Uncomment to enable
        # try:
        #     print("  - Running zero-shot classification...")
        #     zs_results, _ = use_pretrained_topic_models(df['text'].tolist()[:10])
        #     if zs_results:
        #         df['zeroshot_topic'] = [r['top_label'] for r in zs_results]
        #         df['zeroshot_confidence'] = [r['top_score'] for r in zs_results]
        #         print(f"    Zero-shot classification completed")
        # except Exception as e:
        #     print(f"    Zero-shot classification failed: {e}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.basename(input_csv)
    name, ext = os.path.splitext(base)
    out_path = os.path.join(output_dir, f"{name}_cat.csv")
    df.to_csv(out_path, index=False)
    print(f"Results saved to: {out_path}")

    # Generate visualization if requested
    if visualize:
        viz_path = os.path.join(output_dir, f"{name}_visualization.png")
        try:
            visualize_topics(df, output_path=viz_path)
        except Exception as e:
            print(f"Visualization failed: {e}")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Batch categorize news article CSVs with multiple topic classification methods."
    )
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                        help="Directory containing input CSV files")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save categorized output files")
    parser.add_argument("--text_column", type=str, default=None,
                        help="Optional text column name to prefer")
    parser.add_argument("--all_methods", action="store_true",
                        help="Use all classification methods (LDA, clustering, ensemble)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization plots")
    args = parser.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found in {args.input_dir}")
        return

    print(f"Found {len(csv_paths)} CSV file(s) to categorize.")
    print(f"Using all methods: {args.all_methods}")
    print(f"Visualization: {args.visualize}")
    print()

    written = []
    for i, path in enumerate(csv_paths, 1):
        try:
            print(f"[{i}/{len(csv_paths)}] Processing: {os.path.basename(path)}")
            out = categorize_file(
                path,
                args.output_dir,
                args.text_column,
                use_all_methods=args.all_methods,
                visualize=args.visualize
            )
            written.append(out)
            print(f"[{i}/{len(csv_paths)}] ✓ Completed\n")
        except Exception as e:
            print(f"[{i}/{len(csv_paths)}] ✗ Failed {path}: {e}\n")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully processed: {len(written)}/{len(csv_paths)} files")
    for p in written:
        print(f"  ✓ {p}")


if __name__ == "__main__":
    main()
