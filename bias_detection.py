"""
Comprehensive Bias Detection System for News Articles
Combines topic classification with multi-dimensional bias analysis

ENSEMBLE APPROACH FOR IMPROVED BIAS DETECTION:
==============================================

This module uses a sophisticated ensemble approach combining three complementary methods:

1. **Keyword Weighted Score (30%)**
   - Simple and interpretable keyword matching
   - Counts occurrences of curated bias-specific keywords
   - Normalized by text length and keyword list size
   - Fast and efficient for quick baseline assessment

2. **TF-IDF Weighted Score (30%)**
   - Uses Term Frequency-Inverse Document Frequency
   - Weighs keywords by their importance in the document
   - Calculates cosine similarity between article and keyword groups
   - Efficient for batch processing and handles term importance

3. **BERT Embedding Similarity (40%)**
   - Uses sentence-transformers (all-MiniLM-L6-v2 model)
   - Captures semantic context and meaning beyond exact keyword matches
   - Computes cosine similarity between article and ideological embeddings
   - Most powerful but computationally intensive

ADVANTAGES:
-----------
- **Robustness**: Combines multiple signals reduces false positives/negatives
- **Context-Aware**: BERT embeddings capture semantic meaning
- **Interpretability**: Keyword scores provide transparency
- **Efficiency**: TF-IDF balances speed with sophistication
- **Adaptability**: Ensemble weights can be adjusted per use case

BIAS DETECTION DIMENSIONS:
--------------------------
- Political (left/right/centrist)
- Gender (male/female bias with stereotype detection)
- Religious (multi-faith with stereotype detection)
- Caste (upper/lower caste focus)
- Regional (geographic bias detection)
- Socioeconomic (wealth/poverty focus)

Each dimension uses the ensemble approach to produce more accurate bias scores.
"""

from catogorise_the_article import (
    EnsembleTopicClassifier,
    create_ensemble_topic_classifier,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple
import csv
import warnings
warnings.filterwarnings('ignore')

# TF-IDF and ML imports

# BERT embeddings
try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")

# Import topic classification from existing script


class BiasKeywords:
    """Comprehensive bias keyword dictionaries based on IndiBias framework"""

    def __init__(self):
        # Gender Keywords
        self.gender_keywords = {
            "male_terms": [
                "man",
                "men",
                "male",
                "boy",
                "he",
                "him",
                "his",
                "himself",
                "husband",
                "father",
                "son",
                "brother",
                "gentleman",
            ],
            "female_terms": [
                "woman",
                "women",
                "female",
                "girl",
                "she",
                "her",
                "hers",
                "herself",
                "wife",
                "mother",
                "daughter",
                "sister",
                "lady",
            ],
            "stereotypical_female": [
                "nurturing",
                "submissive",
                "sensitive",
                "dependent",
                "emotional",
                "weak",
                "homemaker",
                "caretaker",
                "family-oriented",
            ],
            "stereotypical_male": [
                "tech-savvy",
                "dominant",
                "courageous",
                "workaholic",
                "strong",
                "breadwinner",
                "career-oriented",
                "aggressive",
            ],
        }

        # Religious Keywords
        self.religion_keywords = {
            "hindu": [
                "hindu",
                "hinduism",
                "brahmin",
                "kshatriya",
                "vaishya",
                "shudra",
                "temple",
                "puja",
                "vedic",
                "sanskrit",
                "mandir",
                "pandit",
            ],
            "muslim": [
                "muslim",
                "islam",
                "mullah",
                "maulvi",
                "mosque",
                "namaz",
                "allah",
                "islamic",
                "masjid",
                "quran",
                "imam",
            ],
            "sikh": [
                "sikh",
                "sikhism",
                "gurdwara",
                "khalsa",
                "singh",
                "kaur",
                "guru",
                "gurudwara",
            ],
            "christian": [
                "christian",
                "christianity",
                "church",
                "jesus",
                "bible",
                "priest",
                "pastor",
            ],
            "jain": ["jain", "jainism"],
            "buddhist": ["buddhist", "buddhism", "buddha"],
            "stereotypes": {
                "violent": [
                    "terrorist",
                    "violent",
                    "aggressive",
                    "fundamentalist",
                    "extremist",
                ],
                "superstitious": ["superstitious", "backward", "orthodox"],
                "conservative": ["conservative", "traditional", "rigid"],
            },
        }

        # Caste Keywords
        self.caste_keywords = {
            "upper_caste": [
                "brahmin",
                "kshatriya",
                "thakur",
                "rajput",
                "arya",
                "jaat",
                "baniya",
                "kayastha",
                "upper caste",
                "high caste",
                "forward caste",
            ],
            "lower_caste": [
                "dalit",
                "shudra",
                "sc",
                "st",
                "scheduled caste",
                "scheduled tribe",
                "obc",
                "other backward class",
                "bhangi",
                "chamaar",
                "valmiki",
                "harijan",
                "backward caste",
                "lower caste",
            ],
            "stereotypes": {
                "elitist": ["elitist", "privileged", "entitled", "superior"],
                "oppressed": [
                    "oppressed",
                    "discriminated",
                    "marginalized",
                    "subservient",
                ],
            },
        }

        # Regional Keywords
        self.region_keywords = {
            "northeast": [
                "northeast",
                "assam",
                "nagaland",
                "manipur",
                "tripura",
                "meghalaya",
                "mizoram",
                "arunachal",
            ],
            "north": [
                "delhi",
                "punjab",
                "haryana",
                "himachal",
                "jammu",
                "kashmir",
                "uttarakhand",
                "uttar pradesh",
                "up",
                "bihar",
            ],
            "south": [
                "tamil nadu",
                "kerala",
                "karnataka",
                "andhra pradesh",
                "telangana",
                "chennai",
                "bangalore",
                "hyderabad",
            ],
            "west": ["maharashtra", "gujarat", "rajasthan", "goa", "mumbai"],
            "east": ["west bengal", "odisha", "jharkhand", "kolkata"],
            "central": ["madhya pradesh", "chhattisgarh"],
            "stereotypes": {
                "racist": ["chinky", "chinese-looking", "mongoloid"],
                "backward": ["backward", "undeveloped", "poor", "illiterate"],
            },
        }

        # Socioeconomic Keywords
        self.socioeconomic_keywords = {
            "wealthy": [
                "rich",
                "wealthy",
                "affluent",
                "elite",
                "privileged",
                "upper class",
                "high society",
                "millionaire",
                "billionaire",
            ],
            "poor": [
                "poor",
                "poverty",
                "broke",
                "underprivileged",
                "lower class",
                "slum",
                "backward",
                "disadvantaged",
                "impoverished",
            ],
            "stereotypes": {
                "elite_disconnected": ["out of touch", "privileged", "entitled"],
                "poor_negative": ["criminal", "lazy", "uneducated", "burden"],
            },
        }

        # Political Leaning Keywords (from political_leaning file)
        self.political_keywords = {
            "left_leaning": [
                "equality",
                "justice",
                "progressive",
                "welfare",
                "reform",
                "inclusive",
                "diversity",
                "socialism",
                "redistribution",
                "solidarity",
                "equity",
                "activism",
                "feminism",
                "environment",
                "sustainability",
                "labor rights",
                "social justice",
                "human rights",
                "universal healthcare",
                "climate action",
                "workers rights",
                "anti-discrimination",

            ],
            "right_leaning": [
                "nationalism",
                "patriotism",
                "security",
                "sovereignty",
                "tradition",
                "culture",
                "heritage",
                "values",
                "order",
                "capitalism",
                "market economy",
                "privatization",
                "self-reliance",
                "individualism",
                "law and order",
                "family values",
                "hindu nationalism",
                "national pride",
                "strong defense",
                "immigration control",
                "free market",
            ],
            "centrist": [
                "bipartisan",
                "compromise",
                "pragmatic",
                "middle ground",
                "consensus",
                "balanced policy",
                "moderation",
                "collaboration",
                "harmony",
                "independent",
                "neutral",
                "unity",
                "cooperation",
            ],
        }


class BiasDetector:
    """Multi-dimensional bias detector for news articles with ensemble approach"""

    def __init__(self):
        self.keywords = BiasKeywords()
        self.topic_classifier = create_ensemble_topic_classifier()

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )

        # Initialize BERT model for embeddings
        if BERT_AVAILABLE:
            try:
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✓ BERT model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.bert_model = None
        else:
            self.bert_model = None

        # Ensemble weights (keyword, tfidf, embedding)
        self.ensemble_weights = {
            'keyword': 0.3,
            'tfidf': 0.3,
            'embedding': 0.4
        }

    def calculate_keyword_score(self, text: str, keyword_list: List[str]) -> float:
        """Calculate normalized keyword score for given text"""
        if not text or not keyword_list:
            return 0.0

        text_lower = text.lower()
        matches = sum(1 for keyword in keyword_list if keyword in text_lower)

        # Normalize by text length (per 1000 words) and keyword list size
        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        normalized_score = (matches / len(keyword_list)) * (1000 / max(word_count, 1))
        return min(normalized_score, 1.0)  # Cap at 1.0

    def calculate_tfidf_score(self, text: str, keyword_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate TF-IDF weighted scores for keyword groups"""
        if not text or not keyword_groups:
            return {k: 0.0 for k in keyword_groups.keys()}

        try:
            # Prepare documents: original text + keyword group texts
            documents = [text]
            group_names = []
            for group_name, keywords in keyword_groups.items():
                documents.append(' '.join(keywords))
                group_names.append(group_name)

            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

            # Calculate cosine similarity between text and each keyword group
            text_vector = tfidf_matrix[0:1]
            scores = {}

            for i, group_name in enumerate(group_names):
                group_vector = tfidf_matrix[i + 1:i + 2]
                similarity = cosine_similarity(text_vector, group_vector)[0][0]
                scores[group_name] = max(0.0, similarity)

            # Normalize scores to sum to 1
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

            return scores

        except Exception as e:
            # Fallback to zero scores
            return {k: 0.0 for k in keyword_groups.keys()}

    def calculate_embedding_similarity(self, text: str, keyword_groups: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate BERT embedding similarity scores for keyword groups"""
        if not text or not keyword_groups or self.bert_model is None:
            return {k: 0.0 for k in keyword_groups.keys()}

        try:
            # Get embedding for the text
            text_embedding = self.bert_model.encode([text])[0]

            # Get embeddings for each keyword group
            scores = {}
            for group_name, keywords in keyword_groups.items():
                # Create a representative text from keywords
                group_text = ' '.join(keywords)
                group_embedding = self.bert_model.encode([group_text])[0]

                # Calculate cosine similarity
                similarity = np.dot(text_embedding, group_embedding) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(group_embedding)
                )
                scores[group_name] = max(0.0, similarity)

            # Normalize scores
            total = sum(scores.values())
            if total > 0:
                scores = {k: v / total for k, v in scores.items()}

            return scores

        except Exception as e:
            return {k: 0.0 for k in keyword_groups.keys()}

    def ensemble_bias_score(self, text: str, keyword_groups: Dict[str, List[str]]) -> Tuple[Dict[str, float], str, float]:
        """Calculate ensemble bias score combining keyword, TF-IDF, and embedding approaches"""
        if not text or not keyword_groups:
            return {k: 0.0 for k in keyword_groups.keys()}, 'neutral', 0.0

        # 1. Keyword-based scores
        keyword_scores = {}
        for group_name, keywords in keyword_groups.items():
            keyword_scores[group_name] = self.calculate_keyword_score(text, keywords)

        # 2. TF-IDF scores
        tfidf_scores = self.calculate_tfidf_score(text, keyword_groups)

        # 3. Embedding similarity scores
        embedding_scores = self.calculate_embedding_similarity(text, keyword_groups)

        # Combine scores with weights
        ensemble_scores = {}
        for group_name in keyword_groups.keys():
            combined = (
                self.ensemble_weights['keyword'] * keyword_scores.get(group_name, 0.0)
                + self.ensemble_weights['tfidf'] * tfidf_scores.get(group_name, 0.0)
                + self.ensemble_weights['embedding'] * embedding_scores.get(group_name, 0.0)
            )
            ensemble_scores[group_name] = combined

        # Normalize ensemble scores
        total = sum(ensemble_scores.values())
        if total > 0:
            ensemble_scores = {k: v / total for k, v in ensemble_scores.items()}

        # Determine dominant group and confidence
        if not ensemble_scores or max(ensemble_scores.values()) == 0:
            return ensemble_scores, 'neutral', 0.0

        dominant_group = max(ensemble_scores, key=ensemble_scores.get)
        dominant_score = ensemble_scores[dominant_group]

        # Calculate confidence (difference from average of others)
        other_scores = [v for k, v in ensemble_scores.items() if k != dominant_group]
        avg_other = sum(other_scores) / len(other_scores) if other_scores else 0
        confidence = dominant_score - avg_other

        return ensemble_scores, dominant_group, confidence

    def detect_gender_bias(self, text: str) -> Tuple[float, str]:
        """
        Detect gender bias in text using ensemble approach
        Returns: (bias_score, bias_type)
        """
        # Prepare keyword groups for ensemble
        keyword_groups = {
            'male': self.keywords.gender_keywords["male_terms"]
            + self.keywords.gender_keywords["stereotypical_male"],
            'female': self.keywords.gender_keywords["female_terms"]
            + self.keywords.gender_keywords["stereotypical_female"]
        }

        # Get ensemble scores
        ensemble_scores, dominant_group, confidence = self.ensemble_bias_score(text, keyword_groups)

        # Check for stereotype presence
        text_lower = text.lower()
        stereotype_female = sum(1 for term in self.keywords.gender_keywords["stereotypical_female"]
                                if term in text_lower)
        stereotype_male = sum(1 for term in self.keywords.gender_keywords["stereotypical_male"]
                              if term in text_lower)

        # Calculate bias score incorporating stereotypes
        bias_score = confidence
        if stereotype_female > 0 or stereotype_male > 0:
            stereotype_boost = min((stereotype_female + stereotype_male) * 0.05, 0.3)
            bias_score = min(bias_score + stereotype_boost, 1.0)

        # Determine bias type
        if bias_score < 0.25 or max(ensemble_scores.values()) < 0.1:
            return 0.0, "neutral"
        elif confidence < 0.3:
            bias_type = "low_bias"
        elif dominant_group == 'male' or stereotype_male > stereotype_female:
            bias_type = "male_bias"
        elif dominant_group == 'female' or stereotype_female > stereotype_male:
            bias_type = "female_bias"
        else:
            bias_type = "moderate_bias"

        return round(bias_score, 3), bias_type

    def detect_religious_bias(self, text: str) -> Tuple[float, str]:
        """Detect religious bias using ensemble approach"""
        # Prepare keyword groups for religions
        keyword_groups = {}
        for religion, keywords in self.keywords.religion_keywords.items():
            if religion != "stereotypes":
                keyword_groups[religion] = keywords

        if not keyword_groups:
            return 0.0, "neutral"

        # Get ensemble scores
        ensemble_scores, dominant_group, confidence = self.ensemble_bias_score(text, keyword_groups)

        # Check for negative stereotypes
        text_lower = text.lower()
        violence_count = sum(1 for term in self.keywords.religion_keywords["stereotypes"]["violent"]
                             if term in text_lower)
        negative_count = sum(1 for term in self.keywords.religion_keywords["stereotypes"]["superstitious"]
                             if term in text_lower)

        # Calculate bias score with stereotype boost
        bias_score = confidence
        if violence_count > 0 or negative_count > 0:
            stereotype_boost = min((violence_count + negative_count) * 0.08, 0.4)
            bias_score = min(bias_score + stereotype_boost, 1.0)

        # Determine bias type
        if bias_score < 0.25 or max(ensemble_scores.values()) < 0.1:
            return 0.0, "neutral"
        elif bias_score < 0.3:
            bias_type = "low_bias"
        elif violence_count > 2:
            bias_type = f"{dominant_group}_negative_stereotype"
        else:
            bias_type = f"{dominant_group}_focus"

        return round(bias_score, 3), bias_type

    def detect_caste_bias(self, text: str) -> Tuple[float, str]:
        """Detect caste bias using ensemble approach"""
        # Prepare keyword groups
        keyword_groups = {
            'upper_caste': self.keywords.caste_keywords['upper_caste']
            + self.keywords.caste_keywords['stereotypes']['elitist'],
            'lower_caste': self.keywords.caste_keywords['lower_caste']
            + self.keywords.caste_keywords['stereotypes']['oppressed']
        }

        # Get ensemble scores
        ensemble_scores, dominant_group, confidence = self.ensemble_bias_score(text, keyword_groups)

        # Check for stereotype presence
        text_lower = text.lower()
        elitist_count = sum(1 for term in self.keywords.caste_keywords['stereotypes']['elitist']
                            if term in text_lower)
        oppressed_count = sum(1 for term in self.keywords.caste_keywords['stereotypes']['oppressed']
                              if term in text_lower)

        # Calculate bias score with stereotype consideration
        bias_score = confidence
        if elitist_count > 0 or oppressed_count > 0:
            stereotype_boost = min((elitist_count + oppressed_count) * 0.06, 0.3)
            bias_score = min(bias_score + stereotype_boost, 1.0)

        # Determine bias type
        if bias_score < 0.25 or max(ensemble_scores.values()) < 0.1:
            return 0.0, "neutral"
        elif bias_score < 0.3:
            bias_type = "low_bias"
        elif dominant_group == 'upper_caste':
            bias_type = "upper_caste_focus"
        elif dominant_group == 'lower_caste':
            bias_type = "lower_caste_focus"
        else:
            bias_type = "balanced"

        return round(bias_score, 3), bias_type

    def detect_region_bias(self, text: str) -> Tuple[float, str]:
        """Detect regional bias using ensemble approach"""
        # Prepare keyword groups for regions
        keyword_groups = {}
        for region, keywords in self.keywords.region_keywords.items():
            if region != "stereotypes":
                keyword_groups[region] = keywords

        if not keyword_groups:
            return 0.0, "neutral"

        # Get ensemble scores
        ensemble_scores, dominant_group, confidence = self.ensemble_bias_score(text, keyword_groups)

        # Check for negative stereotypes
        text_lower = text.lower()
        racist_count = sum(1 for term in self.keywords.region_keywords["stereotypes"]["racist"]
                           if term in text_lower)
        backward_count = sum(1 for term in self.keywords.region_keywords["stereotypes"]["backward"]
                             if term in text_lower)

        # Calculate bias score with stereotype boost
        bias_score = confidence
        if racist_count > 0 or backward_count > 0:
            stereotype_boost = min((racist_count + backward_count) * 0.1, 0.4)
            bias_score = min(bias_score + stereotype_boost, 1.0)

        # Determine bias type
        if bias_score < 0.25 or max(ensemble_scores.values()) < 0.1:
            return 0.0, "neutral"
        elif bias_score < 0.3:
            bias_type = "low_bias"
        elif racist_count > 1 or backward_count > 2:
            bias_type = f"{dominant_group}_negative_stereotype"
        else:
            bias_type = f"{dominant_group}_focus"

        return round(bias_score, 3), bias_type

    def detect_socioeconomic_bias(self, text: str) -> Tuple[float, str]:
        """Detect socioeconomic bias using ensemble approach"""
        # Prepare keyword groups
        keyword_groups = {
            'wealthy': self.keywords.socioeconomic_keywords['wealthy']
            + self.keywords.socioeconomic_keywords['stereotypes']['elite_disconnected'],
            'poor': self.keywords.socioeconomic_keywords['poor']
            + self.keywords.socioeconomic_keywords['stereotypes']['poor_negative']
        }

        # Get ensemble scores
        ensemble_scores, dominant_group, confidence = self.ensemble_bias_score(text, keyword_groups)

        # Check for stereotype presence
        text_lower = text.lower()
        elite_count = sum(1 for term in self.keywords.socioeconomic_keywords['stereotypes']['elite_disconnected']
                          if term in text_lower)
        poor_negative_count = sum(1 for term in self.keywords.socioeconomic_keywords['stereotypes']['poor_negative']
                                  if term in text_lower)

        # Calculate bias score with stereotype consideration
        bias_score = confidence
        if elite_count > 0 or poor_negative_count > 0:
            stereotype_boost = min((elite_count + poor_negative_count) * 0.07, 0.35)
            bias_score = min(bias_score + stereotype_boost, 1.0)

        # Determine bias type
        if bias_score < 0.25 or max(ensemble_scores.values()) < 0.1:
            return 0.0, "neutral"
        elif bias_score < 0.3:
            bias_type = "low_bias"
        elif dominant_group == 'wealthy':
            bias_type = "wealthy_focus"
        elif dominant_group == 'poor':
            bias_type = "poverty_focus"
        else:
            bias_type = "balanced"

        return round(bias_score, 3), bias_type

    def detect_political_bias(self, text: str) -> Tuple[float, str]:
        """
        Detect political leaning bias using ensemble approach
        Combines keyword counting, TF-IDF, and BERT embeddings
        """
        # Prepare keyword groups
        keyword_groups = {
            'left_leaning': self.keywords.political_keywords['left_leaning'],
            'right_leaning': self.keywords.political_keywords['right_leaning'],
            'centrist': self.keywords.political_keywords['centrist']
        }

        # Get ensemble scores
        ensemble_scores, dominant_group, confidence = self.ensemble_bias_score(text, keyword_groups)

        # Map to bias types
        if confidence < 0.2 or max(ensemble_scores.values()) < 0.1:
            return 0.0, "neutral"

        bias_type_map = {
            'left_leaning': 'left_leaning',
            'right_leaning': 'right_leaning',
            'centrist': 'centrist'
        }

        bias_type = bias_type_map.get(dominant_group, 'neutral')
        bias_score = min(confidence, 1.0)

        return round(bias_score, 3), bias_type

    def calculate_overall_bias_score(self, bias_scores: Dict[str, float]) -> float:
        """Calculate overall bias score from individual bias dimensions"""
        # Weight different bias types
        weights = {
            "political": 0.20,
            "gender": 0.15,
            "religious": 0.20,
            "caste": 0.15,
            "region": 0.15,
            "socioeconomic": 0.15,
        }

        weighted_score = sum(
            bias_scores.get(bias_type, 0.0) * weight
            for bias_type, weight in weights.items()
        )

        return round(weighted_score, 3)

    def analyze_article(self, article_text: str) -> Dict:
        """Perform comprehensive bias analysis on article"""
        if not article_text or not isinstance(article_text, str):
            return {
                "political_bias": 0.0,
                "political_type": "neutral",
                "gender_bias": 0.0,
                "gender_type": "neutral",
                "religious_bias": 0.0,
                "religious_type": "neutral",
                "caste_bias": 0.0,
                "caste_type": "neutral",
                "region_bias": 0.0,
                "region_type": "neutral",
                "socioeconomic_bias": 0.0,
                "socioeconomic_type": "neutral",
                "overall_bias_score": 0.0,
            }

        # Detect all bias dimensions
        political_bias, political_type = self.detect_political_bias(article_text)
        gender_bias, gender_type = self.detect_gender_bias(article_text)
        religious_bias, religious_type = self.detect_religious_bias(article_text)
        caste_bias, caste_type = self.detect_caste_bias(article_text)
        region_bias, region_type = self.detect_region_bias(article_text)
        socioeconomic_bias, socioeconomic_type = self.detect_socioeconomic_bias(
            article_text
        )

        # Calculate overall bias score
        bias_scores = {
            "political": political_bias,
            "gender": gender_bias,
            "religious": religious_bias,
            "caste": caste_bias,
            "region": region_bias,
            "socioeconomic": socioeconomic_bias,
        }

        overall_score = self.calculate_overall_bias_score(bias_scores)

        return {
            "political_bias": f"{political_bias:.3f} ({political_type})",
            "gender_bias": f"{gender_bias:.3f} ({gender_type})",
            "religious_bias": f"{religious_bias:.3f} ({religious_type})",
            "caste_bias": f"{caste_bias:.3f} ({caste_type})",
            "region_bias": f"{region_bias:.3f} ({region_type})",
            "socioeconomic_bias": f"{socioeconomic_bias:.3f} ({socioeconomic_type})",
            "overall_bias_score": overall_score,
        }


def process_articles(input_csv: str, output_csv: str):
    """
    Process articles from input CSV and generate comprehensive bias analysis

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
    """
    print("Loading data...")
    df = pd.read_csv(input_csv, escapechar="\\", quotechar='"', encoding="utf-8")

    print(f"Loaded {len(df)} articles")
    print(f"Columns: {list(df.columns)}")

    # Initialize detectors
    bias_detector = BiasDetector()
    topic_classifier = create_ensemble_topic_classifier()

    # Prepare results
    results = []

    print("\nProcessing articles...")
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing article {idx + 1}/{len(df)}...")

        try:
            article_text = str(row.get("article_text", ""))

            # Get topic classification
            topic_results = topic_classifier.classify_article(article_text)
            topic, topic_confidence = topic_classifier.get_consensus(topic_results)

            # Get bias analysis
            bias_analysis = bias_detector.analyze_article(article_text)

            # Prepare output row
            result = {
                "url": row.get("url", ""),
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "published_date": row.get("published_date", ""),
                "article_text": article_text,
                "word_count": row.get("word_count", len(article_text.split())),
                "media_name": row.get("media_name", "Indian Express"),
                "topic": topic,
                "topic_confidence": f"{topic_confidence:.3f}",
                "political_bias": bias_analysis["political_bias"],
                "gender_bias": bias_analysis["gender_bias"],
                "religious_bias": bias_analysis["religious_bias"],
                "caste_bias": bias_analysis["caste_bias"],
                "region_bias": bias_analysis["region_bias"],
                "socioeconomic_bias": bias_analysis["socioeconomic_bias"],
                "overall_bias_score": bias_analysis["overall_bias_score"],
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing article {idx}: {e}")
            # Add error row
            results.append(
                {
                    "url": row.get("url", ""),
                    "title": row.get("title", ""),
                    "author": row.get("author", ""),
                    "published_date": row.get("published_date", ""),
                    "article_text": "",
                    "word_count": 0,
                    "media_name": row.get("media_name", "Indian Express"),
                    "topic": "error",
                    "topic_confidence": "0.000",
                    "political_bias": "0.000 (neutral)",
                    "gender_bias": "0.000 (neutral)",
                    "religious_bias": "0.000 (neutral)",
                    "caste_bias": "0.000 (neutral)",
                    "region_bias": "0.000 (neutral)",
                    "socioeconomic_bias": "0.000 (neutral)",
                    "overall_bias_score": 0.000,
                }
            )

    # Create output DataFrame
    output_df = pd.DataFrame(results)

    # Save to CSV
    print(f"\nSaving results to {output_csv}...")
    output_df.to_csv(
        output_csv,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar="\\",
        encoding="utf-8",
    )

    print(f"✓ Successfully processed {len(results)} articles")
    print(f"✓ Output saved to: {output_csv}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("BIAS ANALYSIS SUMMARY")
    print("=" * 60)

    # Topic distribution
    print("\nTopic Distribution:")
    topic_counts = output_df["topic"].value_counts()
    for topic, count in topic_counts.head(10).items():
        print(f"  {topic}: {count} articles ({count / len(output_df) * 100:.1f}%)")

    # Bias statistics
    print("\nAverage Bias Scores:")
    print(f"  Overall Bias: {output_df['overall_bias_score'].mean():.3f}")

    return output_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Bias Detection for News Articles"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./articles/raw/indian_express_article_content_2024_progress_100.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./bias_analysis_results/comprehensive_bias_analysis_2024_progress_100.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    import os

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Process articles
    process_articles(args.input, args.output)