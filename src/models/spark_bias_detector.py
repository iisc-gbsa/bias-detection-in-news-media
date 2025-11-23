"""
Spark ML-based Bias Detection System
Scalable bias detection using PySpark for distributed processing
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    udf,
    col,
    lower,
    size,
    split,
    lit,
    struct,
    array,
    explode,
    when,
    regexp_replace,
    length,
    coalesce,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType,
    DoubleType,
    IntegerType,
    MapType,
    ArrayType,
)
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    CountVectorizer,
    VectorAssembler,
)
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

from src.models.bias_detection import BiasKeywords
from config.config import bias_config


class SparkBiasDetector:
    """
    Distributed bias detection using Spark ML
    Optimized for scalable processing of large article datasets
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.keywords = BiasKeywords()
        self.config = bias_config

        # Broadcast keyword dictionaries for efficient distributed access
        self._broadcast_keywords()

        # Initialize ML pipeline components
        self._init_ml_pipeline()

    def _broadcast_keywords(self):
        """Broadcast keyword dictionaries to all executors"""
        self.gender_keywords_bc = self.spark.sparkContext.broadcast(
            self.keywords.gender_keywords
        )
        self.religion_keywords_bc = self.spark.sparkContext.broadcast(
            self.keywords.religion_keywords
        )
        self.caste_keywords_bc = self.spark.sparkContext.broadcast(
            self.keywords.caste_keywords
        )
        self.region_keywords_bc = self.spark.sparkContext.broadcast(
            self.keywords.region_keywords
        )
        self.socioeconomic_keywords_bc = self.spark.sparkContext.broadcast(
            self.keywords.socioeconomic_keywords
        )
        self.political_keywords_bc = self.spark.sparkContext.broadcast(
            self.keywords.political_keywords
        )

    def _init_ml_pipeline(self):
        """Initialize Spark ML pipeline for text processing"""
        # Tokenization
        self.tokenizer = Tokenizer(inputCol="article_text", outputCol="words")

        # Remove stop words
        self.stopwords_remover = StopWordsRemover(
            inputCol="words", outputCol="filtered_words"
        )

        # TF-IDF vectorization
        self.count_vectorizer = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=self.config.tfidf_max_features,
        )

        self.idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    def _create_keyword_score_udf(self, keyword_dict_name: str):
        """Create UDF for keyword-based scoring"""

        def calculate_keyword_scores(text: str) -> Dict[str, float]:
            """Calculate keyword scores for all groups"""
            if not text:
                return {}

            text_lower = text.lower()
            word_count = len(text.split())

            # Get keyword dict based on type
            if keyword_dict_name == "gender":
                keyword_dict = self.gender_keywords_bc.value
            elif keyword_dict_name == "religion":
                keyword_dict = self.religion_keywords_bc.value
            elif keyword_dict_name == "caste":
                keyword_dict = self.caste_keywords_bc.value
            elif keyword_dict_name == "region":
                keyword_dict = self.region_keywords_bc.value
            elif keyword_dict_name == "socioeconomic":
                keyword_dict = self.socioeconomic_keywords_bc.value
            elif keyword_dict_name == "political":
                keyword_dict = self.political_keywords_bc.value
            else:
                return {}

            scores = {}
            for group_name, keywords in keyword_dict.items():
                if isinstance(keywords, dict):  # Skip nested dicts like stereotypes
                    continue

                matches = sum(1 for kw in keywords if kw in text_lower)
                if word_count > 0 and len(keywords) > 0:
                    normalized = (matches / len(keywords)) * (1000 / max(word_count, 1))
                    scores[group_name] = min(normalized, 1.0)
                else:
                    scores[group_name] = 0.0

            return scores

        return udf(calculate_keyword_scores, MapType(StringType(), FloatType()))

    def _detect_gender_bias_udf(self):
        """UDF for gender bias detection"""
        gender_kw = self.gender_keywords_bc.value

        def detect_gender(text: str) -> Tuple[float, str]:
            if not text:
                return (0.0, "neutral")

            text_lower = text.lower()

            # Count male vs female terms
            male_count = sum(1 for t in gender_kw["male_terms"] if t in text_lower)
            female_count = sum(1 for t in gender_kw["female_terms"] if t in text_lower)

            # Count stereotypes
            male_stereo = sum(
                1 for t in gender_kw["stereotypical_male"] if t in text_lower
            )
            female_stereo = sum(
                1 for t in gender_kw["stereotypical_female"] if t in text_lower
            )

            total_mentions = male_count + female_count + male_stereo + female_stereo

            if total_mentions == 0:
                return (0.0, "neutral")

            # Calculate bias
            male_score = (male_count + male_stereo * 1.5) / total_mentions
            female_score = (female_count + female_stereo * 1.5) / total_mentions

            bias_diff = abs(male_score - female_score)

            if bias_diff < 0.25:
                return (round(bias_diff, 3), "neutral")
            elif male_score > female_score:
                return (round(bias_diff, 3), "male_bias")
            else:
                return (round(bias_diff, 3), "female_bias")

        return udf(
            detect_gender,
            StructType(
                [StructField("score", FloatType()), StructField("type", StringType())]
            ),
        )

    def _detect_religious_bias_udf(self):
        """UDF for religious bias detection"""
        religion_kw = self.religion_keywords_bc.value

        def detect_religious(text: str) -> Tuple[float, str]:
            if not text:
                return (0.0, "neutral")

            text_lower = text.lower()

            # Count mentions by religion
            religion_counts = {}
            for religion, keywords in religion_kw.items():
                if religion != "stereotypes":
                    count = sum(1 for kw in keywords if kw in text_lower)
                    religion_counts[religion] = count

            # Count negative stereotypes
            violent_count = sum(
                1 for t in religion_kw["stereotypes"]["violent"] if t in text_lower
            )
            negative_count = sum(
                1
                for t in religion_kw["stereotypes"]["superstitious"]
                if t in text_lower
            )

            total = sum(religion_counts.values())
            if total == 0:
                return (0.0, "neutral")

            # Find dominant religion
            dominant = max(religion_counts, key=religion_counts.get)
            dominant_ratio = religion_counts[dominant] / total

            # Boost score if negative stereotypes present
            bias_score = dominant_ratio * 0.5
            if violent_count > 0 or negative_count > 0:
                bias_score = min(
                    bias_score + (violent_count + negative_count) * 0.1, 1.0
                )

            if bias_score < 0.25:
                return (round(bias_score, 3), "neutral")
            elif violent_count > 2:
                return (round(bias_score, 3), f"{dominant}_negative_stereotype")
            else:
                return (round(bias_score, 3), f"{dominant}_focus")

        return udf(
            detect_religious,
            StructType(
                [StructField("score", FloatType()), StructField("type", StringType())]
            ),
        )

    def _detect_caste_bias_udf(self):
        """UDF for caste bias detection"""
        caste_kw = self.caste_keywords_bc.value

        def detect_caste(text: str) -> Tuple[float, str]:
            if not text:
                return (0.0, "neutral")

            text_lower = text.lower()

            upper_count = sum(1 for t in caste_kw["upper_caste"] if t in text_lower)
            lower_count = sum(1 for t in caste_kw["lower_caste"] if t in text_lower)

            total = upper_count + lower_count
            if total == 0:
                return (0.0, "neutral")

            # Check stereotypes
            elitist = sum(
                1 for t in caste_kw["stereotypes"]["elitist"] if t in text_lower
            )
            oppressed = sum(
                1 for t in caste_kw["stereotypes"]["oppressed"] if t in text_lower
            )

            bias_score = abs(upper_count - lower_count) / total
            if elitist > 0 or oppressed > 0:
                bias_score = min(bias_score + (elitist + oppressed) * 0.08, 1.0)

            if bias_score < 0.25:
                return (round(bias_score, 3), "neutral")
            elif upper_count > lower_count:
                return (round(bias_score, 3), "upper_caste_focus")
            else:
                return (round(bias_score, 3), "lower_caste_focus")

        return udf(
            detect_caste,
            StructType(
                [StructField("score", FloatType()), StructField("type", StringType())]
            ),
        )

    def _detect_region_bias_udf(self):
        """UDF for regional bias detection"""
        region_kw = self.region_keywords_bc.value

        def detect_region(text: str) -> Tuple[float, str]:
            if not text:
                return (0.0, "neutral")

            text_lower = text.lower()

            # Count by region
            region_counts = {}
            for region, keywords in region_kw.items():
                if region != "stereotypes":
                    count = sum(1 for kw in keywords if kw in text_lower)
                    region_counts[region] = count

            total = sum(region_counts.values())
            if total == 0:
                return (0.0, "neutral")

            # Check negative stereotypes
            racist = sum(
                1 for t in region_kw["stereotypes"]["racist"] if t in text_lower
            )
            backward = sum(
                1 for t in region_kw["stereotypes"]["backward"] if t in text_lower
            )

            dominant = max(region_counts, key=region_counts.get)
            bias_score = region_counts[dominant] / total * 0.5

            if racist > 0 or backward > 0:
                bias_score = min(bias_score + (racist + backward) * 0.12, 1.0)

            if bias_score < 0.25:
                return (round(bias_score, 3), "neutral")
            elif racist > 1 or backward > 2:
                return (round(bias_score, 3), f"{dominant}_negative_stereotype")
            else:
                return (round(bias_score, 3), f"{dominant}_focus")

        return udf(
            detect_region,
            StructType(
                [StructField("score", FloatType()), StructField("type", StringType())]
            ),
        )

    def _detect_socioeconomic_bias_udf(self):
        """UDF for socioeconomic bias detection"""
        socio_kw = self.socioeconomic_keywords_bc.value

        def detect_socioeconomic(text: str) -> Tuple[float, str]:
            if not text:
                return (0.0, "neutral")

            text_lower = text.lower()

            wealthy_count = sum(1 for t in socio_kw["wealthy"] if t in text_lower)
            poor_count = sum(1 for t in socio_kw["poor"] if t in text_lower)

            total = wealthy_count + poor_count
            if total == 0:
                return (0.0, "neutral")

            # Check stereotypes
            elite_negative = sum(
                1
                for t in socio_kw["stereotypes"]["elite_disconnected"]
                if t in text_lower
            )
            poor_negative = sum(
                1 for t in socio_kw["stereotypes"]["poor_negative"] if t in text_lower
            )

            bias_score = abs(wealthy_count - poor_count) / total
            if elite_negative > 0 or poor_negative > 0:
                bias_score = min(
                    bias_score + (elite_negative + poor_negative) * 0.09, 1.0
                )

            if bias_score < 0.25:
                return (round(bias_score, 3), "neutral")
            elif wealthy_count > poor_count:
                return (round(bias_score, 3), "wealthy_focus")
            else:
                return (round(bias_score, 3), "poverty_focus")

        return udf(
            detect_socioeconomic,
            StructType(
                [StructField("score", FloatType()), StructField("type", StringType())]
            ),
        )

    def _detect_political_bias_udf(self):
        """UDF for political bias detection"""
        political_kw = self.political_keywords_bc.value

        def detect_political(text: str) -> Tuple[float, str]:
            if not text:
                return (0.0, "neutral")

            text_lower = text.lower()

            left_count = sum(1 for t in political_kw["left_leaning"] if t in text_lower)
            right_count = sum(
                1 for t in political_kw["right_leaning"] if t in text_lower
            )
            center_count = sum(1 for t in political_kw["centrist"] if t in text_lower)

            total = left_count + right_count + center_count
            if total == 0:
                return (0.0, "neutral")

            # Calculate normalized scores
            left_score = left_count / total
            right_score = right_count / total
            center_score = center_count / total

            max_score = max(left_score, right_score, center_score)

            if max_score < 0.4:  # No clear dominance
                return (round(max_score, 3), "neutral")
            elif left_score == max_score:
                return (round(left_score, 3), "left_leaning")
            elif right_score == max_score:
                return (round(right_score, 3), "right_leaning")
            else:
                return (round(center_score, 3), "centrist")

        return udf(
            detect_political,
            StructType(
                [StructField("score", FloatType()), StructField("type", StringType())]
            ),
        )

    def _calculate_overall_bias_udf(self):
        """UDF to calculate overall bias score"""
        weights = self.config.bias_weights

        def calculate_overall(
            political: float,
            gender: float,
            religious: float,
            caste: float,
            region: float,
            socioeconomic: float,
        ) -> float:
            weighted = (
                weights["political"] * (political or 0.0)
                + weights["gender"] * (gender or 0.0)
                + weights["religious"] * (religious or 0.0)
                + weights["caste"] * (caste or 0.0)
                + weights["region"] * (region or 0.0)
                + weights["socioeconomic"] * (socioeconomic or 0.0)
            )
            return round(weighted, 3)

        return udf(calculate_overall, FloatType())

    def analyze_articles(self, df: DataFrame) -> DataFrame:
        """
        Perform comprehensive bias analysis on articles DataFrame

        Args:
            df: Input DataFrame with at least 'article_text' column

        Returns:
            DataFrame with bias analysis columns added
        """
        # Ensure article_text is string and handle nulls
        df = df.withColumn(
            "article_text", coalesce(col("article_text").cast(StringType()), lit(""))
        )

        # Detect all bias dimensions
        gender_udf = self._detect_gender_bias_udf()
        religious_udf = self._detect_religious_bias_udf()
        caste_udf = self._detect_caste_bias_udf()
        region_udf = self._detect_region_bias_udf()
        socioeconomic_udf = self._detect_socioeconomic_bias_udf()
        political_udf = self._detect_political_bias_udf()
        overall_udf = self._calculate_overall_bias_udf()

        # Apply bias detection
        df = (
            df.withColumn("gender_bias_struct", gender_udf(col("article_text")))
            .withColumn("religious_bias_struct", religious_udf(col("article_text")))
            .withColumn("caste_bias_struct", caste_udf(col("article_text")))
            .withColumn("region_bias_struct", region_udf(col("article_text")))
            .withColumn(
                "socioeconomic_bias_struct", socioeconomic_udf(col("article_text"))
            )
            .withColumn("political_bias_struct", political_udf(col("article_text")))
        )

        # Extract scores and types
        df = (
            df.withColumn("gender_bias", col("gender_bias_struct.score"))
            .withColumn("gender_type", col("gender_bias_struct.type"))
            .withColumn("religious_bias", col("religious_bias_struct.score"))
            .withColumn("religious_type", col("religious_bias_struct.type"))
            .withColumn("caste_bias", col("caste_bias_struct.score"))
            .withColumn("caste_type", col("caste_bias_struct.type"))
            .withColumn("region_bias", col("region_bias_struct.score"))
            .withColumn("region_type", col("region_bias_struct.type"))
            .withColumn("socioeconomic_bias", col("socioeconomic_bias_struct.score"))
            .withColumn("socioeconomic_type", col("socioeconomic_bias_struct.type"))
            .withColumn("political_bias", col("political_bias_struct.score"))
            .withColumn("political_type", col("political_bias_struct.type"))
        )

        # Calculate overall bias score
        df = df.withColumn(
            "overall_bias_score",
            overall_udf(
                col("political_bias"),
                col("gender_bias"),
                col("religious_bias"),
                col("caste_bias"),
                col("region_bias"),
                col("socioeconomic_bias"),
            ),
        )

        # Drop intermediate struct columns
        df = df.drop(
            "gender_bias_struct",
            "religious_bias_struct",
            "caste_bias_struct",
            "region_bias_struct",
            "socioeconomic_bias_struct",
            "political_bias_struct",
        )

        return df

    def get_summary_stats(self, df: DataFrame) -> Dict:
        """Calculate summary statistics for bias analysis"""
        stats = {
            "total_articles": df.count(),
            "avg_overall_bias": df.agg({"overall_bias_score": "avg"}).collect()[0][0],
            "avg_political_bias": df.agg({"political_bias": "avg"}).collect()[0][0],
            "avg_gender_bias": df.agg({"gender_bias": "avg"}).collect()[0][0],
            "avg_religious_bias": df.agg({"religious_bias": "avg"}).collect()[0][0],
            "political_types": df.groupBy("political_type").count().collect(),
            "gender_types": df.groupBy("gender_type").count().collect(),
        }
        return stats
