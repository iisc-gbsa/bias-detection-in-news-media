"""
Streamlit UI for News Article Bias Detection
Analyzes articles for various types of bias and displays scores
"""

import streamlit as st
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.bias_detection import BiasDetector
from datetime import datetime

# Path to sample articles
SAMPLE_ARTICLES_FILE = os.path.join(os.path.dirname(__file__), "sample_articles.json")


def load_sample_articles():
    """Load sample articles from JSON file"""
    if os.path.exists(SAMPLE_ARTICLES_FILE):
        with open(SAMPLE_ARTICLES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# Page configuration
st.set_page_config(
    page_title="News Bias Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .bias-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .bias-medium { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .bias-high { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .stTextArea textarea {
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_bias_detector():
    """Load and cache the bias detector model"""
    with st.spinner("Loading AI models... This may take a moment on first run."):
        detector = BiasDetector()
    return detector


def get_bias_level(score: float) -> tuple[str, str]:
    """Return bias level label and color class based on score"""
    if score < 0.3:
        return "Low", "bias-low"
    elif score < 0.6:
        return "Medium", "bias-medium"
    else:
        return "High", "bias-high"


def analyze_article_raw(detector: BiasDetector, article_text: str) -> dict:
    """
    Analyze article and return raw values matching the schema
    """
    if not article_text or not isinstance(article_text, str):
        return {
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
            "political_bias": 0.0,
            "political_type": "neutral",
            "overall_bias_score": 0.0,
            "word_count": 0,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    # Detect all bias dimensions
    political_bias, political_type = detector.detect_political_bias(article_text)
    gender_bias, gender_type = detector.detect_gender_bias(article_text)
    religious_bias, religious_type = detector.detect_religious_bias(article_text)
    caste_bias, caste_type = detector.detect_caste_bias(article_text)
    region_bias, region_type = detector.detect_region_bias(article_text)
    socioeconomic_bias, socioeconomic_type = detector.detect_socioeconomic_bias(
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
    overall_score = detector.calculate_overall_bias_score(bias_scores)

    return {
        "gender_bias": gender_bias,
        "gender_type": gender_type,
        "religious_bias": religious_bias,
        "religious_type": religious_type,
        "caste_bias": caste_bias,
        "caste_type": caste_type,
        "region_bias": region_bias,
        "region_type": region_type,
        "socioeconomic_bias": socioeconomic_bias,
        "socioeconomic_type": socioeconomic_type,
        "political_bias": political_bias,
        "political_type": political_type,
        "overall_bias_score": overall_score,
        "word_count": len(article_text.split()),
        "analysis_timestamp": datetime.now().isoformat(),
    }


def render_bias_metric(label: str, score: float, bias_type: str):
    """Render a single bias metric with visual indicator"""
    level, color_class = get_bias_level(score)

    # Progress bar color based on level
    if level == "Low":
        color = "#38ef7d"
    elif level == "Medium":
        color = "#F2C94C"
    else:
        color = "#f45c43"

    st.markdown(f"**{label}**")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(min(score, 1.0))
    with col2:
        st.markdown(f"**{score:.3f}**")

    st.caption(f"Type: `{bias_type}` | Level: **{level}**")
    st.markdown("---")


def main():
    # Header
    st.markdown(
        '<p class="main-header">📰 News Article Bias Detector</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Analyze news articles for gender, religious, caste, regional, socioeconomic, and political bias</p>',
        unsafe_allow_html=True,
    )

    # Load model
    detector = load_bias_detector()

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
        This tool analyzes news articles for multiple dimensions of bias:
        
        - **Gender Bias**: Male/female representation
        - **Religious Bias**: Religious group focus
        - **Caste Bias**: Caste-based language
        - **Regional Bias**: Geographic focus
        - **Socioeconomic Bias**: Class-based language
        - **Political Bias**: Left/right leaning
        
        ---
        
        **Bias Score Interpretation:**
        - 🟢 **0.0 - 0.3**: Low bias
        - 🟡 **0.3 - 0.6**: Medium bias  
        - 🔴 **0.6 - 1.0**: High bias
        
        ---
        
        *Powered by ensemble ML models using keyword matching, TF-IDF, and BERT embeddings.*
        """
        )

    # Main content area
    col_input, col_results = st.columns([1, 1])

    with col_input:
        st.subheader("📝 Enter Article Text")

        # Load sample articles from MongoDB export
        sample_articles = load_sample_articles()

        # Initialize session state
        if "selected_sample_idx" not in st.session_state:
            st.session_state.selected_sample_idx = 0

        # Sample selector dropdown
        if sample_articles:
            st.markdown("**📚 Load Sample Article (from MongoDB)**")
            sample_options = ["-- Select a sample --"] + [
                f"{s['category']}: {s['title'][:40]}..." for s in sample_articles
            ]

            selected_idx = st.selectbox(
                "Choose a sample article to analyze:",
                options=range(len(sample_options)),
                format_func=lambda x: sample_options[x],
                label_visibility="collapsed",
                key="sample_selector",
            )

            # Show description and load button for selected sample
            if selected_idx > 0:
                sample = sample_articles[selected_idx - 1]
                st.caption(f"📌 {sample['description']}")

                if st.button("📥 Load Selected Article", use_container_width=True):
                    st.session_state.article_text = sample["article_text"]
                    st.rerun()

            st.markdown("---")

        # Get current article text
        current_text = st.session_state.get("article_text", "")

        article_text = st.text_area(
            "Paste your news article here:",
            value=current_text,
            height=350,
            placeholder="Paste the full text of a news article to analyze its bias...",
            help="Enter the complete article text for accurate analysis",
        )

        # Store text in session state
        st.session_state.article_text = article_text

        # Action buttons
        col_analyze, col_clear = st.columns([2, 1])

        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.article_text = ""
                st.session_state.results = None
                st.rerun()

        with col_analyze:
            analyze_clicked = st.button(
                "🔍 Analyze Bias", type="primary", use_container_width=True
            )

    with col_results:
        st.subheader("📊 Bias Analysis Results")

        if analyze_clicked and article_text.strip():
            with st.spinner("Analyzing article..."):
                results = analyze_article_raw(detector, article_text)

            # Store results in session state
            st.session_state.results = results

        if "results" in st.session_state and st.session_state.results:
            results = st.session_state.results

            # Overall Score - Prominent display
            overall_level, overall_color = get_bias_level(results["overall_bias_score"])

            st.metric(
                label="🎯 Overall Bias Score",
                value=f"{results['overall_bias_score']:.4f}",
                delta=f"{overall_level} Bias Level",
            )

            st.markdown("---")

            # Word count
            st.caption(
                f"📄 Word Count: **{results['word_count']}** | 🕐 Analyzed: `{results['analysis_timestamp']}`"
            )

            st.markdown("---")

            # Individual bias metrics
            st.markdown("### Bias Breakdown")

            # Gender Bias
            render_bias_metric(
                "👫 Gender Bias", results["gender_bias"], results["gender_type"]
            )

            # Religious Bias
            render_bias_metric(
                "🕌 Religious Bias",
                results["religious_bias"],
                results["religious_type"],
            )

            # Caste Bias
            render_bias_metric(
                "📊 Caste Bias", results["caste_bias"], results["caste_type"]
            )

            # Regional Bias
            render_bias_metric(
                "🗺️ Regional Bias", results["region_bias"], results["region_type"]
            )

            # Socioeconomic Bias
            render_bias_metric(
                "💰 Socioeconomic Bias",
                results["socioeconomic_bias"],
                results["socioeconomic_type"],
            )

            # Political Bias
            render_bias_metric(
                "🏛️ Political Bias", results["political_bias"], results["political_type"]
            )

            # Export results as JSON
            st.markdown("---")
            st.download_button(
                label="📥 Download Results (JSON)",
                data=str(results),
                file_name=f"bias_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        elif analyze_clicked and not article_text.strip():
            st.warning("⚠️ Please enter some article text to analyze.")
        else:
            st.info("👈 Enter article text and click 'Analyze Bias' to see results.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>Built with Streamlit | Bias Detection using ML Ensemble Methods</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
