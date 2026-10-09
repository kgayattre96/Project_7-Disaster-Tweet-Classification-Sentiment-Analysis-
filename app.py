
import streamlit as st
import torch
import torch.nn.functional as F
import nltk

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Disaster Tweet Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. CUSTOM STYLING
# ==================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1150px;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 3. APPLICATION HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🌍 Disaster Tweet Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Analyze disaster-related tweets using a fine-tuned DistilBERT
    model and examine their emotional tone using VADER sentiment analysis.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

MODEL_PATH = "distilbert_disaster_model"


@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


@st.cache_resource
def load_sentiment_analyzer():

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")

    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    return SentimentIntensityAnalyzer()


try:
    tokenizer, model, device = load_model()
    sentiment_analyzer = load_sentiment_analyzer()

except Exception as error:
    st.error(f"Unable to load the model or sentiment analyzer: {error}")
    st.stop()


# ==================================================
# 5. EXAMPLE TWEETS
# ==================================================

EXAMPLES = {
    "Choose an example...": "",

    "Disaster: Wildfire evacuation":
        "Severe wildfires are spreading rapidly, forcing thousands of residents to evacuate.",

    "Disaster: Flooding":
        "Severe flooding has forced hundreds of families to evacuate their homes.",

    "Ambiguous: Earthquake movie":
        "I watched a movie about a massive earthquake yesterday.",

    "Non-Disaster: Happy day":
        "Had a wonderful day with friends and enjoyed a great dinner."
}


def select_example():

    selected = st.session_state.get("example_select", "")

    st.session_state["tweet_input"] = EXAMPLES.get(
        selected, ""
    )


def clear_inputs():

    st.session_state["tweet_input"] = ""
    st.session_state["example_select"] = "Choose an example..."


# ==================================================
# 6. TWEET INPUT
# ==================================================

st.subheader("📝 Analyze a Tweet")

st.selectbox(
    "Try a sample tweet",
    options=list(EXAMPLES.keys()),
    key="example_select",
    on_change=select_example
)

tweet = st.text_area(
    "Tweet text",
    key="tweet_input",
    height=140,
    max_chars=5000,
    placeholder="Enter a tweet to analyze its disaster classification and sentiment..."
)

button_col1, button_col2 = st.columns([3, 1])

with button_col1:
    analyze_button = st.button(
        "🔍 Analyze Tweet",
        type="primary",
        use_container_width=True
    )

with button_col2:
    st.button(
        "🗑️ Clear",
        on_click=clear_inputs,
        use_container_width=True
    )


# ==================================================
# 7. PREDICTION FUNCTION
# ==================================================

def predict_tweet(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_id = int(
        torch.argmax(probabilities).item()
    )

    prediction = model.config.id2label[predicted_id]

    confidence = float(
        probabilities[predicted_id].item()
    )

    # Normalize class labels.
    normalized_label = (
        str(prediction).strip().lower().replace("_", "-")
    )

    if normalized_label in ("1", "disaster", "label-1"):
        prediction = "Disaster"

    elif normalized_label in (
        "0", "non-disaster", "non disaster", "label-0"
    ):
        prediction = "Non-Disaster"

    # VADER sentiment analysis.
    sentiment_scores = sentiment_analyzer.polarity_scores(text)

    compound_score = sentiment_scores["compound"]

    if compound_score >= 0.05:
        sentiment = "Positive"

    elif compound_score <= -0.05:
        sentiment = "Negative"

    else:
        sentiment = "Neutral"

    return {
        "prediction": prediction,
        "confidence": confidence,
        "sentiment": sentiment,
        "sentiment_score": compound_score
    }


# ==================================================
# 8. ANALYZE TWEET
# ==================================================

if analyze_button:

    if not tweet.strip():

        st.warning("Please enter a tweet before analyzing.")

    else:

        with st.spinner("Analyzing your tweet..."):

            try:
                result = predict_tweet(tweet.strip())

            except Exception as error:
                st.error(f"Analysis failed: {error}")
                st.stop()

        st.divider()

        # Only ONE Analysis Results heading.
        st.subheader("📊 Analysis Results")

        # --------------------------------------------------
        # DISASTER CLASSIFICATION AND SENTIMENT
        # --------------------------------------------------

        col1, col2 = st.columns(2, gap="medium")

        with col1:

            st.markdown("### 🚨 Disaster Classification")

            if result["prediction"] == "Disaster":

                st.error("🚨 Disaster Tweet")

            elif result["prediction"] == "Non-Disaster":

                st.success("✅ Non-Disaster Tweet")

            else:

                st.info(result["prediction"])

            st.metric(
                "Model Confidence",
                f'{result["confidence"]:.2%}'
            )

            st.progress(result["confidence"])

        with col2:

            st.markdown("### 💭 Sentiment Analysis")

            sentiment_icons = {
                "Positive": "😊",
                "Negative": "😟",
                "Neutral": "😐"
            }

            icon = sentiment_icons[result["sentiment"]]

            if result["sentiment"] == "Positive":

                st.success(f'{icon} {result["sentiment"]}')

            elif result["sentiment"] == "Negative":

                st.warning(f'{icon} {result["sentiment"]}')

            else:

                st.info(f'{icon} {result["sentiment"]}')

            st.metric(
                "VADER Sentiment Score",
                f'{result["sentiment_score"]:.4f}',
                help="Scores range from -1 (negative) to +1 (positive)."
            )

            # Map the score from [-1, 1] to [0, 1] for display.
            sentiment_progress = (
                result["sentiment_score"] + 1
            ) / 2

            st.progress(sentiment_progress)

        # --------------------------------------------------
        # CONFIDENCE INTERPRETATION
        # --------------------------------------------------

        st.subheader("🔎 Prediction Confidence")

        if result["confidence"] >= 0.85:

            st.success(
                "High model confidence. The model strongly favors "
                "the predicted class, but the prediction may still be wrong."
            )

        elif result["confidence"] >= 0.60:

            st.info(
                "Moderate model confidence. Review the tweet context "
                "before relying on the prediction."
            )

        else:

            st.warning(
                "Low model confidence. The model is relatively uncertain. "
                "Consider reviewing the tweet manually."
            )

        st.caption(
            "Model confidence is not a guarantee of correctness. "
            "Disaster classification and sentiment analysis are independent."
        )


# ==================================================
# 9. ABOUT THIS APPLICATION
# ==================================================

with st.expander("ℹ️ About this application"):

    st.markdown(
        """
        **Disaster classification**

        - Uses a fine-tuned DistilBERT sequence-classification model.
        - Predicts whether a tweet belongs to the Disaster or Non-Disaster class.

        **Sentiment analysis**

        - Uses VADER to calculate a compound sentiment score.
        - Categorizes the score as Positive, Negative, or Neutral.

        **Important limitation**

        - Predictions depend on the training data and the wording of the tweet.
        - Results should not replace independent verification of a real-world emergency.
        """
    )


# ==================================================
# 10. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="small-note">
    NLP Project · DistilBERT · VADER Sentiment Analysis
    </div>
    """,
    unsafe_allow_html=True
)
