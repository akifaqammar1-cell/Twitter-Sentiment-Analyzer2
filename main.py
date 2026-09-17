import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import time
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = Path(__file__).parent / "models"


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

naive_bayes_model = joblib.load(
    MODEL_DIR / "naive_bayes_model.pkl"
)

tfidf_vectorizer = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)


# ============================================================
# LOAD DEEP LEARNING MODELS
# ============================================================

rnn_model = load_model(
    MODEL_DIR / "sentiment_rnn.keras"
)

lstm_model = load_model(
    MODEL_DIR / "sentiment_lstm.keras"
)

gru_model = load_model(
    MODEL_DIR / "sentiment_gru.keras"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = joblib.load(
    MODEL_DIR / "tokenizer.pkl"
)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

label_encoder = joblib.load(
    MODEL_DIR / "label_encoder.pkl"
)


# ============================================================
# DEEP LEARNING MODEL ACCURACIES AND LOSSES
# ============================================================
# These are the actual results from your notebook.
# ============================================================

SIMPLE_RNN_ACCURACY = 0.972000002861023
SIMPLE_RNN_LOSS = 0.13127963244915009

LSTM_ACCURACY = 0.9549999833106995
LSTM_LOSS = 0.21013255417346954

GRU_ACCURACY = 0.9599999785423279
GRU_LOSS = 0.23639418184757233


# ============================================================
# ACCURACY DATAFRAME
# ============================================================

accuracy_data = pd.DataFrame(
    {
        "Model": [
            "SimpleRNN",
            "LSTM",
            "GRU"
        ],
        "Accuracy (%)": [
            SIMPLE_RNN_ACCURACY * 100,
            LSTM_ACCURACY * 100,
            GRU_ACCURACY * 100
        ]
    }
)


# ============================================================
# LOSS DATAFRAME
# ============================================================

loss_data = pd.DataFrame(
    {
        "Model": [
            "SimpleRNN",
            "LSTM",
            "GRU"
        ],
        "Loss": [
            SIMPLE_RNN_LOSS,
            LSTM_LOSS,
            GRU_LOSS
        ]
    }
)


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_tweet(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove @mentions
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # Remove hashtag symbol
    text = re.sub(
        r"#",
        "",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        color: gray;
        font-size: 17px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '💬 Twitter Sentiment Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze Twitter text using Machine Learning and Deep Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SAMPLE TWEETS
# ============================================================

st.subheader("⚡ Try a Sample Tweet")

sample1, sample2, sample3 = st.columns(3)


if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""


with sample1:

    if st.button(
        "🔥 Loved the service!",
        use_container_width=True
    ):

        st.session_state.tweet_input = (
            "I absolutely loved the service! 🔥"
        )


with sample2:

    if st.button(
        "😡 Worst delay ever!",
        use_container_width=True
    ):

        st.session_state.tweet_input = (
            "This is the worst delay ever! 😡"
        )


with sample3:

    if st.button(
        "😊 Amazing experience!",
        use_container_width=True
    ):

        st.session_state.tweet_input = (
            "I had an amazing experience! 😊"
        )


# ============================================================
# MODEL SELECTION
# ============================================================

st.subheader("🤖 Select Model")


model_choice = st.selectbox(
    "Choose the model for prediction:",
    [
        "ML - Naive Bayes",
        "DL - SimpleRNN",
        "DL - LSTM",
        "DL - GRU"
    ]
)


# ============================================================
# TWEET INPUT
# ============================================================

tweet = st.text_area(
    "📝 Enter your tweet:",
    value=st.session_state.tweet_input,
    placeholder="Example: I really love this game! 😍",
    height=130
)


# ============================================================
# PREDICT SENTIMENT
# ============================================================

if st.button(
    "🔍 Predict Sentiment",
    use_container_width=True
):

    if not tweet.strip():

        st.warning(
            "⚠️ Please enter a tweet first."
        )

    else:

        # ----------------------------------------------------
        # CLEAN TWEET
        # ----------------------------------------------------

        cleaned_tweet = clean_tweet(tweet)


        # ----------------------------------------------------
        # VIEW TEXT PREPROCESSING
        # ----------------------------------------------------

        with st.expander(
            "🔎 View Text Preprocessing"
        ):

            st.write("**Original Tweet:**")

            st.code(tweet)

            st.write("**Cleaned Tweet:**")

            st.code(cleaned_tweet)


        # ====================================================
        # NAIVE BAYES PREDICTION
        # ====================================================

        if model_choice == "ML - Naive Bayes":

            start_time = time.perf_counter()


            # Convert tweet into TF-IDF
            tweet_tfidf = tfidf_vectorizer.transform(
                [cleaned_tweet]
            )


            # Predict sentiment
            prediction = naive_bayes_model.predict(
                tweet_tfidf
            )


            # Get probabilities
            probabilities = (
                naive_bayes_model.predict_proba(
                    tweet_tfidf
                )[0]
            )


            end_time = time.perf_counter()


            # Convert encoded label into sentiment
            sentiment = label_encoder.inverse_transform(
                prediction
            )[0]


            inference_time = (
                end_time - start_time
            ) * 1000


        # ====================================================
        # DEEP LEARNING PREDICTION
        # ====================================================

        else:

            start_time = time.perf_counter()


            # ------------------------------------------------
            # Convert tweet to sequence
            # ------------------------------------------------

            sequence = tokenizer.texts_to_sequences(
                [cleaned_tweet]
            )


            # ------------------------------------------------
            # Pad sequence
            # ------------------------------------------------
            # maxlen=100 must match the value used during
            # training of your Deep Learning models.
            # ------------------------------------------------

            padded_sequence = pad_sequences(
                sequence,
                maxlen=100,
                padding="post",
                truncating="post"
            )


            # ------------------------------------------------
            # Select Deep Learning model
            # ------------------------------------------------

            if model_choice == "DL - SimpleRNN":

                selected_model = rnn_model


            elif model_choice == "DL - LSTM":

                selected_model = lstm_model


            elif model_choice == "DL - GRU":

                selected_model = gru_model


            # ------------------------------------------------
            # Make prediction
            # ------------------------------------------------

            prediction_probabilities = (
                selected_model.predict(
                    padded_sequence,
                    verbose=0
                )
            )


            # ------------------------------------------------
            # Get predicted class
            # ------------------------------------------------

            prediction = np.argmax(
                prediction_probabilities,
                axis=1
            )


            # ------------------------------------------------
            # Get probabilities
            # ------------------------------------------------

            probabilities = (
                prediction_probabilities[0]
            )


            end_time = time.perf_counter()


            # ------------------------------------------------
            # Convert prediction into sentiment
            # ------------------------------------------------

            sentiment = label_encoder.inverse_transform(
                prediction
            )[0]


            inference_time = (
                end_time - start_time
            ) * 1000


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.subheader("🎯 Prediction Result")


        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        with col1:

            st.metric(
                "Sentiment",
                sentiment
            )


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        with col2:

            st.metric(
                "Model",
                model_choice
            )


        # ----------------------------------------------------
        # INFERENCE TIME
        # ----------------------------------------------------

        with col3:

            st.metric(
                "Inference Time",
                f"{inference_time:.2f} ms"
            )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        confidence = (
            np.max(probabilities) * 100
        )


        st.write(
            f"### 🎯 Confidence: {confidence:.2f}%"
        )


        # ====================================================
        # SENTIMENT PROBABILITY GRAPH
        # ====================================================

        st.subheader(
            "📊 Sentiment Probability"
        )


        class_names = label_encoder.classes_


        probability_percent = (
            probabilities * 100
        )


        probability_df = pd.DataFrame(
            {
                "Sentiment": class_names,
                "Probability (%)": probability_percent
            }
        )


        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )


        st.bar_chart(
            probability_df.set_index(
                "Sentiment"
            )
        )


        # ====================================================
        # PROBABILITY DETAILS
        # ====================================================

        st.subheader(
            "📋 Probability Details"
        )


        display_df = probability_df.copy()


        display_df["Probability (%)"] = (
            display_df["Probability (%)"].round(2)
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.subheader(
            "💡 Interpretation"
        )


        st.write(
            f"The **{model_choice}** model classified "
            f"the tweet as **{sentiment}** with a "
            f"confidence of **{confidence:.2f}%**."
        )


# ============================================================
# DEEP LEARNING ACCURACY COMPARISON
# ============================================================

st.divider()

st.subheader(
    "📈 Deep Learning Model Accuracy Comparison"
)


st.write(
    "Comparison of test accuracy for SimpleRNN, LSTM and GRU."
)


# Accuracy graph
st.bar_chart(
    accuracy_data.set_index(
        "Model"
    )
)


# Accuracy table
st.subheader(
    "📋 Accuracy Results"
)


accuracy_display = accuracy_data.copy()


accuracy_display["Accuracy (%)"] = (
    accuracy_display["Accuracy (%)"].round(2)
)


st.dataframe(
    accuracy_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DEEP LEARNING LOSS COMPARISON
# ============================================================

st.subheader(
    "📉 Deep Learning Model Loss Comparison"
)


st.write(
    "Comparison of test loss for SimpleRNN, LSTM and GRU."
)


# Loss graph
st.bar_chart(
    loss_data.set_index(
        "Model"
    )
)


# Loss table
st.subheader(
    "📋 Loss Results"
)


loss_display = loss_data.copy()


loss_display["Loss"] = (
    loss_display["Loss"].round(4)
)


st.dataframe(
    loss_display,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()


with st.expander(
    "ℹ️ About This Project"
):

    st.write(
        """
        **Twitter Sentiment Analyzer** is a Natural Language
        Processing (NLP) application that classifies tweets
        into four sentiment categories:

        - Positive
        - Negative
        - Neutral
        - Irrelevant

        **Machine Learning Model:**

        - Naive Bayes

        **Deep Learning Models:**

        - SimpleRNN
        - LSTM
        - GRU

        The application allows users to enter a tweet and
        select a trained Machine Learning or Deep Learning
        model to predict its sentiment.

        Text preprocessing includes converting text to lowercase,
        removing URLs, removing mentions, removing hashtags,
        and removing unnecessary characters.

        LSTM and GRU are sequence-based Deep Learning models
        that can capture relationships between words in text.

        The application also provides accuracy and loss
        comparison graphs for the Deep Learning models.
        """
    )