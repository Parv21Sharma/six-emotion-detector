import streamlit as st
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
nltk.download("stopwords")

st.set_page_config(
    page_title="Six Emotion Detector",
    page_icon="🧠",
    layout="wide"
)

with open("models/label_encoder.pkl", "rb") as file:
    lb = pickle.load(file)
with open("models/tfidf_vectorizer.pkl", "rb") as file:
    tfidf_vectorizer = pickle.load(file)
with open("models/logistic_regression/logistic_regression.pkl", "rb") as file:
    logistic_regression = pickle.load(file)
with open("models/random_forest/random_forest.pkl", "rb") as file:
    random_forest = pickle.load(file)
with open("models/svm/svm.pkl", "rb") as file:
    svm = pickle.load(file)
lstm_model = load_model("models/lstm/lstm_model.keras")
with open("models/lstm/vocab_info.pkl", "rb") as file:
    vocab_info = pickle.load(file)
vocab_size = vocab_info["vocab_size"]
max_len = vocab_info["max_len"]

stop_words = set(stopwords.words("english"))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

def sentence_cleaning(sentence):
    text = clean_text(sentence)
    one_hot_word = [one_hot(input_text=text, n=vocab_size)]
    pad = pad_sequences(
        sequences=one_hot_word,
        maxlen=max_len,
        padding="pre"
    )
    return pad

emotion_emoji = {
    "anger": "😠",
    "fear": "😨",
    "joy": "😊",
    "love": "❤️",
    "sadness": "😢",
    "surprise": "😲"
}

st.title("🧠 Six Emotion Detector")
st.markdown(
    "### Compare predictions from Machine Learning and Deep Learning models"
)
st.divider()

st.subheader("Enter Your Sentence")
user_input = st.text_area(
    "Type a sentence below:",
    height=120,
    placeholder="Example: I am really happy today!"
)

if st.button("🔍 Predict Emotion", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        cleaned_text = clean_text(user_input)
        tfidf_text = tfidf_vectorizer.transform([cleaned_text])

        # =========================
        # Logistic Regression
        # =========================

        lr_prediction = logistic_regression.predict(tfidf_text)
        lr_result = lb.inverse_transform(lr_prediction)[0]
        lr_confidence = (
            np.max(logistic_regression.predict_proba(tfidf_text)) * 100
        )

        # =========================
        # Random Forest
        # =========================

        rf_prediction = random_forest.predict(tfidf_text)
        rf_result = lb.inverse_transform(rf_prediction)[0]
        rf_confidence = (
            np.max(random_forest.predict_proba(tfidf_text)) * 100
        )

        # =========================
        # Support Vector Machine
        # =========================

        svm_prediction = svm.predict(tfidf_text)
        svm_result = lb.inverse_transform(svm_prediction)[0]
        svm_score = np.max(svm.decision_function(tfidf_text))

        # =========================
        # LSTM
        # =========================

        lstm_text = sentence_cleaning(user_input)
        lstm_prediction = lstm_model.predict(
            lstm_text,
            verbose=0
        )
        lstm_result = lb.inverse_transform(
            np.argmax(lstm_prediction, axis=-1)
        )[0]
        lstm_confidence = np.max(lstm_prediction) * 100

        # =========================
        # Results
        # =========================

        st.divider()
        st.subheader("📊 Model Comparison")
        col1, col2 = st.columns(2)

        # Logistic Regression

        with col1:
            st.markdown("### Logistic Regression")
            st.success(
                f"{emotion_emoji.get(lr_result, '🙂')} "
                f"**{lr_result.upper()}**"
            )
            st.metric(
                "Confidence",
                f"{lr_confidence:.2f}%"
            )

        # Random Forest

        with col2:
            st.markdown("### Random Forest")
            st.success(
                f"{emotion_emoji.get(rf_result, '🙂')} "
                f"**{rf_result.upper()}**"
            )
            st.metric(
                "Confidence",
                f"{rf_confidence:.2f}%"
            )
        col3, col4 = st.columns(2)

        # SVM

        with col3:
            st.markdown("### Support Vector Machine")
            st.success(
                f"{emotion_emoji.get(svm_result, '🙂')} "
                f"**{svm_result.upper()}**"
            )
            st.metric(
                "Decision Score",
                f"{svm_score:.2f}"
            )

        # LSTM

        with col4:
            st.markdown("### LSTM")
            st.success(
                f"{emotion_emoji.get(lstm_result, '🙂')} "
                f"**{lstm_result.upper()}**"
            )
            st.metric(
                "Confidence",
                f"{lstm_confidence:.2f}%"
            )

        # =========================
        # Consensus
        # =========================

        predictions = [
            lr_result,
            rf_result,
            svm_result,
            lstm_result
        ]
        prediction_counts = {}
        for emotion in predictions:
            prediction_counts[emotion] = prediction_counts.get(emotion, 0) + 1

        final_emotion = max(
            prediction_counts,
            key=prediction_counts.get
        )
        votes = prediction_counts[final_emotion]
        st.divider()
        st.subheader("🏆 Model Consensus")
        st.success(
            f"{emotion_emoji.get(final_emotion, '🙂')} "
            f"**{final_emotion.upper()}**"
        )

        st.write(
            f"**{votes} out of 4 models** predicted "
            f"**{final_emotion}**."
        )


        # =========================
        # Score Explanation
        # =========================

        st.divider()

        with st.expander("ℹ️ Understanding the Scores"):

            st.markdown("### Confidence Score")

            st.write(
                "The confidence score shows how strongly the model "
                "favors its predicted emotion. A higher percentage "
                "means the model is more confident in that prediction."
            )

            st.markdown("### SVM Decision Score")

            st.write(
                "The SVM decision score shows how strongly the SVM "
                "favors its predicted class. It is not a probability "
                "or percentage. A larger magnitude generally indicates "
                "a stronger decision."
            )

st.divider()
st.caption(
    "Six Emotion Detector • Machine Learning & Deep Learning"
)
