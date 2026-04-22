import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- Page Config ----------
st.set_page_config(
    page_title="Veritas · Fake News Detector",
    page_icon="📰",
    layout="centered"
)

MODELS_DIR = "models"

# ---------- NLTK Setup ----------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOP_WORDS, LEMMATIZER = setup_nltk()

# ---------- Text Cleaning ----------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        LEMMATIZER.lemmatize(word)
        for word in text.split()
        if word not in STOP_WORDS and len(word) > 2
    ]

    return " ".join(tokens)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    try:
        with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "rb") as f:
            model = pickle.load(f)

        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer

    except FileNotFoundError:
        return None, None

# ---------- Prediction ----------
def predict(text):
    model, vec = load_model()

    if model is None or vec is None:
        return None

    cleaned = clean_text(text)

    if not cleaned:
        return {
            "label": "UNKNOWN",
            "confidence": 0
        }

    x = vec.transform([cleaned])
    prob = model.predict_proba(x)[0][1]

    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if label == "REAL" else 1 - prob

    return {
        "label": label,
        "confidence": confidence,
        "prob_real": prob,
        "cleaned": cleaned
    }

# ---------- UI ----------
st.title("📰 Veritas - Fake News Detector")
st.markdown("Detect whether a news article is **REAL or FAKE** using NLP")

text = st.text_area(
    "Paste news article here",
    height=250,
    placeholder="Enter full news content..."
)

col1, col2 = st.columns(2)
with col1:
    st.caption(f"Characters: {len(text)}")
with col2:
    st.caption(f"Words: {len(text.split()) if text else 0}")

# ---------- Button ----------
if st.button("🔍 Analyze Article"):
    if len(text.strip()) < 20:
        st.error("⚠️ Please enter at least 20 characters.")
    else:
        with st.spinner("Analyzing..."):
            result = predict(text)

        if result is None:
            st.error("❌ Model not found. Run `train.py` and push model files.")
        else:
            label = result["label"]
            confidence = result["confidence"] * 100

            if label == "REAL":
                st.success(f"✅ REAL NEWS ({confidence:.2f}%)")
            else:
                st.error(f"🚨 FAKE NEWS ({confidence:.2f}%)")

            # Confidence bar
            st.progress(result["confidence"])

            # Details
            with st.expander("🔬 Detailed Info"):
                st.write(f"**P(REAL)**: {result['prob_real']:.4f}")
                st.write(f"**P(FAKE)**: {1 - result['prob_real']:.4f}")

            with st.expander("🧹 Cleaned Text"):
                st.code(result["cleaned"][:1000])

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<p style='color: gray;'>Academic project using TF-IDF + Logistic Regression. "
    "Results depend on training dataset.</p>",
    unsafe_allow_html=True
)
