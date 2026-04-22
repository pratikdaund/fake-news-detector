import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

MODELS_DIR = "models"

# ---------- NLTK ----------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOP_WORDS, LEMMATIZER = setup_nltk()

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([
        LEMMATIZER.lemmatize(w)
        for w in text.split()
        if w not in STOP_WORDS and len(w) > 2
    ])

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    try:
        with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vec = pickle.load(f)
        return model, vec
    except:
        return None, None

# ---------- PREDICT ----------
def predict(text):
    model, vec = load_model()

    if model is None:
        return None

    cleaned = clean_text(text)
    if not cleaned:
        return "UNKNOWN", 0

    x = vec.transform([cleaned])
    prob = model.predict_proba(x)[0][1]

    label = "REAL" if prob >= 0.5 else "FAKE"
    confidence = prob if label == "REAL" else 1 - prob

    return label, confidence

# ---------- UI ----------
st.title("📰 Fake News Detector")

text = st.text_area("Enter News Article")

if st.button("Analyze"):
    if len(text.strip()) < 20:
        st.error("Enter at least 20 characters")
    else:
        result = predict(text)

        if result is None:
            st.error("Model not found. Run train.py first.")
        else:
            label, conf = result

            if label == "REAL":
                st.success(f"✅ REAL ({conf*100:.2f}%)")
            else:
                st.error(f"🚨 FAKE ({conf*100:.2f}%)")
