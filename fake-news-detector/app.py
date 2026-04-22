"""
app.py - Fake News Detection (Streamlit UI + inference)
========================================================
Run locally:   streamlit run app.py
Deploy:        push to GitHub, connect to Render / Streamlit Cloud.
"""
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------- Page config (must be first Streamlit call) ----------
st.set_page_config(
    page_title="Veritas · Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"
MAX_LEN = 300

# ---------- NLTK setup (cached so Render doesn't redownload every request) ----------
@st.cache_resource
def setup_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()

STOP_WORDS, LEMMATIZER = setup_nltk()


# ---------- Preprocessing (MUST match train.py) ----------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in text.split()
        if tok not in STOP_WORDS and len(tok) > 2
    ]
    return " ".join(tokens)


# ---------- Model loaders (cached, loaded once) ----------
@st.cache_resource
def load_logreg():
    try:
        with open(os.path.join(MODELS_DIR, "logreg_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vec = pickle.load(f)
        return model, vec
        if model is None:
            return None


@st.cache_resource
def load_lstm():
    try:
        from tensorflow.keras.models import load_model
        path = os.path.join(MODELS_DIR, "lstm_model.keras")
        if not os.path.exists(path):
            return None, None
        model = load_model(path)
        with open(os.path.join(MODELS_DIR, "lstm_tokenizer.pkl"), "rb") as f:
            tok = pickle.load(f)
        return model, tok
    except Exception as e:
        st.warning(f"LSTM not loaded: {e}")
        return None, None


# ---------- Prediction ----------
def predict_logreg(text):
    model, vec = load_logreg()
    if model is None:
        return None
    cleaned = clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0, "model": "LogReg",
                "tokens": 0, "cleaned": cleaned}
    x = vec.transform([cleaned])
    real_prob = float(model.predict_proba(x)[0][1])
    label = "REAL" if real_prob >= 0.5 else "FAKE"
    conf = real_prob if label == "REAL" else 1 - real_prob
    return {
        "label": label, "confidence": conf, "model": "TF-IDF + Logistic Regression",
        "tokens": len(cleaned.split()), "real_prob": real_prob, "cleaned": cleaned,
    }


def predict_lstm(text):
    model, tok = load_lstm()
    if model is None:
        return None
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned = clean_text(text)
    if not cleaned:
        return {"label": "UNKNOWN", "confidence": 0.0, "model": "LSTM",
                "tokens": 0, "cleaned": cleaned}
    seq = tok.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    real_prob = float(model.predict(padded, verbose=0).ravel()[0])
    label = "REAL" if real_prob >= 0.5 else "FAKE"
    conf = real_prob if label == "REAL" else 1 - real_prob
    return {
        "label": label, "confidence": conf, "model": "Bidirectional LSTM",
        "tokens": len(cleaned.split()), "real_prob": real_prob, "cleaned": cleaned,
    }


# ---------- Custom CSS ----------
st.markdown("""
<style>
    .stApp { background: #f5f1e8; }
    .main .block-container { padding-top: 2rem; max-width: 820px; }
    h1, h2, h3 { font-family: Georgia, serif; color: #111; }
    .result-box {
        padding: 1.5rem; border: 2px solid #111; margin: 1rem 0;
        box-shadow: 6px 6px 0 #111; background: #fff;
    }
    .real-box { border-color: #15803d; box-shadow: 6px 6px 0 #15803d; }
    .fake-box { border-color: #b91c1c; box-shadow: 6px 6px 0 #b91c1c; }
    .verdict { font-size: 3rem; font-family: Georgia, serif; font-weight: bold; margin: 0; }
    .real-txt { color: #15803d; }
    .fake-txt { color: #b91c1c; }
    .stButton>button {
        background: #111; color: #f5f1e8; border: 2px solid #111;
        font-weight: 700; padding: 0.5rem 1.5rem; border-radius: 0;
        letter-spacing: 0.05em; text-transform: uppercase; font-size: 0.85rem;
    }
    .stButton>button:hover { background: #b91c1c; border-color: #b91c1c; color: #fff; }
    .stTextArea textarea {
        font-family: Georgia, serif; font-size: 1rem;
        border: 1px solid #111; border-radius: 0; background: #fff;
    }
    .eyebrow {
        font-family: 'Courier New', monospace; font-size: 0.75rem;
        letter-spacing: 0.2em; text-transform: uppercase; color: #b91c1c;
        border: 1px solid #b91c1c; padding: 0.25rem 0.6rem; display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    logreg_available = load_logreg()[0] is not None
    lstm_available = load_lstm()[0] is not None

    options = []
    if logreg_available:
        options.append("Logistic Regression (fast)")
    if lstm_available:
        options.append("Bidirectional LSTM (deep learning)")

    if not options:
        st.error("No trained models found. Run `python train.py` first.")
        model_choice = None
    else:
        model_choice = st.radio("Model", options, index=0)

    st.markdown("---")
    st.markdown("### 📊 System")
    st.markdown(f"- LogReg: {'✅' if logreg_available else '❌'}")
    st.markdown(f"- LSTM: {'✅' if lstm_available else '❌'}")

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown(
        "M.Tech project implementing fake news detection using NLP and deep learning. "
        "Trained on the ISOT dataset. "
        "See README.md for full documentation."
    )

    with st.expander("⚠️ Limitations"):
        st.markdown(
            "- Trained on one dataset (ISOT); may not generalise to real-world news.\n"
            "- Performs best on English, news-style writing.\n"
            "- Cannot detect satire, manipulated imagery, or factual errors in well-written text.\n"
            "- This is an academic demonstrator, not a production fact-checker."
        )


# ---------- Main UI ----------
st.markdown('<p class="eyebrow">M.TECH PROJECT · NLP + DEEP LEARNING</p>',
            unsafe_allow_html=True)
st.markdown("# Veritas")
st.markdown(
    "##### *Classify the veracity of news articles using TF-IDF and a Bi-LSTM neural network.*"
)
st.markdown("---")

# Sample articles
with st.expander("📝 Try a sample article"):
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load sample REAL article"):
            st.session_state.text = (
                "The United Nations Security Council on Tuesday unanimously adopted "
                "a resolution calling for a cessation of hostilities during the "
                "upcoming holiday period. The 15-member body passed the measure "
                "after weeks of negotiations between permanent members. The "
                "resolution urges all parties to allow humanitarian aid to reach "
                "civilians in affected regions and calls for the release of hostages."
            )
    with col2:
        if st.button("Load sample FAKE article"):
            st.session_state.text = (
                "SHOCKING: Scientists have discovered that drinking lemon water "
                "every morning can cure all forms of cancer in just 7 days! "
                "Big Pharma doesn't want you to know this ONE WEIRD TRICK that "
                "doctors HATE. A secret study suppressed by the government shows "
                "that the acidic properties of citrus destroy tumors overnight. "
                "Share this before it gets deleted!"
            )

text = st.text_area(
    "**Paste the news article**",
    value=st.session_state.get("text", ""),
    height=250,
    placeholder="Paste the full article text here, including the headline if available.",
    key="text_input",
)

# Stats row
col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    st.caption(f"**{len(text)}** chars")
with col_b:
    words = len(text.split()) if text else 0
    st.caption(f"**{words}** words")
with col_c:
    if words < 30 and words > 0:
        st.caption("⚠️ Short articles give unreliable predictions")

analyse = st.button("🔍 Analyse Article", type="primary", disabled=(model_choice is None))

# ---------- Prediction ----------
if analyse:
    if not text or len(text.strip()) < 20:
        st.error("Please enter at least 20 characters of text.")
    else:
        with st.spinner("Analysing..."):
            if "Logistic" in model_choice:
                result = predict_logreg(text)
            else:
                result = predict_lstm(text)

        if result is None:
            st.error("Model not available. Did you run `python train.py`?")
        else:
            label = result["label"]
            conf = result["confidence"]
            pct = conf * 100

            box_class = "real-box" if label == "REAL" else "fake-box"
            txt_class = "real-txt" if label == "REAL" else "fake-txt"

            st.markdown(
                f"""
                <div class="result-box {box_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem;">
                        <div>
                            <div style="font-family: monospace; font-size: 0.7rem; letter-spacing: 0.2em; color: #666;">VERDICT</div>
                            <div class="verdict {txt_class}">{label}</div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-family: monospace; font-size: 0.7rem; letter-spacing: 0.2em; color: #666;">CONFIDENCE</div>
                            <div style="font-size: 2rem; font-family: monospace; font-weight: bold;">{pct:.1f}%</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(conf)

            colx, coly, colz = st.columns(3)
            colx.metric("Model", result["model"].split()[-1])
            coly.metric("Tokens analysed", result["tokens"])
            level = "High" if conf >= 0.9 else "Moderate" if conf >= 0.7 else "Low"
            colz.metric("Confidence level", level)

            # Probability breakdown
            with st.expander("🔬 Detailed probabilities"):
                real_prob = result["real_prob"]
                st.write(f"**P(REAL)** = `{real_prob:.4f}`")
                st.write(f"**P(FAKE)** = `{1 - real_prob:.4f}`")
                st.write(f"**Decision threshold** = `0.5`")

            with st.expander("🧹 Preprocessed text (what the model actually sees)"):
                st.code(result["cleaned"][:1000] + ("..." if len(result["cleaned"]) > 1000 else ""))

            if conf < 0.7:
                st.warning(
                    "⚠️ Low confidence — the model is uncertain about this article. "
                    "Treat the verdict with scepticism."
                )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<p style='color: #666; font-style: italic; font-family: Georgia, serif;'>"
    "<b>Academic disclaimer.</b> This model was trained on the ISOT Fake News Dataset. "
    "Its predictions reflect the patterns in that specific corpus and should not be "
    "used as a sole arbiter of truth. Always cross-check with reputable sources."
    "</p>",
    unsafe_allow_html=True,
)
