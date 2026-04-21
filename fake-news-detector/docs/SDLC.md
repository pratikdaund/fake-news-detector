# SDLC Documentation — Fake News Detection System

*For inclusion in M.Tech dissertation / project report.*

---

## 1. Introduction

The proliferation of misinformation on digital platforms has become a significant
societal concern. This project develops an automated fake news classification
system using Natural Language Processing and deep learning, deployed as an
interactive web application.

### 1.1 Objectives
- Build a supervised text classifier capable of labelling news articles as
  real or fake with high accuracy on the ISOT benchmark dataset.
- Compare a classical NLP approach (TF-IDF + Logistic Regression) with a
  deep learning approach (Bidirectional LSTM).
- Deliver an end-to-end deployable system with a responsive user interface.
- Document the known limitations of dataset-driven fake news detection.

### 1.2 Scope
In scope: English-language text classification, supervised learning, web
deployment. Out of scope: multimodal (image/video) misinformation, fact
verification against external knowledge bases, real-time social-media scraping,
multilingual support.

---

## 2. SDLC Methodology

An **Iterative and Incremental** model was adopted — appropriate for
research-and-development projects where experimentation (hyperparameter tuning,
model comparison) is central. Waterfall was rejected because the model
architecture could not be finalised before initial empirical results.

The project proceeded through four iterations:

| Iteration | Focus | Outcome |
|-----------|-------|---------|
| 1 | Baseline classifier with TF-IDF + LogReg | Working CLI prediction script |
| 2 | Deep learning model (LSTM) | Two-model comparison notebook |
| 3 | Web UI (Streamlit) | Interactive demo running locally |
| 4 | Cloud deployment | Publicly accessible application on Render |

---

## 3. Requirements Analysis

### 3.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | The system shall accept news article text as input from the user. |
| FR-2 | The system shall classify the input as REAL or FAKE. |
| FR-3 | The system shall return a confidence score between 0 and 1. |
| FR-4 | The user shall be able to select between two trained models. |
| FR-5 | The system shall display decision internals (preprocessed text, probabilities). |
| FR-6 | The system shall reject inputs below a minimum character threshold. |
| FR-7 | The system shall be accessible via a public URL over HTTPS. |

### 3.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Model accuracy on ISOT test split | ≥ 95% |
| NFR-2 | Single prediction latency | < 2 s |
| NFR-3 | Cold-start time on free tier | < 60 s |
| NFR-4 | Reproducibility (seeded train/test split) | Deterministic |
| NFR-5 | UI must be responsive (mobile-friendly) | Works on 320px viewport |

### 3.3 Stakeholders
- **Primary:** Academic examiner / viva panel evaluating the project.
- **Secondary:** End users experimenting with the web demo.
- **Developer:** The M.Tech candidate.

---

## 4. System Design

### 4.1 Architecture

```
           ┌────────────────────────────────────────┐
           │            Browser (Client)            │
           │        HTML/CSS/JS via Streamlit       │
           └──────────────────┬─────────────────────┘
                              │ HTTPS
                              ▼
           ┌────────────────────────────────────────┐
           │       Streamlit Server (Python)        │
           │  ┌──────────────────────────────────┐  │
           │  │   UI Layer (app.py)              │  │
           │  │   - Form inputs                  │  │
           │  │   - Results rendering            │  │
           │  └──────────────┬───────────────────┘  │
           │                 ▼                      │
           │  ┌──────────────────────────────────┐  │
           │  │   Inference Layer                │  │
           │  │   - Preprocessing                │  │
           │  │   - Model selection              │  │
           │  │   - Prediction                   │  │
           │  └──────────────┬───────────────────┘  │
           │                 ▼                      │
           │  ┌──────────────────────────────────┐  │
           │  │   Model Artefacts (models/)      │  │
           │  │   - logreg_model.pkl             │  │
           │  │   - tfidf_vectorizer.pkl         │  │
           │  │   - lstm_model.keras             │  │
           │  │   - lstm_tokenizer.pkl           │  │
           │  └──────────────────────────────────┘  │
           └────────────────────────────────────────┘
                              ▲
                              │ (offline, one-time)
           ┌──────────────────┴─────────────────────┐
           │     Training Pipeline (train.py)       │
           │     Runs locally on developer machine  │
           └────────────────────────────────────────┘
```

### 4.2 Data Flow

1. User pastes article text into the Streamlit textarea.
2. Submit button triggers `predict_logreg()` or `predict_lstm()`.
3. Text is cleaned (lowercase, strip URLs/punctuation, remove stopwords, lemmatise).
4. Cleaned text is vectorised (TF-IDF) or tokenised + padded (LSTM).
5. The loaded model produces a probability `P(REAL)`.
6. Threshold of 0.5 converts probability to a discrete label.
7. Streamlit renders the verdict, confidence bar, and diagnostic metadata.

### 4.3 Tech Stack Rationale

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11 | Industry standard for ML; stable Streamlit support |
| UI Framework | Streamlit | Eliminates separate frontend/backend — single file, single deploy |
| Classical ML | scikit-learn | Mature, reliable, small model footprint |
| Deep Learning | TensorFlow / Keras | Well-documented, easy to save/load models |
| NLP Preprocessing | NLTK | Standard lemmatiser and stopword list |
| Deployment | Render | Free HTTPS, auto-deploy from Git, `render.yaml` as IaC |
| Version Control | Git + GitHub | Required by Render's auto-deploy |

---

## 5. Implementation

### 5.1 Dataset

**ISOT Fake News Dataset**
- Source: University of Victoria, ISOT Research Lab
- Size: ~21,400 real + ~23,500 fake articles
- Real news: Reuters (2016–2017)
- Fake news: Aggregated from fact-check flagged websites

**Preprocessing step applied to mitigate dataset leakage:**
The string `(Reuters) -` was programmatically stripped from all real articles
before training. Without this, the model learns to detect the substring instead
of textual patterns of truthfulness. This is a known artifact of the ISOT
dataset documented in the literature.

### 5.2 Preprocessing Pipeline

1. Lowercase conversion
2. URL removal (`http://...`, `www....`)
3. HTML tag removal
4. Non-alphabetic character stripping
5. Whitespace normalisation
6. Stopword removal (NLTK English stopwords)
7. Lemmatisation (WordNet lemmatiser)
8. Minimum token length filter (≥ 3 characters)

### 5.3 Model Details

**Model A — TF-IDF + Logistic Regression**
- Vectoriser: TF-IDF with unigrams and bigrams, max 20,000 features
- Classifier: L2-regularised logistic regression, `C = 1.0`
- Training time: ~30 s on a modern laptop CPU
- Artefact size: ~2 MB

**Model B — Bidirectional LSTM**
- Embedding layer: 64-dim trainable embeddings, vocab = 10,000
- Spatial dropout: 0.3
- Bi-LSTM: 48 units per direction, 0.3 dropout
- Dense: 24 units, ReLU
- Output: 1 unit, sigmoid
- Max sequence length: 300 tokens (truncated/padded)
- Training: 4 epochs, batch size 64, early stopping on validation loss
- Optimiser: Adam (default learning rate)
- Loss: Binary cross-entropy
- Training time: ~10 min on CPU, ~2 min on GPU
- Artefact size: ~15 MB

### 5.4 Key Files

| File | Purpose | Approx. lines |
|------|---------|---------------|
| `app.py` | Streamlit UI + inference logic | ~250 |
| `train.py` | End-to-end training pipeline | ~220 |
| `requirements.txt` | Pinned Python dependencies | 6 |
| `render.yaml` | Render deployment blueprint | 15 |
| `.streamlit/config.toml` | Streamlit theme configuration | 12 |

---

## 6. Testing

### 6.1 Test Strategy

Given the project's small scale and research orientation, testing focused on:

1. **Unit-level** — preprocessing idempotence, cleaner behaviour on edge cases
   (empty string, HTML, non-English characters).
2. **Model-level** — classification report, confusion matrix, ROC-AUC.
3. **Integration** — end-to-end sanity with hand-crafted REAL / FAKE samples.
4. **Deployment** — cold-start verification, memory usage under load.

### 6.2 Results Summary

| Model | Accuracy | Precision (FAKE) | Recall (FAKE) | F1 (FAKE) |
|-------|----------|------------------|---------------|-----------|
| Logistic Regression | 0.993 | 0.99 | 1.00 | 0.99 |
| Bi-LSTM | 0.995 | 1.00 | 1.00 | 1.00 |

*(Numbers will vary slightly across runs due to LSTM weight initialisation.)*

### 6.3 Manual Test Cases

| TC | Input | Expected | Result |
|----|-------|----------|--------|
| TC-1 | Empty string | Reject with error | ✅ Pass |
| TC-2 | String < 20 chars | Reject with error | ✅ Pass |
| TC-3 | Clickbait sample ("SHOCKING! Doctors HATE...") | FAKE | ✅ Pass |
| TC-4 | Reuters-style article | REAL | ✅ Pass |
| TC-5 | Gibberish / random characters | UNKNOWN or low confidence | ✅ Pass |

### 6.4 Known Failure Modes

- Neutral or well-written fake news that mimics Reuters style is often
  misclassified as REAL. This is a fundamental limitation of ISOT-trained models.
- Very short inputs (under 30 words) produce unreliable probabilities.
- The model has no knowledge cutoff awareness — an article claiming "India won
  the 2042 World Cup" will be judged on style alone.

---

## 7. Deployment

### 7.1 Deployment Target
**Render** (free web-service tier).

### 7.2 Deployment Pipeline
1. Push code and model artefacts to a GitHub repository.
2. Render reads `render.yaml` and provisions:
   - A Python 3.11 environment
   - `pip install -r requirements.txt` at build time
   - Starts Streamlit on `$PORT` bound to `0.0.0.0`
3. Render assigns a public HTTPS URL (e.g. `fake-news-detector.onrender.com`).
4. Subsequent commits to the main branch trigger auto-redeploy.

### 7.3 Operational Notes
- Free tier sleeps after 15 minutes of inactivity; first request after sleep
  takes ~30 seconds.
- 512 MB RAM ceiling required dropping the LSTM model from some deployments.
  A lightweight-mode requirements file is provided as a fallback.
- Models are bundled with the Git repository (no external model storage).

---

## 8. Maintenance

### 8.1 Retraining
The classifier should be retrained periodically because:
- Fake news patterns drift over time.
- The ISOT dataset is from 2016–2017 and its stylistic cues are dated.
- Newer datasets (LIAR-PLUS, FakeNewsNet, WELFake) should be evaluated.

### 8.2 Monitoring
For a production deployment (beyond the academic scope), recommended
instrumentation would include:
- Structured logging of predictions (without storing user text) for audit.
- Latency / error-rate metrics via Prometheus.
- Drift detection by tracking the distribution of predicted probabilities.

### 8.3 Future Work
1. Replace LSTM with a fine-tuned BERT or DistilBERT model for cross-domain
   robustness.
2. Incorporate source credibility and publication date as auxiliary features.
3. Add explainability via LIME or SHAP to show which tokens drove the decision.
4. Train on a merged dataset (ISOT + LIAR + FakeNewsNet) to reduce single-source bias.
5. Multilingual support using XLM-R.

---

## 9. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset bias produces inflated accuracy | High | High | Documented explicitly in Limitations |
| Model files exceed free-tier deploy size | Medium | Medium | Lightweight requirements variant; drop LSTM |
| NLTK download fails at runtime | Low | Medium | Cached with `@st.cache_resource`; prebuilt download |
| Render free tier discontinued | Low | High | Alternative: Streamlit Community Cloud, Hugging Face Spaces |
| User inputs adversarial / very long text | Medium | Low | Input length validation; truncation at MAX_LEN |

---

## 10. Limitations and Ethical Considerations

This project is an **academic demonstrator**, not a production fact-checker.

1. **Dataset dependence.** A model trained on one dataset reflects that
   dataset's editorial choices. Decisions by this system should never be taken
   as authoritative.
2. **No factual verification.** The model judges surface-level textual patterns
   — vocabulary, sentence structure, stylistic cues — not the factual accuracy
   of claims.
3. **Potential harms.** Deploying models like this in content-moderation
   pipelines without human review can lead to suppression of legitimate speech
   or failure to flag well-written disinformation.
4. **Transparency.** The decision threshold, model weights, and training data
   are all published as part of this project, to prevent opaque decision-making.

---

## 11. Conclusion

The system meets all stated functional and non-functional requirements on its
training distribution. Both models exceed the accuracy target of 95%, and the
application deploys successfully on a free-tier cloud provider.

However, the project's honest contribution is not its accuracy — which is
inflated by dataset artifacts — but its **demonstration of an end-to-end ML
pipeline**: from preprocessing through training, evaluation, serving, and cloud
deployment. The documented limitations are themselves a significant academic
contribution, providing a realistic assessment of what benchmark-trained
fake-news classifiers can and cannot do.

---

## Appendix A — Dependencies

See `requirements.txt`.

## Appendix B — Git Repository Structure

See `README.md` project structure section.

## Appendix C — References

1. Ahmed, H., Traore, I., & Saad, S. (2017). *Detection of online fake news
   using N-gram analysis and machine learning techniques.* ISOT.
2. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News
   Detection on Social Media: A Data Mining Perspective.*
3. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional
   Transformers for Language Understanding.*
4. Streamlit Documentation. https://docs.streamlit.io
5. Render Documentation. https://render.com/docs
