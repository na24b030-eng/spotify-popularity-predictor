import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Popularity Predictor",
    page_icon="🎵",
    layout="wide"
)

# ── Load artifacts ─────────────────────────────────────────
ARTIFACTS = Path("artifacts")

@st.cache_resource
def load_model():
    with open(ARTIFACTS / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ARTIFACTS / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(ARTIFACTS / "genres.json") as f:
        genres = json.load(f)
    with open(ARTIFACTS / "metrics.json") as f:
        metrics = json.load(f)
    return model, feature_names, genres, metrics

model, feature_names, genres, metrics = load_model()

# ── Sidebar inputs ─────────────────────────────────────────
st.sidebar.title("🎛️ Track Features")
st.sidebar.markdown("Adjust sliders to describe your track.")

genre = st.sidebar.selectbox(
    "Genre",
    genres,
    index=genres.index("pop") if "pop" in genres else 0
)

st.sidebar.markdown("---")

duration_ms      = st.sidebar.slider("Duration (seconds)", 30, 600, 210) * 1000
explicit         = st.sidebar.selectbox("Explicit?", ["No", "Yes"])
explicit_val     = 1 if explicit == "Yes" else 0
danceability     = st.sidebar.slider("Danceability",     0.0, 1.0, 0.65, 0.01)
energy           = st.sidebar.slider("Energy",           0.0, 1.0, 0.70, 0.01)
key              = st.sidebar.slider("Key",              0,   11,  5)
loudness         = st.sidebar.slider("Loudness (dB)",   -60.0, 0.0, -6.0, 0.5)
mode             = st.sidebar.selectbox("Mode", ["Minor (0)", "Major (1)"])
mode_val         = 1 if "Major" in mode else 0
speechiness      = st.sidebar.slider("Speechiness",      0.0, 1.0, 0.05, 0.01)
acousticness     = st.sidebar.slider("Acousticness",     0.0, 1.0, 0.15, 0.01)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01)
liveness         = st.sidebar.slider("Liveness",         0.0, 1.0, 0.12, 0.01)
valence          = st.sidebar.slider("Valence",          0.0, 1.0, 0.50, 0.01)
tempo            = st.sidebar.slider("Tempo (BPM)",      50.0, 220.0, 120.0, 1.0)
time_signature   = st.sidebar.slider("Time Signature",   1, 5, 4)

# ── Build input vector ─────────────────────────────────────
def build_input(feature_names, genre, values):
    row = {f: 0.0 for f in feature_names}
    for k, v in values.items():
        if k in row:
            row[k] = v
    genre_col = f"track_genre_{genre}"
    if genre_col in row:
        row[genre_col] = 1.0
    return np.array([row[f] for f in feature_names]).reshape(1, -1)

values = {
    "duration_ms":      duration_ms,
    "explicit":         explicit_val,
    "danceability":     danceability,
    "energy":           energy,
    "key":              key,
    "loudness":         loudness,
    "mode":             mode_val,
    "speechiness":      speechiness,
    "acousticness":     acousticness,
    "instrumentalness": instrumentalness,
    "liveness":         liveness,
    "valence":          valence,
    "tempo":            tempo,
    "time_signature":   time_signature,
}

X_input = build_input(feature_names, genre, values)
prediction = float(model.predict(X_input)[0])
prediction = max(0.0, min(100.0, prediction))

# ── Main page ──────────────────────────────────────────────
st.title("🎵 Spotify Popularity Predictor")
st.markdown("Predict how popular a track might be based on its audio features and genre.")

# Score
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Popularity", f"{prediction:.1f} / 100")
col2.metric("Model", "Random Forest")
col3.metric("Model R²", f"{metrics['r2']}")
col4.metric("Model MAE", f"{metrics['mae']}")

# Verdict
if prediction >= 60:
    st.success(f"🔥 High popularity predicted ({prediction:.1f}/100) — strong mainstream potential.")
elif prediction >= 35:
    st.info(f"📻 Moderate popularity predicted ({prediction:.1f}/100) — solid niche appeal.")
else:
    st.warning(f"📦 Low popularity predicted ({prediction:.1f}/100) — likely underground or niche.")

# Popularity bar
st.markdown("### Popularity Score")
st.progress(int(prediction) / 100)

st.divider()

# ── Tabs ───────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA Charts", "🔍 Feature Importance", "ℹ️ About"])

with tab1:
    st.subheader("Correlation Heatmap")
    st.image(str(ARTIFACTS / "heatmap.png"),use_container_width=True)

    st.subheader("Genre Analysis — Average Popularity")
    st.image(str(ARTIFACTS / "genre_analysis.png"), use_container_width=True)

    st.subheader("Model Comparison")
    st.image(str(ARTIFACTS / "model_comparison.png"), use_container_width=True)

with tab2:
    st.subheader("Top 20 Feature Importances — Random Forest")
    st.image(str(ARTIFACTS / "feature_importance.png"), use_container_width=True)
    st.markdown("""
    **Key finding:** Audio features (duration, loudness, acousticness, danceability)
    dominate over genre features — contradicting the initial hypothesis that genre alone drives popularity.
    Popularity depends on **nonlinear interactions** between multiple audio characteristics.
    """)

with tab3:
    st.subheader("About this project")
    st.markdown(f"""
**Dataset:** 114,000 raw Spotify tracks → 89,741 after removing duplicate track IDs

**Problem discovered:** The same song appeared under multiple genres (e.g. *Mr. Brightside*
appeared under rock, alternative, and alt-rock). This caused 24,259 duplicate rows —
a hidden data leakage risk that was identified and resolved before modelling.

**Models compared:**

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (predict mean) | 0.00 | 17.12 | 20.44 |
| Linear Regression | 0.32 | 12.01 | 16.84 |
| Ridge Regression | 0.32 | 12.02 | 16.85 |
| **Random Forest** | **0.47** | **10.03** | **14.89** |

**Why Random Forest won:** Popularity depends on nonlinear interactions between features —
something linear models cannot capture. The R² jump from 0.32 → 0.47 confirms this.

**Limitation:** The model explains 47% of popularity variance.
The remaining 53% is driven by artist fame, playlist placement, marketing, and virality —
none of which are in the dataset. This is an honest limitation, not a modelling failure.

**Stack:** Python · Pandas · Scikit-learn · Random Forest · Streamlit
    """)