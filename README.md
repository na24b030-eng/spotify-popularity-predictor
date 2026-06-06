# 🎵 Spotify Track Popularity Predictor

> A machine learning web application that predicts the popularity score of a Spotify track (0–100) based on its audio features and genre — built as part of a 5-project ML portfolio.

**🔗 Live Demo:** [https://spotify-popularity-predictor.streamlit.app/](https://spotify-popularity-predictor.streamlit.app/)
**📓 Kaggle Notebook:** [https://www.kaggle.com/code/radiantbright/spotify-popularity-predictor](https://www.kaggle.com/code/radiantbright/spotify-popularity-predictor)

---

## 📌 What is this project?

Spotify assigns every track a popularity score from 0 to 100. This score is updated regularly and reflects recent stream counts and user engagement. The question this project asks is:

> **Can we predict how popular a song will be, just from its audio characteristics?**

This project builds a complete machine learning pipeline — from raw data to a deployed web application — to answer that question honestly. The answer turns out to be: *partially yes, but with important caveats that reveal something interesting about how popularity actually works.*

---

## 🔍 The Hidden Data Problem

The first and most important finding came before any modelling.

The raw dataset contains **114,000 rows** across 114 genres. A standard duplicate check returned zero duplicates:

```python
df.duplicated().sum()  # → 0
```

This looked clean. But checking by track identity told a different story:

```python
df["track_id"].duplicated().sum()  # → 24,259
```

**24,259 duplicate tracks.** The same song appeared multiple times under different genre labels.

| Track | Appearances |
|---|---|
| Mr. Brightside | rock, alternative, alt-rock |
| Comedy (Gen Hoshino) | acoustic, j-pop, singer-songwriter, songwriter |

This is how Spotify structures its genre taxonomy — one track can belong to multiple genres. But for machine learning, this creates a serious problem: the same song can appear in both the training set and the test set simultaneously, making the model appear more accurate than it actually is. This is called **data leakage**.

**Fix:** Keep only the first occurrence of each `track_id` before any modelling.

```python
df = df.drop_duplicates(subset="track_id", keep="first")
```

**Result:** 114,000 → **89,741 unique tracks**

---

## 📊 Dataset Overview

| Property | Value |
|---|---|
| Source | Spotify Tracks Dataset (Kaggle) |
| Raw rows | 114,000 |
| After deduplication | 89,741 |
| Genres | 114 |
| Audio features | 13 |
| Engineered features (total) | 126 |
| Target variable | popularity (0–100) |
| Train / Test split | 80% / 20% |

### Audio Features Used

| Feature | Description |
|---|---|
| `danceability` | How suitable the track is for dancing (0–1) |
| `energy` | Perceptual measure of intensity and activity (0–1) |
| `loudness` | Overall loudness in decibels (−60 to 0) |
| `speechiness` | Presence of spoken words (0–1) |
| `acousticness` | Confidence the track is acoustic (0–1) |
| `instrumentalness` | Likelihood the track has no vocals (0–1) |
| `liveness` | Presence of a live audience (0–1) |
| `valence` | Musical positiveness — sad vs happy (0–1) |
| `tempo` | Estimated tempo in BPM |
| `duration_ms` | Track length in milliseconds |
| `key` | Musical key (0–11) |
| `mode` | Major (1) or Minor (0) |
| `time_signature` | Beats per bar |

Genre was one-hot encoded into **113 binary columns** (one per genre, drop_first=True), giving 126 total features.

---

## 📈 Exploratory Data Analysis

### Popularity Distribution

Popularity is not uniformly distributed. Most tracks cluster in the 15–50 range, with very few reaching above 80. This long tail reflects the reality of music: a small number of tracks dominate streaming while the vast majority remain niche or unknown.

### Correlation Analysis

Audio features showed very weak linear correlations with popularity:

| Feature | Correlation with Popularity |
|---|---|
| loudness | +0.07 |
| danceability | +0.06 |
| explicit | +0.05 |
| instrumentalness | −0.13 |

**No single audio feature explains popularity.** This was the first signal that linear models would struggle and that nonlinear approaches might do better.

### Genre Analysis

Average popularity varied dramatically across genres:

**Most popular genres:** k-pop (~59), pop-film (~59), metal (~56), chill (~54), latino (~52)

**Least popular genres:** iranian (~2), romance (~10), jazz (~10), detroit-techno (~11)

This confirmed that genre carries real predictive signal and should be included as a feature.

---

## 🤖 Modelling

Four models were trained and compared on identical train/test splits.

### Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (predict mean always) | 0.00 | 17.12 | 20.44 |
| Linear Regression | 0.32 | 12.01 | 16.84 |
| Ridge Regression | 0.32 | 12.02 | 16.85 |
| **Random Forest** | **0.47** | **10.03** | **14.89** |

### Why Random Forest Won

Linear Regression and Ridge produced nearly identical results (R²=0.32), which tells us two things:

1. Multicollinearity between features was not a major problem (Ridge adding regularisation changed almost nothing)
2. The relationship between audio features and popularity is **not linear**

Random Forest captures nonlinear interactions — for example, the combination of high danceability + high loudness + k-pop genre matters more than any one of those features alone. The jump from R²=0.32 to R²=0.47 is the quantified evidence of this.

---

## 🔎 Feature Importance

Random Forest revealed which features actually drive predictions:

| Rank | Feature | Importance |
|---|---|---|
| 1 | duration_ms | 0.064 |
| 2 | loudness | 0.060 |
| 3 | acousticness | 0.059 |
| 4 | danceability | 0.058 |
| 5 | valence | 0.057 |
| 6 | tempo | 0.056 |
| 7 | speechiness | 0.056 |
| 8 | liveness | 0.054 |
| 9 | energy | 0.054 |
| 10 | instrumentalness | 0.042 |

**Surprising finding:** Audio features dominated over genre dummy features. The initial hypothesis was that genre would be the primary driver. Instead, audio characteristics collectively proved more predictive, with genre features appearing further down the importance ranking (track_genre_iranian, track_genre_romance, track_genre_k-pop were the top genre dummies).

This contradicts the naive assumption that genre alone determines popularity. The *way* a track sounds matters as much as what genre it belongs to.

---

## ⚠️ Honest Limitations

The model explains **47% of popularity variance**. This is a meaningful result but not a complete one. The remaining 53% is driven by factors not present in this dataset:

- **Artist fame** — a song by a globally known artist starts with a built-in audience
- **Playlist placement** — editorial playlists on Spotify can multiply streams overnight
- **Release timing** — dropping a track during a cultural moment matters
- **Social media virality** — TikTok trends have launched obscure songs to #1
- **Marketing spend** — label promotional budgets are invisible in audio features

This is not a modelling failure. It is an honest acknowledgement that audio characteristics are one input into popularity, not the whole story. The model predicts **audio-based popularity potential** — what the song's sound suggests about its commercial appeal — not guaranteed chart performance.

---

## 🖥️ Web Application

The Streamlit app allows anyone to:

- Select a genre from all 114 Spotify genre categories
- Adjust 13 audio feature sliders to describe a hypothetical track
- Receive an instant predicted popularity score (0–100)
- See a colour-coded verdict (high / moderate / low potential)
- Explore EDA charts, genre analysis, model comparison, and feature importance

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core language |
| Pandas / NumPy | Data processing and manipulation |
| Scikit-learn | Model training (Random Forest, Linear, Ridge) |
| Matplotlib / Seaborn | Visualisation |
| Streamlit | Web application framework |
| Kaggle Notebooks | Training environment |
| Git LFS | Large file storage for model.pkl |

---

## 📁 Project Structure

```
spotify-popularity-predictor/
├── app.py                      # Streamlit application
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── artifacts/
    ├── model.pkl               # Trained Random Forest model
    ├── feature_names.json      # Feature column order
    ├── genres.json             # All 114 genres
    ├── metrics.json            # Model performance metrics
    ├── heatmap.png             # Correlation heatmap
    ├── genre_analysis.png      # Genre popularity analysis
    ├── model_comparison.png    # Model comparison chart
    └── feature_importance.png  # Feature importance chart
```