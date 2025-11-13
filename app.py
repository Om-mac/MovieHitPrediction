# app.py — Streamlit app for Movie Hit Predictor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Paths (assumes you run from MovieHitPrediction/)
ARTIFACTS = './artifacts'
FEATURED = os.path.join(ARTIFACTS, 'movies_featured.csv')
XGB_MODEL = os.path.join(ARTIFACTS, 'xgb_model.joblib')
RF_MODEL = os.path.join(ARTIFACTS, 'rf_model.joblib')
META_MODEL = os.path.join(ARTIFACTS, 'meta_clf.joblib')

st.set_page_config(page_title="Movie Hit Predictor", layout="centered")
st.title("🎬 Movie Hit Predictor")
st.markdown("Enter movie metadata and get a **Hit / Flop** probability using your trained ensemble model.")

# --- Load models & featured dataset
@st.cache_resource
def load_artifacts():
    models = {
        'xgb': joblib.load(XGB_MODEL),
        'rf': joblib.load(RF_MODEL),
        'meta': joblib.load(META_MODEL)
    }
    df = pd.read_csv(FEATURED)
    return models, df

models, df = load_artifacts()

# -----------------------------------------------
# Precompute helper maps
# -----------------------------------------------

# Cast success mapping
if 'cast_list' in df.columns:
    try:
        df['cast_list'] = df['cast_list'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    except:
        pass
    actor_rows = df[['id', 'cast_list', 'revenue']].explode('cast_list')
    actor_rows = actor_rows[actor_rows['cast_list'].notna()]
    actor_success = actor_rows.groupby('cast_list')['revenue'].mean().to_dict()
else:
    actor_success = {}

# Director success
if 'director_name' in df.columns and 'director_success_score' in df.columns:
    director_success = df.groupby('director_name')['director_success_score'].first().to_dict()
else:
    director_success = {}

# dataset stats for normalization
pop_mean, pop_std = df['popularity'].mean(), df['popularity'].std()
vote_mean, vote_std = df['vote_count'].mean(), df['vote_count'].std()
prod_mean, prod_std = df['prod_company_success'].mean(), df['prod_company_success'].std()
max_voteavg = df['vote_average'].max()

# -----------------------------------------------
# User inputs
# -----------------------------------------------
st.sidebar.header("Movie Input")

budget = st.sidebar.number_input("Budget (USD)", value=5_000_000, step=100000, format="%d")
popularity = st.sidebar.number_input("TMDB Popularity", value=10.0, step=0.1)
runtime = st.sidebar.number_input("Runtime (minutes)", value=120, step=1)
vote_average = st.sidebar.number_input("Average Vote", value=6.5, step=0.1)
vote_count = st.sidebar.number_input("Vote Count", value=1000, step=1)

st.sidebar.markdown("---")
cast_1 = st.sidebar.text_input("Top Cast 1")
cast_2 = st.sidebar.text_input("Top Cast 2")
cast_3 = st.sidebar.text_input("Top Cast 3")
director = st.sidebar.text_input("Director Name")
prod_company = st.sidebar.text_input("Production Company")

# ---------------------------------------------------
# Predict
# ---------------------------------------------------
if st.sidebar.button("Predict"):

    # --------------------------------------------------------
    # Load EXACT features used during TRAINING
    # --------------------------------------------------------
    base_features = [
        'budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
        'cast_success_score', 'director_success_score', 'prod_company_success',
        'sentiment_score', 'hype_score',
        'release_month'
    ]
    genre_features = [c for c in df.columns if c.startswith("genre_")]
    df_features = base_features + genre_features

    # --------------------------------------------------------
    # Cast success
    # --------------------------------------------------------
    cast_scores = []
    for name in [cast_1, cast_2, cast_3]:
        name = name.strip()
        if not name:
            continue
        if name in actor_success:
            cast_scores.append(actor_success[name])
        else:
            partial = [v for k, v in actor_success.items() if name.lower() in k.lower()]
            if partial:
                cast_scores.append(np.mean(partial))
    cast_success_score = np.mean(cast_scores) if cast_scores else 0.0

    # Director success
    director_success_score = director_success.get(director, 0.0)

    # Production company success (safe fallback)
    if 'prod_company_success' in df.columns:
        prod_company_success = df['prod_company_success'].mean()
    else:
        prod_company_success = 0.0

    # Sentiment score proxy
    sentiment_score = vote_average / max_voteavg

    # Hype score
    pop_norm = (popularity - pop_mean) / (pop_std + 1e-9)
    vote_norm = (vote_count - vote_mean) / (vote_std + 1e-9)
    prod_norm = (prod_company_success - prod_mean) / (prod_std + 1e-9)
    hype_score = (pop_norm + vote_norm + prod_norm) / 3.0

    # --------------------------------------------------------
    # Build final feature vector
    # --------------------------------------------------------
    X_row = {}

    for col in df_features:
        if col.startswith("genre_"):
            X_row[col] = 0
        elif col == 'budget':
            X_row[col] = budget
        elif col == 'popularity':
            X_row[col] = popularity
        elif col == 'runtime':
            X_row[col] = runtime
        elif col == 'vote_average':
            X_row[col] = vote_average
        elif col == 'vote_count':
            X_row[col] = vote_count
        elif col == 'cast_success_score':
            X_row[col] = cast_success_score
        elif col == 'director_success_score':
            X_row[col] = director_success_score
        elif col == 'prod_company_success':
            X_row[col] = prod_company_success
        elif col == 'sentiment_score':
            X_row[col] = sentiment_score
        elif col == 'hype_score':
            X_row[col] = hype_score
        elif col == 'release_month':
            X_row[col] = 1
        else:
            X_row[col] = 0

    X_df = pd.DataFrame([X_row], columns=df_features)

    # --------------------------------------------------------
    # Predict using RF, XGB, META (stacking)
    # --------------------------------------------------------
    rf = models['rf']
    xgb = models['xgb']
    meta = models['meta']

    rf_p = rf.predict_proba(X_df)[:, 1]
    xgb_p = xgb.predict_proba(X_df)[:, 1]
    meta_input = np.vstack([rf_p, xgb_p]).T

    meta_prob = meta.predict_proba(meta_input)[:, 1][0]
    label = "HIT" if meta_prob >= 0.5 else "FLOP"

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------
    st.subheader("🎯 Prediction Result")
    st.markdown(f"### 🔥 Hit Probability: **{meta_prob:.3f}**")
    st.markdown(f"### 🎬 Final Verdict: **{label}**")

    st.write("RandomForest p(Hit):", round(rf_p[0], 3))
    st.write("XGBoost p(Hit):", round(xgb_p[0], 3))

    st.subheader("Features Used")
    st.dataframe(X_df.T)

# Footer
st.markdown("---")
st.markdown("Extend this app with trailer sentiment, Twitter hype, or BERT-based reviews!")