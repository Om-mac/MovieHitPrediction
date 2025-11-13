# MasterMovieHit_Pipeline.py
# End-to-end master-level pipeline for predicting Hit vs Flop (Bollywood + Hollywood)
# Run as a Jupyter notebook (split by comments) or as a script. Replace paths where indicated.

################################################################################
# 0. Setup & Instructions
################################################################################
# Requirements:
# - Python 3.9+
# - Install: pip install pandas numpy scikit-learn xgboost lightgbm transformers torch sentence-transformers
#              pip install rapidfuzz shap matplotlib seaborn joblib optuna pyarrow
# - Datasets you must download and place in the ./data folder:
#   1) tmdb_5000_movies.csv
#   2) tmdb_5000_credits.csv
#   3) IMDB 50K reviews CSV (imdb_50k.csv) or original imdb labeled file
#   4) (Optional) Additional box office datasets for Bollywood/Hollywood
#
# How to run:
# - Open this file in JupyterLab/Notebook and run cells sequentially
# - Or run as script: python MasterMovieHit_Pipeline.py (make sure to guard long-running parts)

################################################################################
# 1. Imports
################################################################################
import os
import ast
import json
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# NLP
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Explainability
import shap

# Warnings
import warnings
warnings.filterwarnings('ignore')

################################################################################
# 2. Utilities
################################################################################

def normalize_title(t):
    if pd.isna(t):
        return ""
    s = ''.join(ch for ch in str(t).lower() if ch.isalnum() or ch.isspace())
    return ' '.join(s.split())


def safe_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

################################################################################
# 3. Load datasets (user must download into ./data)
################################################################################
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Replace filenames if different
TMDB_MOVIES = os.path.join(DATA_DIR, 'tmdb_5000_movies.csv')
TMDB_CREDITS = os.path.join(DATA_DIR, 'tmdb_5000_credits.csv')
IMDB_50K = os.path.join(DATA_DIR, 'IMDB Dataset.csv')  # your IMDB 50k reviews file

# load with safe checks
print('Loading datasets...')
if not os.path.exists(TMDB_MOVIES) or not os.path.exists(TMDB_CREDITS):
    raise FileNotFoundError('Place tmdb_5000_movies.csv and tmdb_5000_credits.csv into ./data')

movies = pd.read_csv(TMDB_MOVIES)
credits = pd.read_csv(TMDB_CREDITS)

print('TMDB loaded:', movies.shape, credits.shape)

# Optional: load IMDB reviews for sentiment model training
if os.path.exists(IMDB_50K):
    reviews = pd.read_csv(IMDB_50K)
    print('IMDB reviews loaded:', reviews.shape)
else:
    reviews = None
    print('IMDB reviews not found. Sentiment module will use pretrained HF model for inference.')

################################################################################
# 4. Basic merging & cleaning
################################################################################
# Merge credits into movies
movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')

# -------- FIX DUPLICATE TITLE COLUMNS --------
if 'title_x' in movies.columns:
    movies.rename(columns={'title_x': 'title'}, inplace=True)
if 'title_y' in movies.columns:
    movies.drop(columns=['title_y'], inplace=True)
    

# Normalize titles and release years
movies['title_norm'] = movies['title'].apply(normalize_title)
movies['release_date'] = pd.to_datetime(movies['release_date'], errors='coerce')
movies['release_year'] = movies['release_date'].dt.year.fillna(0).astype(int)

# Numeric cleanup
movies['budget'] = pd.to_numeric(movies['budget'], errors='coerce').fillna(0)
movies['revenue'] = pd.to_numeric(movies['revenue'], errors='coerce').fillna(0)
movies['popularity'] = pd.to_numeric(movies['popularity'], errors='coerce').fillna(0)
movies['runtime'] = pd.to_numeric(movies['runtime'], errors='coerce').fillna(movies['runtime'].median())

# Filter out entries with missing key info
movies = movies[movies['budget'] > 0]

print('After cleaning:', movies.shape)

################################################################################
# 5. Create HIT / FLOP label
################################################################################
# Option: ROI threshold (1.5x)
movies['label'] = (movies['revenue'] >= movies['budget'] * 1.5).astype(int)

# Quick distribution
print('Label distribution (0=Flop,1=Hit):')
print(movies['label'].value_counts(normalize=True))

################################################################################
# 6. Feature engineering (genres, cast, director, textual features)
################################################################################
# --- genres multi-hot
movies['genres_list'] = movies['genres'].apply(lambda g: [d['name'] for d in safe_literal_eval(g)] if pd.notna(g) else [])
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_ohe = pd.DataFrame(mlb.fit_transform(movies['genres_list']), columns=[f'genre_{c}' for c in mlb.classes_], index=movies.index)
movies = pd.concat([movies, genres_ohe], axis=1)

# --- cast: compute top-3 cast names and placeholder for actor success score
movies['cast_list'] = movies['cast'].apply(lambda c: [p.get('name') for p in safe_literal_eval(c)[:5]] if pd.notna(c) else [])
movies['top_cast_1'] = movies['cast_list'].apply(lambda l: l[0] if len(l)>0 else '')
movies['top_cast_2'] = movies['cast_list'].apply(lambda l: l[1] if len(l)>1 else '')
movies['top_cast_3'] = movies['cast_list'].apply(lambda l: l[2] if len(l)>2 else '')

# Compute simple actor_success_score: mean revenue of movies each actor appeared in (global)
# explode cast_list
actor_df = movies[['id','title','revenue','cast_list']].explode('cast_list')
actor_df = actor_df[actor_df['cast_list'].notna()]
actor_success = actor_df.groupby('cast_list')['revenue'].mean().rename('actor_success_mean')

# map back to movies using top cast mean
movies = movies.join(actor_success, on='top_cast_1')
movies.rename(columns={'actor_success_mean':'top_cast_1_success'}, inplace=True)
movies = movies.join(actor_success, on='top_cast_2')
movies.rename(columns={'actor_success_mean':'top_cast_2_success'}, inplace=True)
movies = movies.join(actor_success, on='top_cast_3')
movies.rename(columns={'actor_success_mean':'top_cast_3_success'}, inplace=True)

# aggregate cast_score
movies['cast_success_score'] = movies[['top_cast_1_success','top_cast_2_success','top_cast_3_success']].mean(axis=1).fillna(0)

# --- director score
movies['director_list'] = movies['crew'].apply(lambda c: [p for p in safe_literal_eval(c) if p.get('job')=='Director'] if pd.notna(c) else [])
movies['director_name'] = movies['director_list'].apply(lambda l: l[0].get('name') if len(l)>0 else '')

director_df = movies[['id','title','revenue','director_name']]
director_success = director_df.groupby('director_name')['revenue'].mean().rename('director_success_mean')
movies = movies.join(director_success, on='director_name')
movies.rename(columns={'director_success_mean':'director_success_score'}, inplace=True)
movies['director_success_score'] = movies['director_success_score'].fillna(0)

# --- production company score
movies['prod_companies'] = movies['production_companies'].apply(lambda g: [d['name'] for d in safe_literal_eval(g)] if pd.notna(g) else [])
pc = movies[['id','title','revenue','prod_companies']].explode('prod_companies')
company_success = pc.groupby('prod_companies')['revenue'].mean().rename('prod_company_success')
# map using first company
movies['primary_prod'] = movies['prod_companies'].apply(lambda l: l[0] if len(l)>0 else '')
movies = movies.join(company_success, on='primary_prod')
movies.rename(columns={'prod_company_success':'prod_company_success'}, inplace=True)
movies['prod_company_success'] = movies['prod_company_success'].fillna(0)

# --- time features
movies['release_month'] = movies['release_date'].dt.month.fillna(0).astype(int)
movies['release_quarter'] = movies['release_date'].dt.quarter.fillna(0).astype(int)

# --- rating features
movies['vote_average'] = pd.to_numeric(movies['vote_average'], errors='coerce').fillna(movies['vote_average'].median())
movies['vote_count'] = pd.to_numeric(movies['vote_count'], errors='coerce').fillna(0)

################################################################################
# 7. Sentiment scoring using pretrained HF sentiment model (fast path)
################################################################################
# If you have IMDB 50K and want to fine-tune, separate notebook cell is provided later.
# For now, apply a pretrained sentiment model to a subset of reviews or to review texts if available.

SENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'  # good off-the-shelf option (3-class)
print('Loading sentiment model:', SENT_MODEL)

tokenizer = AutoTokenizer.from_pretrained(SENT_MODEL)
sent_model = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL)
sent_model.eval()

# If reviews dataset exists with mapping to movie titles, you can compute average sentiment per movie.
# We'll provide a function to compute sentiment for a list of texts (batched)

def batch_sentiment(texts, batch_size=16):
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
        with torch.no_grad():
            logits = sent_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # positive prob is last index in this model (neg, neu, pos)
        pos_probs = probs[:, -1]
        scores.extend(pos_probs.tolist())
    return scores

# If you have a reviews dataframe with columns ['movie_title','review_text'] map titles and aggregate.
if reviews is not None:
    # Normalize title in reviews
    if 'review' in reviews.columns and 'sentiment' in reviews.columns:
        # IMDB 50k original columns are 'review' and 'sentiment' (pos/neg)
        # If you have movie mapping, use it. If not, we compute sentiment model on reviews and then show example.
        print('IMDB 50K present; you can fine-tune sentiment model. Skipping heavy aggregation by default.')
    else:
        print('Reviews present but missing expected columns. You may adapt mapping to movie titles.')
else:
    print('No local reviews dataset. The pipeline will run using pretrained sentiment model only for example texts.')

# Place a placeholder sentiment score: use vote_average normalized as a proxy if no per-movie reviews
movies['sentiment_score'] = movies['vote_average'] / movies['vote_average'].max()

################################################################################
# 8. Hype score placeholder (social features require external APIs)
################################################################################
# We provide a simple hype proxy: popularity normalized + vote_count normalized + prod_company_success zscore
movies['pop_norm'] = (movies['popularity'] - movies['popularity'].mean()) / (movies['popularity'].std()+1e-9)
movies['vote_count_norm'] = (movies['vote_count'] - movies['vote_count'].mean()) / (movies['vote_count'].std()+1e-9)
movies['prod_company_norm'] = (movies['prod_company_success'] - movies['prod_company_success'].mean())/(movies['prod_company_success'].std()+1e-9)
movies['hype_score'] = (movies['pop_norm'] + movies['vote_count_norm'] + movies['prod_company_norm'])/3.0

################################################################################
# 9. Final feature set & train/test split (time-based)
################################################################################
FEATURES = [
    'budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
    'cast_success_score', 'director_success_score', 'prod_company_success', 'sentiment_score', 'hype_score',
    'release_month'
]
# add genres
for c in movies.columns:
    if c.startswith('genre_'):
        FEATURES.append(c)

# Ensure no missing
X = movies[FEATURES].fillna(0)
y = movies['label']

# Time-based split: train before 2016, test 2016+
train_df = movies[movies['release_year'] < 2016]
test_df = movies[movies['release_year'] >= 2016]

if train_df.shape[0] < 50:
    # fallback to random split if not enough history
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df['label']
    X_test = test_df[FEATURES].fillna(0)
    y_test = test_df['label']

print('Train/Test sizes:', X_train.shape, X_test.shape)

################################################################################
# 10. Baseline models: RandomForest & XGBoost
################################################################################
print('Training RandomForest...')
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:,1]
print('RF Accuracy:', accuracy_score(y_test, rf_pred))
print('RF ROC AUC:', roc_auc_score(y_test, rf_proba))

print('\nTraining XGBoost...')
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    eval_metric='auc'
)
xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

print('XGB Accuracy:', accuracy_score(y_test, xgb_pred))
print('XGB ROC AUC:', roc_auc_score(y_test, xgb_proba))

################################################################################
# 11. Proper stacking using out-of-fold predictions (stack RF + XGB)
################################################################################
print('Starting OOF stacking...')
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

train_meta = np.zeros((X_train.shape[0], 2))
test_meta = np.zeros((X_test.shape[0], 2))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print('Fold', fold+1)
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # First base model: Random Forest
    m1 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    m1.fit(X_tr, y_tr)
    val_pred1 = m1.predict_proba(X_val)[:, 1]
    train_meta[val_idx, 0] = val_pred1

    # accumulate test set predictions (average over folds)
    test_meta[:, 0] += m1.predict_proba(X_test)[:, 1] / N_FOLDS

    # Second base model: XGBoost (no early stopping in fit for XGBoost 2.x)
    m2 = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        eval_metric='auc'
    )
    m2.fit(X_tr, y_tr)  # correct: simple fit call
    val_pred2 = m2.predict_proba(X_val)[:, 1]
    train_meta[val_idx, 1] = val_pred2
    test_meta[:, 1] += m2.predict_proba(X_test)[:, 1] / N_FOLDS

# Train meta learner on OOF predictions
meta_clf = LogisticRegression()
meta_clf.fit(train_meta, y_train)
meta_proba = meta_clf.predict_proba(test_meta)[:, 1]
meta_pred = (meta_proba >= 0.5).astype(int)

print('Stacked Model Accuracy:', accuracy_score(y_test, meta_pred))
print('Stacked Model ROC AUC:', roc_auc_score(y_test, meta_proba))
print(classification_report(y_test, meta_pred))

################################################################################
# 12. Explainability with SHAP (on XGB model)
################################################################################
# print('Computing SHAP values (may take time)...')
# explainer = shap.TreeExplainer(xgb_model)
# shap_values = explainer.shap_values(X_test)
# # summary plot (in notebook this will display)
# try:
#     shap.summary_plot(shap_values, X_test, show=False)
#     print('SHAP summary_plot generated. (In notebook, view the figure)')
# except Exception as e:
#     print('SHAP plotting error (display not available):', e)

print("SHAP skipped because XGBoost 2.x is incompatible.")

################################################################################
# 13. Save artifacts
################################################################################
ARTIFACTS_DIR = './artifacts'
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)

joblib.dump(xgb_model, os.path.join(ARTIFACTS_DIR, 'xgb_model.joblib'))
joblib.dump(rf, os.path.join(ARTIFACTS_DIR, 'rf_model.joblib'))
joblib.dump(meta_clf, os.path.join(ARTIFACTS_DIR, 'meta_clf.joblib'))
movies.to_csv(os.path.join(ARTIFACTS_DIR, 'movies_featured.csv'), index=False)
print('Saved models and featured dataset to ./artifacts')

################################################################################
# 14. (Optional) Fine-tune sentiment model on IMDB 50K (outline)
################################################################################
# If you want to fine-tune a BERT-style model on the IMDB 50K labelled dataset, follow this outline.
# This cell is intentionally non-executable in automatic scripts because it requires GPU/time.

# 1) Prepare dataset: reviews with labels (pos/neg -> 1/0)
# 2) Use transformers Trainer API to fine-tune (tokenizer, model, training arguments)
# 3) Save fine-tuned model and use batch_sentiment() to compute per-review probs then aggregate per-movie

# Example skeleton (not executed automatically):
# from datasets import Dataset
# reviews_small = reviews[['review','sentiment']].copy()
# reviews_small['label'] = reviews_small['sentiment'].map({'positive':1,'negative':0})
# ds = Dataset.from_pandas(reviews_small)
# tokenized = ds.map(lambda x: tokenizer(x['review'], truncation=True, padding='max_length', max_length=256), batched=True)
# ... set up Trainer and train ...

################################################################################
# 15. Next steps & notes
################################################################################
# - Replace sentiment_score placeholder with true aggregated model predictions if you fine-tune BERT
# - Add social media and trailer features for better performance
# - Adjust label definition (e.g., inflation-adjusted revenue or percentiles)
# - Use Optuna for hyperparameter tuning
# - Use proper OOF stacking method for meta learner using full training set

print('\nPipeline complete. Check ./artifacts for outputs.\n')
