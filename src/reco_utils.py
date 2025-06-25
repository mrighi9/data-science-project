# src/reco_utils.py
"""
Funções utilitárias para os sistemas de recomendação
----------------------------------------------------
- get_top_n(model, user_id, ratings_df, movies_df, n=10)
- recommend_similar(movie_title, n=10, movies_df, sim_matrix)
"""
from surprise.dump import load
import pandas as pd
import numpy as np

DATA_DIR   = "data/processed"
MODEL_DIR  = "models"

# ---------- loader genérico ----------
def load_knn_model(path=f"{MODEL_DIR}/knn_item.pkl"):
    _preds, algo = load(path)          # retorna modelo Surprise
    return algo

def load_movies():
    return pd.read_csv(f"{DATA_DIR}/movies_clean.csv")

def load_ratings():
    return pd.read_csv(f"{DATA_DIR}/ratings_clean.csv")

# ---------- 1. Top-N para um usuário ----------
def get_top_n(model, user_id, ratings_df, movies_df, n=10):
    seen = ratings_df.loc[ratings_df.userId == user_id, "movieId"]
    unseen = movies_df.loc[~movies_df.movieId.isin(seen), "movieId"]

    preds = [model.predict(user_id, mid).est for mid in unseen]
    top_idx = np.argsort(preds)[-n:][::-1]
    top_movies = movies_df.loc[unseen.iloc[top_idx].index]
    top_scores = np.array(preds)[top_idx]

    return list(zip(top_movies.title.values, top_scores))

# ---------- 2. Conteúdo: similares ----------
def recommend_similar(movie_title, n, movies_df, sim_matrix):
    idx_list = movies_df.index[movies_df.title == movie_title].tolist()
    if not idx_list:
        raise ValueError(f"'{movie_title}' não encontrado.")
    idx = idx_list[0]

    sims = sim_matrix[idx].toarray().ravel()
    similar_idx = np.argsort(sims)[::-1][1 : n + 1]
    return movies_df.iloc[similar_idx]["title"].tolist()
