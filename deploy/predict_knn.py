#!/usr/bin/env python
"""
Uso:
python deploy/predict_knn.py --user 42 --top 10
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import argparse
from src.reco_utils import (
    load_knn_model,
    load_movies,
    load_ratings,
    get_top_n,
)

parser = argparse.ArgumentParser()
parser.add_argument("--user", type=int, required=True)
parser.add_argument("--top", type=int, default=10)
args = parser.parse_args()

# carregar artefatos
model   = load_knn_model()
movies  = load_movies()
ratings = load_ratings()

# gerar recomendações
top_recs = get_top_n(model, args.user, ratings, movies, n=args.top)

print(f"\nTop {args.top} recomendações para usuário {args.user}:")
for title, score in top_recs:
    print(f"{title:<50} {score:.2f}★")
