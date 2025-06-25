import pandas as pd, pathlib as pl

RAW_DIR = pl.Path(__file__).parents[1] / "data" / "raw"
PROC_DIR = pl.Path(__file__).parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

def build_movies_clean(min_ratings=100):
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")

    # filtro por nº de avaliações
    counts = ratings.groupby("movieId").size()
    popular_ids = counts[counts >= min_ratings].index
    ratings = ratings[ratings.movieId.isin(popular_ids)]
    movies = movies[movies.movieId.isin(popular_ids)]

    # one-hot dos gêneros
    movies["genres_list"] = movies.genres.str.split("|")
    dummies = (
        movies["genres_list"].explode()
        .str.get_dummies()
        .groupby(level=0)
        .sum()
    )
    movies_clean = pd.concat([movies.drop(columns=["genres"]), dummies], axis=1)
    movies_clean.to_csv(PROC_DIR / "movies_clean.csv", index=False)
    ratings.to_csv(PROC_DIR / "ratings_clean.csv", index=False)
    return movies_clean, ratings
if __name__ == "__main__":
    build_movies_clean()
