import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def save(name, obj):
    with open(os.path.join(ARTIFACTS_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  saved {name}.pkl")


def main():
    csv_path = os.path.join(DATA_DIR, "tmdb_movies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run: python fetch_tmdb.py --api-key YOUR_KEY"
        )

    print("Loading tmdb_movies.csv...")
    df = pd.read_csv(csv_path, dtype={"year": str, "poster_path": str})
    df["poster_path"] = df["poster_path"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["keywords"] = df["keywords"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["director"] = df["director"].fillna("")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset="movie_id").reset_index(drop=True)
    print(f"  {len(df):,} movies")

    print("Building TF-IDF corpus...")
    corpus = (
        df["title"] + " " +
        df["overview"] + " " +
        df["genres"].str.replace("|", " ", regex=False) + " " +
        df["keywords"] + " " +
        df["cast"] + " " +
        df["director"]
    )

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    content_mat = tfidf.fit_transform(corpus)
    print(f"  vocab size: {len(tfidf.vocabulary_):,}  matrix: {content_mat.shape}")

    content_index = {int(mid): i for i, mid in enumerate(df["movie_id"])}

    popularity_score = np.log1p(df["vote_count"].values) * df["vote_average"].values

    genre_index = {}
    for _, row in df.iterrows():
        for g in str(row["genres"]).split("|"):
            g = g.strip()
            if g:
                genre_index.setdefault(g, []).append(int(row["movie_id"]))

    movie_lookup = {
        int(row["movie_id"]): {
            "title": row["title"],
            "year": row["year"],
            "genres": [g for g in str(row["genres"]).split("|") if g.strip()],
            "overview": row["overview"],
            "cast": row["cast"],
            "director": row["director"],
            "poster_path": row["poster_path"],
            "vote_average": float(row["vote_average"]),
            "vote_count": int(row["vote_count"]),
            "popularity": float(row["popularity"]),
        }
        for _, row in df.iterrows()
    }

    genre_list_path = os.path.join(DATA_DIR, "genre_list.json")
    if os.path.exists(genre_list_path):
        with open(genre_list_path) as f:
            genre_list = json.load(f)
    else:
        genre_list = [{"id": 0, "name": g} for g in sorted(genre_index.keys())]

    print("Saving artifacts...")
    save("tfidf", tfidf)
    save("content_mat", content_mat)
    save("content_index", content_index)
    save("popularity_score", popularity_score)
    save("genre_index", genre_index)
    save("movie_lookup", movie_lookup)
    save("genre_list", genre_list)
    save("movie_ids", df["movie_id"].tolist())

    print(f"\nDone. {len(df):,} movies · vocab {len(tfidf.vocabulary_):,} · matrix {content_mat.shape}")


if __name__ == "__main__":
    main()
