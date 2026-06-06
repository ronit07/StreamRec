import os
import pickle
import re

import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
POSTER_BASE = "https://image.tmdb.org/t/p/w500"

app = Flask(__name__)
_cache = {}


def _load(name):
    if name not in _cache:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact '{name}' not found. Run pipeline.py first.")
        with open(path, "rb") as f:
            _cache[name] = pickle.load(f)
    return _cache[name]


def artifacts_available():
    return os.path.exists(os.path.join(ARTIFACTS_DIR, "content_mat.pkl"))


def poster_url(poster_path):
    if poster_path:
        return f"{POSTER_BASE}{poster_path}"
    return None


def format_movie(mid, meta, score=None):
    out = {
        "movie_id": mid,
        "title": meta["title"],
        "year": meta.get("year"),
        "genres": meta.get("genres", []),
        "overview": meta.get("overview", ""),
        "cast": meta.get("cast", ""),
        "director": meta.get("director", ""),
        "vote_average": meta.get("vote_average", 0),
        "vote_count": meta.get("vote_count", 0),
        "poster_url": poster_url(meta.get("poster_path", "")),
    }
    if score is not None:
        out["score"] = round(score, 4)
    return out


def recommend_for_taste(genres, liked_movie_ids, disliked_movie_ids, k=20):
    content_mat = _load("content_mat")
    content_index = _load("content_index")
    movie_lookup = _load("movie_lookup")
    popularity_score = _load("popularity_score")
    movie_ids = _load("movie_ids")

    exclude = set(disliked_movie_ids or []) | set(liked_movie_ids or [])
    genre_set = set(genres or [])

    liked_indices = [
        content_index[mid] for mid in (liked_movie_ids or [])
        if mid in content_index
    ]
    taste_vec = None
    if liked_indices:
        taste_vec = content_mat[liked_indices].mean(axis=0)
        if hasattr(taste_vec, "A"):
            taste_vec = taste_vec.A

    max_pop = popularity_score.max() or 1.0

    scores = []
    for i, mid in enumerate(movie_ids):
        mid = int(mid)
        if mid in exclude or mid not in movie_lookup:
            continue
        meta = movie_lookup[mid]

        if genre_set:
            movie_genres = set(meta.get("genres", []))
            genre_match = len(genre_set & movie_genres) / len(genre_set)
        else:
            genre_match = 0.5

        content_sim = 0.5
        if taste_vec is not None and mid in content_index:
            idx = content_index[mid]
            sim = cosine_similarity(content_mat[idx], taste_vec.reshape(1, -1))
            content_sim = float(sim.flat[0])

        quality = float(popularity_score[i]) / max_pop

        score = 0.45 * content_sim + 0.25 * genre_match + 0.30 * quality
        scores.append((mid, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return [format_movie(mid, movie_lookup[mid], score) for mid, score in scores[:k]]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "artifacts_ready": artifacts_available()})


@app.route("/api/genres")
def api_genres():
    if not artifacts_available():
        return jsonify({"error": "Model not trained."}), 503
    genre_list = _load("genre_list")
    return jsonify({"genres": [g["name"] for g in genre_list]})


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    if not artifacts_available():
        return jsonify({"error": "Model not trained. Run python pipeline.py first."}), 503
    try:
        body = request.get_json(force=True) or {}
        genres = body.get("genres", [])
        liked = body.get("liked_movies", [])
        disliked = body.get("disliked_movies", [])
        k = min(int(body.get("k", 20)), 50)
        recs = recommend_for_taste(genres, liked, disliked, k)
        return jsonify({"recommendations": recs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/movies/popular")
def api_movies_popular():
    if not artifacts_available():
        return jsonify({"error": "Model not trained."}), 503
    try:
        genres_param = request.args.get("genres", "").strip()
        n = min(int(request.args.get("n", 8)), 20)
        movie_lookup = _load("movie_lookup")
        popularity_score = _load("popularity_score")
        movie_ids = _load("movie_ids")

        selected_genres = {g.strip() for g in genres_param.split(",") if g.strip()} if genres_param else set()

        scored = sorted(
            zip(movie_ids, popularity_score),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for mid, _ in scored:
            mid = int(mid)
            meta = movie_lookup.get(mid)
            if not meta:
                continue
            if selected_genres and not (selected_genres & set(meta.get("genres", []))):
                continue
            results.append(format_movie(mid, meta))
            if len(results) >= n:
                break

        return jsonify({"movies": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def api_search():
    if not artifacts_available():
        return jsonify({"error": "Model not trained."}), 503
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify({"results": []})
    movie_lookup = _load("movie_lookup")
    pattern = re.compile(re.escape(q), re.IGNORECASE)
    results = []
    for mid, meta in movie_lookup.items():
        if pattern.search(meta.get("title", "")):
            results.append(format_movie(mid, meta))
        if len(results) >= 20:
            break
    return jsonify({"results": results})


@app.route("/api/movie/<int:movie_id>")
def api_movie(movie_id):
    if not artifacts_available():
        return jsonify({"error": "Model not trained."}), 503
    movie_lookup = _load("movie_lookup")
    meta = movie_lookup.get(movie_id)
    if not meta:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(format_movie(movie_id, meta))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
