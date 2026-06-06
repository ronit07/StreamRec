import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

BASE_URL = "https://api.themoviedb.org/3"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get(path, params, api_key, retries=3):
    params["api_key"] = api_key
    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}{path}", params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def fetch_genres(api_key):
    data = get("/genre/movie/list", {"language": "en-US"}, api_key)
    return data["genres"]


def fetch_discover_page(api_key, genre_id, page):
    return get("/discover/movie", {
        "with_genres": genre_id,
        "sort_by": "popularity.desc",
        "page": page,
        "language": "en-US",
        "include_adult": "false",
    }, api_key)


def fetch_movie_details(api_key, movie_id):
    try:
        kw = get(f"/movie/{movie_id}/keywords", {}, api_key)
        credits = get(f"/movie/{movie_id}/credits", {}, api_key)
        keywords = " ".join(k["name"] for k in kw.get("keywords", []))
        cast = [m["name"] for m in credits.get("cast", [])[:5]]
        director = next(
            (m["name"] for m in credits.get("crew", []) if m["job"] == "Director"),
            ""
        )
        return keywords, " ".join(cast), director
    except Exception:
        return "", "", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="TMDB API key")
    parser.add_argument("--pages-per-genre", type=int, default=20, help="Pages per genre (20 results/page)")
    args = parser.parse_args()

    api_key = args.api_key
    pages = args.pages_per_genre

    print("Fetching genre list...")
    genres = fetch_genres(api_key)
    with open(os.path.join(DATA_DIR, "genre_list.json"), "w") as f:
        json.dump(genres, f, indent=2)
    print(f"  {len(genres)} genres")

    genre_map = {g["id"]: g["name"] for g in genres}

    print("Fetching movie discover pages...")
    raw_movies = {}

    discover_tasks = [
        (g["id"], p) for g in genres for p in range(1, pages + 1)
    ]

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(fetch_discover_page, api_key, gid, page): (gid, page)
            for gid, page in discover_tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Discover pages"):
            gid, page = futures[fut]
            time.sleep(0.05)
            try:
                data = fut.result()
                for m in data.get("results", []):
                    mid = m["id"]
                    if mid not in raw_movies:
                        raw_movies[mid] = m
            except Exception:
                pass

    print(f"  {len(raw_movies):,} unique movies found")

    print("Fetching keywords + credits for each movie...")
    rows = []

    def enrich(mid, m):
        keywords, cast, director = fetch_movie_details(api_key, mid)
        time.sleep(0.05)
        genre_names = "|".join(
            genre_map[gid] for gid in m.get("genre_ids", []) if gid in genre_map
        )
        return {
            "movie_id": mid,
            "title": m.get("title", ""),
            "year": (m.get("release_date") or "")[:4],
            "overview": (m.get("overview") or "").replace("\n", " "),
            "genres": genre_names,
            "keywords": keywords,
            "cast": cast,
            "director": director,
            "vote_average": m.get("vote_average", 0),
            "vote_count": m.get("vote_count", 0),
            "popularity": m.get("popularity", 0),
            "poster_path": m.get("poster_path") or "",
        }

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(enrich, mid, m): mid for mid, m in raw_movies.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Enriching movies"):
            try:
                rows.append(fut.result())
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df = df[df["title"].str.strip().astype(bool)]
    df = df[df["vote_count"] >= 10]
    df = df.drop_duplicates(subset="movie_id")

    out = os.path.join(DATA_DIR, "tmdb_movies.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} movies to {out}")


if __name__ == "__main__":
    main()
