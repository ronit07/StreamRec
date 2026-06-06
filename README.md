# StreamRec

Movie recommendations without an account. Pick genres, rate a few films, get picks — that's it.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-CDN-38bdf8)
![TMDB](https://img.shields.io/badge/Data-TMDB%20API-01b4e4)

---

## How it works

1. Pick the genres you're into
2. Thumbs up or skip 6–8 popular films from those genres
3. Get a ranked list of recommendations with real posters

Recommendations are scored by:

```
score = 0.45 × content similarity   (TF-IDF cosine vs your liked films)
      + 0.25 × genre match
      + 0.30 × quality              (log(vote count) × TMDB rating)
```

---

## Data

Pulls from the [TMDB API](https://www.themoviedb.org/settings/api) — free tier, no credit card. The `fetch_tmdb.py` script downloads ~20,000 movies with overviews, genres, keywords, cast, and directors, then `pipeline.py` builds a TF-IDF model on top of that.

---

## Setup

**1. Get a free TMDB API key**

Register at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api). Takes about 2 minutes.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Fetch movies** (run once, ~15 minutes)

```bash
python fetch_tmdb.py --api-key YOUR_TMDB_API_KEY
```

Saves to `data/tmdb_movies.csv`.

**4. Build the model** (~1 minute)

```bash
python pipeline.py
```

Saves artifacts to `artifacts/`.

**5. Run**

```bash
python app.py
# http://localhost:5000
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Status check |
| `GET` | `/api/genres` | List of genres |
| `POST` | `/api/recommend` | Get recommendations |
| `GET` | `/api/movies/popular` | Popular movies, filterable by genre |
| `GET` | `/api/search?q=` | Search by title |
| `GET` | `/api/movie/<id>` | Single movie detail |

**POST /api/recommend**

```json
{
  "genres": ["Action", "Sci-Fi"],
  "liked_movies": [550, 13, 680],
  "disliked_movies": [11],
  "k": 20
}
```

```json
{
  "recommendations": [
    {
      "movie_id": 157336,
      "title": "Interstellar",
      "year": "2014",
      "genres": ["Adventure", "Drama", "Sci-Fi"],
      "avg_rating": 8.4,
      "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
      "score": 0.8821
    }
  ]
}
```

---

## Project structure

```
StreamRec/
├── fetch_tmdb.py       # Download movie data from TMDB
├── pipeline.py         # Build TF-IDF model
├── app.py              # Flask server
├── templates/
│   └── index.html      # Frontend (Tailwind CSS)
├── requirements.txt
├── data/               # gitignored, created by fetch_tmdb.py
└── artifacts/          # gitignored, created by pipeline.py
```

---

## Notes

- `data/` and `artifacts/` are gitignored. Run steps 3 and 4 after cloning.
- TMDB free tier allows 40 requests/10s. The fetch script stays within that.
- Poster images load directly from TMDB's CDN.
