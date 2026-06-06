# StreamRec

A content-based movie recommendation engine with a dark cinema UI, real TMDB posters, and a personalised onboarding flow — no account or login required. Works for anyone.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-lightgrey)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-CDN-38bdf8)
![TMDB](https://img.shields.io/badge/Data-TMDB%20API-01b4e4)

---

## How it works

1. **Pick genres** — choose what you're in the mood for from a grid of genre chips
2. **Rate a few films** — thumbs up or skip 6–8 popular movies in your genres
3. **Get recommendations** — personalised picks with real posters, ratings, and overviews

No user ID, no account, no history needed. Each session builds a fresh taste profile.

---

## Architecture

```
  TMDB API (20,000+ movies)
       │
       ▼
  fetch_tmdb.py  ──►  data/tmdb_movies.csv
                            │
                            ▼
                       pipeline.py
                            │
              TF-IDF on title + overview +
              genres + keywords + cast + director
                            │
                            ▼
                       artifacts/
                            │
                            ▼
  User picks genres ──► app.py ──► Scored recommendations
  User rates movies        │
                           ▼
                    templates/index.html
                    (Tailwind CSS + vanilla JS)
```

**Scoring formula:**

```
score = 0.45 × content_similarity
      + 0.25 × genre_match
      + 0.30 × quality  (log(vote_count) × vote_average)
```

Content similarity is computed as cosine similarity between a movie's TF-IDF vector and the average vector of movies the user liked during onboarding.

---

## Tech Stack

| Layer | Stack |
|---|---|
| Data | TMDB API — 20,000+ movies up to 2025 |
| Model | TF-IDF (50k features, bigrams, scikit-learn) |
| Backend | Flask |
| Frontend | Tailwind CSS (CDN), vanilla JS |
| Posters | TMDB image CDN |

---

## Setup

### 1. Get a free TMDB API key

Register at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api) — free, takes about 2 minutes. No credit card.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Fetch the movie database

```bash
python fetch_tmdb.py --api-key YOUR_TMDB_API_KEY
```

Fetches ~20,000 movies with overviews, genres, keywords, cast, and directors using 10 parallel threads. Takes 10–20 minutes. Saves to `data/tmdb_movies.csv`.

### 4. Build the model

```bash
python pipeline.py
```

Fits TF-IDF on the full corpus and saves artifacts to `artifacts/`. Takes about 1 minute.

### 5. Start the server

```bash
python app.py
# → http://localhost:5000
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main UI |
| `GET` | `/api/health` | Health check |
| `GET` | `/api/genres` | All genre names |
| `POST` | `/api/recommend` | Get recommendations |
| `GET` | `/api/movies/popular` | Popular movies by genre |
| `GET` | `/api/search?q=` | Search by title |
| `GET` | `/api/movie/<id>` | Movie detail |

**POST /api/recommend — request body:**

```json
{
  "genres": ["Action", "Sci-Fi"],
  "liked_movies": [550, 13, 680],
  "disliked_movies": [11],
  "k": 20
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "movie_id": 157336,
      "title": "Interstellar",
      "year": "2014",
      "genres": ["Adventure", "Drama", "Sci-Fi"],
      "avg_rating": 8.4,
      "popularity": 98234,
      "poster_url": "https://image.tmdb.org/t/p/w500/gEU2QniE6E77NI6lCU6MxlNBvIx.jpg",
      "score": 0.8821
    }
  ]
}
```

---

## Project Structure

```
StreamRec/
├── fetch_tmdb.py        # Fetch & cache TMDB movie database
├── pipeline.py          # Build TF-IDF content model
├── app.py               # Flask API + recommendation logic
├── templates/
│   └── index.html       # Single-page app (Tailwind CSS)
├── requirements.txt
├── data/                # Created by fetch_tmdb.py (gitignored)
│   ├── tmdb_movies.csv
│   └── genre_list.json
└── artifacts/           # Created by pipeline.py (gitignored)
```

---

## Notes

- `data/` and `artifacts/` are gitignored — run the setup steps above to regenerate them
- TMDB API has a rate limit of 40 requests/10s on the free tier; `fetch_tmdb.py` stays well within this
- Poster images are served directly from TMDB's CDN — no storage needed
