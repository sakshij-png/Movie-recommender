# 🎬 CineMatch: Movie Recommender

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E?logo=scikitlearn&logoColor=white)
![TMDB](https://img.shields.io/badge/Data-TMDB-01B4E4)

A content-based movie recommender for Hollywood, Bollywood and regional Indian cinema, built with Python, NLP (TF-IDF, cosine similarity) and Streamlit.

**Live demo:** (https://movie-recommender-wnycfuk3zttqtxnrmui7ir.streamlit.app/)

![Home](screenshots/home.png)

## Features

- **Similar-movie search:** pick a movie you like and get close matches.
- **Mood-based picks:** choose a mood (Happy, Mind-bending, Nostalgic, ...) and get a fresh set on every click.
- **Filters:** language/industry, genre, minimum rating and release year, e.g. "Hindi movies similar to Inception".
- **Rich UI:** dark poster-grid theme with a details pop-up, embedded trailers and streaming availability in India.

![Details](screenshots/details.png)

## Dataset

Hollywood titles come from the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) on Kaggle. Indian movies (Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Bengali, Punjabi) are pulled from the TMDB API with `build_indian_dataset.py`. Both are merged into `movies_all.pkl`. Posters, trailers and streaming info are fetched live from TMDB.

## How it works

1. For each movie, the overview, genres, keywords, top 3 cast and director are combined into one text "tags" string.
2. Tags are stemmed (NLTK) and converted to TF-IDF vectors.
3. Cosine similarity between the chosen movie and all others gives the closest matches, after applying the sidebar filters.
4. Mood picks filter by genre and rank with a weighted rating, so movies with very few votes don't dominate.

## Project structure

```
movie-recommender/
├── app.py                    # Streamlit app
├── build_indian_dataset.py   # Fetches Indian movies from the TMDB API
├── recommender.ipynb         # Cleaning, feature building, algorithm comparison
├── movies_all.pkl            # Final dataset used by the app
├── requirements.txt
├── .streamlit/config.toml    # Dark theme
└── screenshots/
```

## Run it locally

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
pip install -r requirements.txt
```

Get a free API key from [themoviedb.org](https://www.themoviedb.org) and save it in `.streamlit/secrets.toml`:

```toml
TMDB_API_KEY = "your_key_here"
```

Then start the app:

```bash
streamlit run app.py
```

## Key learnings

- **Messy data needs real cleaning:** TMDB's genre, cast and keyword columns are JSON stored as text, and collapsing names like `Sam Worthington` into one token stopped unrelated people from matching.
- **Stemming + TF-IDF beat raw word counts** in side-by-side checks, though evaluating a recommender without user ratings is hard (next step: precision@k on MovieLens).
- **Weighted ratings fix noisy rankings** from movies with only a handful of votes.
- **Compute one similarity row per query** instead of a full movie-by-movie matrix. It's lighter on memory and starts faster.
- **Streamlit reruns the whole script on every click,** so results live in `st.session_state` and API calls are cached.
- **Keep secrets out of Git and pin dependency versions,** since a pickle can break across pandas versions.

## Author

**Sakshi Joshi**
final-year B.Tech student in AI & Data Science.

_Movie data and images from TMDB. Streaming availability by JustWatch. This product uses the TMDB API but is not endorsed or certified by TMDB._
