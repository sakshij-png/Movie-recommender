import re
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

# Reads your key from the secrets file you already made
API_KEY = re.search(r'TMDB_API_KEY\s*=\s*"([^"]+)"', open('.streamlit/secrets.toml').read()).group(1)
BASE = "https://api.themoviedb.org/3"

# language code: number of pages to pull (20 movies per page)
LANGS = {"hi": 25, "ta": 12, "te": 12, "ml": 10, "kn": 8, "mr": 6, "bn": 6, "pa": 5}


def get(path, params=None):
    params = dict(params or {})
    params["api_key"] = API_KEY
    for _ in range(3):
        try:
            r = requests.get(f"{BASE}{path}", params=params, timeout=10)
            if r.status_code == 429:
                time.sleep(2)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1)
    return None


def discover(lang, pages):
    ids = []
    for p in range(1, pages + 1):
        data = get("/discover/movie", {
            "with_original_language": lang,
            "sort_by": "popularity.desc",
            "vote_count.gte": 20,
            "include_adult": "false",
            "page": p,
        })
        if not data or not data.get("results"):
            break
        ids += [m["id"] for m in data["results"]]
    return ids


def squash(names):
    return [n.replace(" ", "") for n in names]


def details(movie_id):
    d = get(f"/movie/{movie_id}", {"append_to_response": "credits,keywords"})
    if not d:
        return None
    overview = d.get("overview") or ""
    genres = [g["name"] for g in d.get("genres", [])]
    keywords = [k["name"] for k in d.get("keywords", {}).get("keywords", [])]
    cast = [c["name"] for c in d.get("credits", {}).get("cast", [])[:3]]
    director = [c["name"] for c in d.get("credits", {}).get("crew", []) if c.get("job") == "Director"][:1]

    words = overview.split() + squash(genres) + squash(keywords) + squash(cast) + squash(director)
    if not words:
        return None
    release = d.get("release_date") or ""
    return {
        "movie_id": d["id"],
        "title": d["title"],
        "tags": " ".join(words).lower(),
        "overview_text": overview or "No description available.",
        "genres": genres,
        "vote_average": d.get("vote_average", 0),
        "vote_count": d.get("vote_count", 0),
        "year": int(release[:4]) if release else 0,
        "language": d["original_language"],
    }


if __name__ == "__main__":
    rows = []
    for lang, pages in LANGS.items():
        ids = discover(lang, pages)
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(details, ids))
        rows += [r for r in results if r]
        print(f"{lang}: found {len(ids)} movies")

    df = pd.DataFrame(rows).drop_duplicates(subset="movie_id")
    df.to_pickle("movies_indian.pkl")
    print(df.shape)
    print(df["language"].value_counts())