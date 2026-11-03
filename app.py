import html
import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="CineMatch", page_icon="🎬", layout="wide")

PLACEHOLDER = "https://placehold.co/500x750/1a1a22/9a9aa6?text=No+Poster"
COUNTRY = "IN"  # streaming availability country
COLS = 5        # posters per row

LANG_NAMES = {
    "en": "English (Hollywood)", "hi": "Hindi (Bollywood)", "ta": "Tamil",
    "te": "Telugu", "ml": "Malayalam", "kn": "Kannada", "mr": "Marathi",
    "bn": "Bengali", "pa": "Punjabi", "fr": "French", "es": "Spanish",
    "ja": "Japanese", "ko": "Korean", "de": "German", "zh": "Chinese", "it": "Italian",
}

MOODS = {
    "😄 Happy / feel-good": {"genres": ["Comedy", "Family", "Animation", "Music"], "avoid": ["Horror", "War", "Crime"]},
    "😢 Sad / want a good cry": {"genres": ["Drama", "Romance"], "avoid": ["Horror", "Comedy"]},
    "🔥 Pumped up": {"genres": ["Action", "Adventure", "Science Fiction"], "avoid": ["Documentary"]},
    "😱 Want a scare": {"genres": ["Horror", "Thriller", "Mystery"], "avoid": ["Family", "Animation"]},
    "💕 Romantic": {"genres": ["Romance"], "avoid": ["Horror", "War"]},
    "🤯 Mind-bending": {"genres": ["Science Fiction", "Mystery", "Thriller"], "avoid": ["Family"]},
    "😌 Chill / relaxed": {"genres": ["Comedy", "Romance", "Family", "Music", "Animation"],
                           "avoid": ["Horror", "Crime", "War", "Thriller"]},
    "🌍 Adventurous": {"genres": ["Adventure", "Fantasy"], "avoid": ["Horror"]},
    "🕰️ Nostalgic (pre-2005)": {"genres": [], "avoid": [], "year_to": 2005},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');

#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1300px; }

.hero { padding: 1rem 0 1.4rem; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1.2rem; }
.hero-title { font-family: 'Bebas Neue', sans-serif; font-size: 4rem; letter-spacing: 3px; line-height: 1; color: #f5f5f7; }
.hero-title span { color: #e11d48; }
.hero-sub { color: #9a9aa6; margin-top: 0.4rem; font-size: 1.05rem; }

.card { background: #1a1a22; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.2s ease, border-color 0.2s ease; margin-bottom: 8px; }
.card:hover { transform: translateY(-4px); border-color: #e11d48; }
.poster { position: relative; aspect-ratio: 2 / 3; overflow: hidden; background: #0f0f14; }
.poster img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform 0.3s ease; }
.card:hover .poster img { transform: scale(1.05); }
.badge { position: absolute; top: 8px; right: 8px; background: rgba(15,15,20,0.85); color: #ffffff;
         padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
.info { padding: 10px 12px 12px; }
.title { font-weight: 600; font-size: 14px; line-height: 1.3; margin-bottom: 2px; color: #f5f5f7;
         display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.meta { color: #9a9aa6; font-size: 12px; }
.pills span { display: inline-block; background: rgba(225,29,72,0.15); color: #fb7185; font-size: 11px;
              padding: 2px 8px; border-radius: 10px; margin: 6px 4px 0 0; }

.stButton > button, div[data-testid="stButton"] > button { width: 100%; border-radius: 8px; }
</style>
"""


@st.cache_resource
def load_data():
    movies = pd.read_pickle('movies_all.pkl')
    tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
    matrix = tfidf.fit_transform(movies['tags'])

    C = movies.loc[movies['vote_count'] > 0, 'vote_average'].mean()
    m = 100
    v = movies['vote_count']
    movies['score'] = (v / (v + m)) * movies['vote_average'] + (m / (v + m)) * C
    return movies, matrix


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_extras(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]
    extras = {"poster": PLACEHOLDER, "trailer": None, "stream": [], "rent_buy": []}
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={
                "api_key": api_key,
                "append_to_response": "videos,watch/providers",
                "include_video_language": "en,hi,ta,te,ml,kn,mr,bn,pa,null",
            },
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return extras

    if data.get("poster_path"):
        extras["poster"] = f"https://image.tmdb.org/t/p/w500{data['poster_path']}"

    videos = data.get("videos", {}).get("results", [])
    trailers = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
    if trailers:
        extras["trailer"] = f"https://www.youtube.com/watch?v={trailers[0]['key']}"

    country = data.get("watch/providers", {}).get("results", {}).get(COUNTRY, {})
    extras["stream"] = [p["provider_name"] for p in country.get("flatrate", [])]
    rb = country.get("rent", []) + country.get("buy", [])
    extras["rent_buy"] = sorted({p["provider_name"] for p in rb})
    return extras


movies, matrix = load_data()


def filter_mask(language, genres, min_rating, year_range):
    mask = (movies['vote_average'] >= min_rating) & movies['year'].between(year_range[0], year_range[1])
    if language != "all":
        mask &= movies['language'] == language
    if genres:
        mask &= movies['genres'].apply(lambda g: any(x in g for x in genres))
    return mask


def recommend(label, n, mask):
    idx = movies.index[movies['label'] == label][0]
    sims = pd.Series((matrix @ matrix[idx].T).toarray().ravel(), index=movies.index)
    mask = mask.copy()
    mask[idx] = False
    top = sims[mask].sort_values(ascending=False).head(n)
    result = movies.loc[top.index].copy()
    result['match'] = top
    return result


def mood_picks(mood, n, mask):
    cfg = MOODS[mood]
    m = mask.copy()
    if cfg["genres"]:
        m &= movies['genres'].apply(lambda g: any(x in g for x in cfg["genres"]))
    if cfg["avoid"]:
        m &= ~movies['genres'].apply(lambda g: any(x in g for x in cfg["avoid"]))
    if "year_to" in cfg:
        m &= movies['year'] <= cfg["year_to"]
    pool = movies[m].sort_values('score', ascending=False).head(40)
    if pool.empty:
        return pool
    return pool.sample(min(n, len(pool))).sort_values('score', ascending=False)


def card_html(row):
    extras = fetch_extras(row['movie_id'])
    year = row['year'] if row['year'] > 0 else "N/A"
    lang_name = LANG_NAMES.get(row['language'], row['language'])
    pills = "".join(f"<span>{html.escape(g)}</span>" for g in row['genres'][:2])
    match = f" · {row['match']:.0%} match" if 'match' in row.index else ""
    title = html.escape(row['title'])
    return (
        f'<div class="card"><div class="poster"><img src="{extras["poster"]}" alt="{title} poster">'
        f'<span class="badge">★ {row["vote_average"]:.1f}</span></div>'
        f'<div class="info"><div class="title">{title}</div>'
        f'<div class="meta">{year} · {html.escape(lang_name)}{match}</div>'
        f'<div class="pills">{pills}</div></div></div>'
    )


@st.dialog("Movie details", width="large")
def details_dialog(row):
    extras = fetch_extras(row['movie_id'])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(extras['poster'], width=220)
    with c2:
        st.subheader(row['title'])
        year = row['year'] if row['year'] > 0 else "N/A"
        lang_name = LANG_NAMES.get(row['language'], row['language'])
        st.caption(f"★ {row['vote_average']:.1f}  ·  {year}  ·  {lang_name}  ·  {', '.join(row['genres'])}")
        st.write(row['overview_text'])
        if extras['stream']:
            st.write("📺 Stream on: " + ", ".join(extras['stream']))
        elif extras['rent_buy']:
            st.write("🛒 Rent/buy on: " + ", ".join(extras['rent_buy']))
        else:
            st.write("📺 No streaming info for India")
    if extras['trailer']:
        st.video(extras['trailer'])
    else:
        url = "https://www.youtube.com/results?search_query=" + quote_plus(f"{row['title']} trailer")
        st.link_button("🔎 Search trailer on YouTube", url)


def show_grid(results):
    for start in range(0, len(results), COLS):
        chunk = results.iloc[start:start + COLS]
        cols = st.columns(COLS)
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                st.markdown(card_html(row), unsafe_allow_html=True)
                if st.button("Details & trailer", key=f"details_{row['movie_id']}"):
                    details_dialog(row)


# ---------- Header ----------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><div class="hero-title">CINE<span>MATCH</span></div>'
    '<div class="hero-sub">Find your next favourite movie. Hollywood, Bollywood and beyond.</div></div>',
    unsafe_allow_html=True,
)

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")

num_recs = st.sidebar.selectbox("Number of recommendations", list(range(3, 11)), index=2)

available = set(movies['language'])
lang_options = ["all"] + [c for c in LANG_NAMES if c in available]
language = st.sidebar.selectbox(
    "Language / industry", lang_options,
    format_func=lambda c: "All languages" if c == "all" else LANG_NAMES[c],
)

all_genres = sorted({g for gl in movies['genres'] for g in gl})
selected_genres = st.sidebar.multiselect("Genres (any of)", all_genres)

min_rating = st.sidebar.selectbox(
    "Minimum rating", [0, 5, 6, 7, 8], index=1,
    format_func=lambda x: "Any" if x == 0 else f"{x}+",
)

years = sorted(movies.loc[movies['year'] > 0, 'year'].unique())
year_from = st.sidebar.selectbox("From year", years, index=0)
year_to = st.sidebar.selectbox("To year", years, index=len(years) - 1)
if year_from > year_to:
    st.sidebar.error("'From year' must be earlier than 'To year'.")
    st.stop()

mask = filter_mask(language, selected_genres, min_rating, (year_from, year_to))

# ---------- Controls ----------
tab1, tab2 = st.tabs(["🎬 Similar to a movie", "😊 Pick by mood"])

with tab1:
    st.write("Pick a movie you like. Try combining it with the language filter!")
    selected = st.selectbox("Choose a movie", sorted(movies['label']))
    if st.button("Recommend", type="primary", key="recommend_btn"):
        st.session_state['results'] = recommend(selected, num_recs, mask)
        st.session_state['heading'] = f"Because you liked {selected}"

with tab2:
    st.write("Tell me how you feel. Click again for a different set.")
    mood = st.pills("How are you feeling?", list(MOODS), selection_mode="single", key="mood_pills")
    if st.button("Pick for me", type="primary", key="mood_btn"):
        if mood is None:
            st.warning("Choose a mood first.")
        else:
            st.session_state['results'] = mood_picks(mood, num_recs, mask)
            st.session_state['heading'] = f"Picks for: {mood}"

# ---------- Results ----------
results = st.session_state.get('results')
if results is not None:
    st.subheader(st.session_state.get('heading', 'Results'))
    if results.empty:
        st.warning("No movies match these filters. Try loosening them in the sidebar.")
    else:
        show_grid(results)

st.caption(
    "Movie data and images from TMDB. Streaming availability data provided by JustWatch. "
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)