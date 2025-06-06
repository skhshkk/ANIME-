# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import ast

# ───────────────────────────────────────────────────────────────────────────────
# 1) Page Configuration & Title
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎥 Anime Recommender",
    page_icon="🎬",
    layout="wide",
)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Inject Updated CSS (Darker Cards, Box‐Shadow, Rounded Corners, Hover)
# ───────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Centered Header Styling */
        .main-header {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        .sub-text {
            text-align: center;
            font-size: 1rem;
            color: #CCCCCC;
            margin-bottom: 2rem;
        }

        /* Recommendation Card Styling */
        .anime-card {
            background-color: #1E1E2E;      /* Dark tinted background */
            border-radius: 12px;            /* Rounded corners */
            padding: 1.2rem;                /* Generous internal padding */
            margin-bottom: 1.5rem;          /* Space between cards */
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);  /* Soft shadow */
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .anime-card:hover {
            transform: translateY(-4px);    /* Lift up on hover */
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        /* Title Styling */
        .anime-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #FFFFFF;                 /* White text on dark background */
            margin-bottom: 0.3rem;
        }

        /* Metadata Styling (Genres, Rating, Episodes) */
        .anime-meta {
            font-size: 0.9rem;
            color: #DDDDDD;                 /* Soft gray text */
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }

        /* Link Styling */
        .anime-link {
            font-size: 0.9rem;
            text-decoration: none;
            color: #7FBFFF;                 /* Light blue link */
        }
        .anime-link:hover {
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ───────────────────────────────────────────────────────────────────────────────
# 3) Page Title & Subtitle
# ───────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎥 Anime Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Filter by Genre(s), Rating, and Episodes to find your next watch!</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# 4) Load & Cache the Anime Dataset
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_anime_data(csv_path="Anime_data.csv"):
    """
    Loads the anime CSV, parses genres into lists, 
    coerces numeric columns, and returns DataFrame + sorted unique genres.
    """
    df = pd.read_csv(csv_path)

    # Drop rows missing critical fields
    df = df.dropna(subset=["Genre", "Rating", "Episodes", "Title"])

    # Parse Genre from string representation of list to actual Python list
    def parse_genre(g):
        try:
            return list(ast.literal_eval(g))
        except Exception:
            return []

    df["GenreList"] = df["Genre"].apply(parse_genre)

    # Convert Episodes to numeric (drop if non-numeric)
    df["Episodes"] = pd.to_numeric(df["Episodes"], errors="coerce")
    df = df.dropna(subset=["Episodes"])
    df["Episodes"] = df["Episodes"].astype(int)

    # Convert Rating to numeric (drop if non-numeric)
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])

    # Collect all unique genres for the sidebar multiselect
    all_genres = sorted({genre for sub in df["GenreList"] for genre in sub})

    return df.reset_index(drop=True), all_genres

anime_df, ALL_GENRES = load_anime_data()

# ───────────────────────────────────────────────────────────────────────────────
# 5) Sidebar: User Filters
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter Your Preferences")

# 5a) Multi‐select for Genre
selected_genres = st.sidebar.multiselect(
    "Pick one or more genres:", ALL_GENRES, default=[]
)

# 5b) Slider for Minimum Rating
min_rating = st.sidebar.slider(
    "Minimum Rating:",
    min_value=float(np.floor(anime_df["Rating"].min())),
    max_value=float(np.ceil(anime_df["Rating"].max())),
    value=7.0,
    step=0.1,
)

# 5c) Episodes Range Slider
max_eps = int(anime_df["Episodes"].max())
episodes_range = st.sidebar.slider(
    "Episode Count Range:",
    min_value=0,
    max_value=max_eps,
    value=(0, max_eps),
    step=1,
)

# 5d) Number of Recommendations to Show
K = st.sidebar.slider(
    "Number of Recommendations to Show:",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)

# 5e) Recommend Button
should_recommend = st.sidebar.button("🔎 Recommend Anime")

# ───────────────────────────────────────────────────────────────────────────────
# 6) Recommendation Logic Function
# ───────────────────────────────────────────────────────────────────────────────
def recommend_anime(df, genres, min_rating, eps_range, top_k=10):
    """
    Filters anime by rating and episodes, then ranks by:
     1) Number of matching genres (if any selected)
     2) Rating (descending)
    Returns top_k results.
    """
    low_eps, high_eps = eps_range

    # Filter by rating and episodes
    filtered = df[
        (df["Rating"] >= min_rating)
        & (df["Episodes"] >= low_eps)
        & (df["Episodes"] <= high_eps)
    ].copy()

    if genres:
        # Compute how many of the selected genres each anime has
        def count_matches(row):
            return sum(1 for g in row["GenreList"] if g in genres)
        filtered["GenreMatchCount"] = filtered.apply(count_matches, axis=1)

        # Keep only those with at least 1 genre match
        filtered = filtered[filtered["GenreMatchCount"] > 0]

        # Sort by match count (desc), then rating (desc)
        filtered = filtered.sort_values(
            ["GenreMatchCount", "Rating"], ascending=[False, False]
        )
    else:
        # If no genres selected, just sort by rating
        filtered = filtered.sort_values("Rating", ascending=False)
        filtered["GenreMatchCount"] = 0

    return filtered.head(top_k)

# ───────────────────────────────────────────────────────────────────────────────
# 7) Display Recommendations or Instruction
# ───────────────────────────────────────────────────────────────────────────────
if should_recommend:
    with st.spinner("Finding the best anime for you..."):
        results_df = recommend_anime(
            anime_df,
            selected_genres,
            min_rating,
            episodes_range,
            top_k=K,
        )

    if results_df.empty:
        st.warning("No anime matched your filter criteria. Try adjusting filters.")
    else:
        st.markdown(f"### 🎯 Top {len(results_df)} Recommendations")
        # Render each recommendation as a styled card
        for _, row in results_df.reset_index(drop=True).iterrows():
            st.markdown(
                f"""
                <div class="anime-card">
                  <div class="anime-title">{row['Title']}</div>
                  <div class="anime-meta">
                    Genres: {', '.join(row['GenreList'])} <br>
                    Rating: {row['Rating']:.1f} &nbsp;|&nbsp; Episodes: {row['Episodes']} &nbsp;|&nbsp; Type: {row['Type']} <br>
                    <a class="anime-link" href="{row['Link']}" target="_blank">View on MyAnimeList</a>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.info("Use the filters in the sidebar and click **“🔎 Recommend Anime”** to see your personalized list.")

# ───────────────────────────────────────────────────────────────────────────────
# 8) Footer / Credits
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #999; font-size: 0.85rem;">
      Built with ❤️ using Streamlit · Anime dataset from MyAnimeList.com
    </div>
    """,
    unsafe_allow_html=True,
)
