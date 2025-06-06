# app.py

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
st.markdown('<div class="sub-text">Filter by Genre(s), Rating Bucket, and Episode Count Bucket to find your next watch!</div>', unsafe_allow_html=True)

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

# 5b) Dropdown for Rating Bucket
rating_buckets = {
    "Any": (0.0, 10.0),
    "1–2": (1.0, 2.0),
    "3–5": (3.0, 5.0),
    "5–7": (5.0, 7.0),
    "7–9": (7.0, 9.0),
    "9–10": (9.0, 10.0),
}
selected_rating_bucket = st.sidebar.selectbox(
    "Select Rating Bucket:",
    options=list(rating_buckets.keys()),
    index=list(rating_buckets.keys()).index("7–9")
)

# 5c) Dropdown for Episodes Bucket
episode_buckets = {
    "Any": (0, anime_df["Episodes"].max()),
    "1 (Movie/OVA)": (1, 1),
    "1–12": (1, 12),
    "13–24": (13, 24),
    "25–50": (25, 50),
    "51–100": (51, 100),
    "101+": (101, anime_df["Episodes"].max()),
}
selected_episode_bucket = st.sidebar.selectbox(
    "Select Episode Bucket:",
    options=list(episode_buckets.keys()),
    index=list(episode_buckets.keys()).index("1–12")
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
def recommend_anime(df, genres, rating_bucket, episode_bucket, top_k=10):
    """
    Filters anime by rating bucket and episode bucket, then ranks by:
     1) Number of matching genres (if any selected)
     2) Rating (descending)
    Returns top_k results.
    """
    # Unpack numeric ranges for the buckets
    min_rating, max_rating = rating_buckets[rating_bucket]
    min_eps, max_eps = episode_buckets[episode_bucket]

    # 1) Filter by rating and episodes
    filtered = df[
        (df["Rating"] >= min_rating)
        & (df["Rating"] <= max_rating)
        & (df["Episodes"] >= min_eps)
        & (df["Episodes"] <= max_eps)
    ].copy()

    if genres:
        # 2) Compute number of matching genres
        def count_matches(row):
            return sum(1 for g in row["GenreList"] if g in genres)
        filtered["GenreMatchCount"] = filtered.apply(count_matches, axis=1)

        # Keep only anime that match at least one selected genre
        filtered = filtered[filtered["GenreMatchCount"] > 0]

        # 3) Sort by match count (desc), then rating (desc)
        filtered = filtered.sort_values(
            ["GenreMatchCount", "Rating"], ascending=[False, False]
        )
    else:
        # If no genres selected, simply sort by rating
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
            selected_rating_bucket,
            selected_episode_bucket,
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
      Built with ❤ using Streamlit · Anime dataset from MyAnimeList.com
    </div>
    """,
    unsafe_allow_html=True,
)
