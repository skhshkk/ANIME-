# app.py

import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ───────────────────────────────────────────────────────────────────────────────
# 1) Page Configuration & Title
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎥 Anime Recommender",
    page_icon="🎬",
    layout="wide",
)

# ───────────────────────────────────────────────────────────────────────────────
# 2) Inject Updated CSS (Darker Cards, Box-Shadow, Rounded Corners, Hover)
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
            background-color: #1E1E2E;
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .anime-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        /* Title Styling */
        .anime-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-bottom: 0.3rem;
        }

        /* Metadata Styling */
        .anime-meta {
            font-size: 0.9rem;
            color: #DDDDDD;
            margin-bottom: 0.5rem;
            line-height: 1.4;
        }

        /* Link Styling */
        .anime-link {
            font-size: 0.9rem;
            text-decoration: none;
            color: #7FBFFF;
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
st.markdown('<div class="sub-text">Filter by Type, Genre(s), Rating, and Episode Count to find your next watch!</div>', unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# 4) Load, Cache, and Prepare the Anime Dataset with TF-IDF on Genres
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_anime_data(csv_path="Anime_data.csv"):
    """
    Loads the anime CSV, parses genres into lists, coerces numeric columns,
    builds a TF-IDF matrix on genres, and returns DataFrame, genre list,
    type list, TF-IDF vectorizer, and TF-IDF matrix.
    """
    df = pd.read_csv(csv_path)

    # Drop rows missing critical fields
    df = df.dropna(subset=["Genre", "Rating", "Episodes", "Title", "Type"])

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

    # Combine GenreList into a single string for TF-IDF
    df["GenreString"] = df["GenreList"].apply(lambda x: " ".join(x))

    # Build TF-IDF matrix on GenreString
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["GenreString"])

    # Collect all unique genres for the sidebar multiselect
    all_genres = sorted({genre for sub in df["GenreList"] for genre in sub})

    # Collect all unique types for the sidebar multiselect
    all_types = sorted(df["Type"].unique())

    return df.reset_index(drop=True), all_genres, all_types, tfidf, tfidf_matrix

anime_df, ALL_GENRES, ALL_TYPES, tfidf_vectorizer, tfidf_matrix = load_anime_data()

# ───────────────────────────────────────────────────────────────────────────────
# 5) Sidebar: User Filters
# ───────────────────────────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter Your Preferences")

# 5a) Map original types to display labels
type_mapping = {
    "TV": "TV (Series)",
    "Movie": "Movie",
    "OVA": "OVA",
    "Special": "Special"
}
# Fallback: if a type not in mapping, display as-is
display_types = [type_mapping.get(t, t) for t in ALL_TYPES]
# Build reverse mapping to retrieve original type from display label
reverse_type_mapping = {v: k for k, v in type_mapping.items()}

# 5b) Multi-select for Type (display labels)
selected_display_types = st.sidebar.multiselect(
    "Select Type(s):", display_types, default=["TV (Series)", "Movie"]
)
# Convert back from display labels to original types
selected_types = [
    reverse_type_mapping.get(lbl, lbl) for lbl in selected_display_types
]

# 5c) Multi-select for Genre
selected_genres = st.sidebar.multiselect(
    "Pick one or more genres:", ALL_GENRES, default=[]
)

# 5d) Sliders for Minimum and Maximum Rating
min_rating = st.sidebar.slider(
    "Minimum Rating:",
    min_value=float(np.floor(anime_df["Rating"].min())),
    max_value=float(np.ceil(anime_df["Rating"].max())),
    value=7.0,
    step=0.1,
)
max_rating = st.sidebar.slider(
    "Maximum Rating:",
    min_value=float(np.floor(anime_df["Rating"].min())),
    max_value=float(np.ceil(anime_df["Rating"].max())),
    value=10.0,
    step=0.1,
)

# 5e) Dropdown for Episodes Bucket
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

# 5f) Number of Recommendations to Show
K = st.sidebar.slider(
    "Number of Recommendations to Show:",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
)

# 5g) Recommend Button
should_recommend = st.sidebar.button("🔎 Recommend Anime")

# ───────────────────────────────────────────────────────────────────────────────
# 6) Recommendation Logic Function (Content-Based using TF-IDF)
# ───────────────────────────────────────────────────────────────────────────────
def recommend_anime(df, types, genres, min_rating, max_rating, episode_bucket, top_k=10):
    """
    Filters anime by type, rating range, and episode bucket, then uses
    TF-IDF cosine similarity against a combined genre string of selected genres.
    If no genres selected, falls back to sorting by rating.
    """
    # Unpack numeric ranges for the episode bucket
    min_eps, max_eps = episode_buckets[episode_bucket]

    # 1) Filter by type, rating, and episodes
    filtered = df[
        (df["Type"].isin(types))
        & (df["Rating"] >= min_rating)
        & (df["Rating"] <= max_rating)
        & (df["Episodes"] >= min_eps)
        & (df["Episodes"] <= max_eps)
    ].copy()

    # If nothing remains after filtering, return an empty DataFrame
    if filtered.shape[0] == 0:
        return pd.DataFrame(columns=df.columns.tolist() + ["Score"])

    if genres:
        # Create a pseudo-document from selected genres
        query_string = " ".join(genres)
        query_vec = tfidf_vectorizer.transform([query_string])

        # Compute cosine similarity between query and each anime in filtered set
        indices = filtered.index.tolist()
        filtered_tfidf = tfidf_matrix[indices]

        # If filtered_tfidf has no rows, return empty DataFrame
        if filtered_tfidf.shape[0] == 0:
            return pd.DataFrame(columns=df.columns.tolist() + ["Score"])

        cosine_scores = list(enumerate(linear_kernel(query_vec, filtered_tfidf)[0]))

        # Map back to original indices and scores
        cosine_scores = [(indices[i], score) for i, score in cosine_scores]

        # Sort by similarity score descending, then by rating descending
        sorted_scores = sorted(
            cosine_scores,
            key=lambda x: (x[1], df.loc[x[0], "Rating"]),
            reverse=True,
        )

        # Take top_k
        top_indices = [idx for idx, score in sorted_scores[:top_k]]
        results = df.loc[top_indices].copy()
        results["Score"] = [score for idx, score in sorted_scores[:top_k]]
    else:
        # If no genres selected, simply sort by rating
        filtered = filtered.sort_values("Rating", ascending=False)
        results = filtered.head(top_k).copy()
        results["Score"] = 0.0

    return results

# ───────────────────────────────────────────────────────────────────────────────
# 7) Display Recommendations or Instruction
# ───────────────────────────────────────────────────────────────────────────────
if should_recommend:
    with st.spinner("Finding the best anime for you..."):
        results_df = recommend_anime(
            anime_df,
            selected_types,
            selected_genres,
            min_rating,
            max_rating,
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
                    Type: {row['Type']} &nbsp;|&nbsp;
                    Genres: {', '.join(row['GenreList'])} <br>
                    Rating: {row['Rating']:.1f} &nbsp;|&nbsp;
                    Episodes: {row['Episodes']} <br>
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
      Built with 🪐 using Streamlit · Anime dataset from MyAnimeList.com
    </div>
    """,
    unsafe_allow_html=True,
)
