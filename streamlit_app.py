"""Interactive Streamlit User Interface for MovieInsight."""

import os
import sys
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="MovieInsight | Intelligent Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_resource(show_spinner="Loading recommendation engine...")
def get_local_recommender():
    """Load and cache the recommender locally if API is not running."""
    from movieinsight.config import ARTIFACTS_DIR
    from movieinsight.data_loader import load_raw_data
    from movieinsight.preprocessing import preprocess_data
    from movieinsight.recommender import MovieRecommender

    processed_csv = ARTIFACTS_DIR / "processed_movies.csv"
    if processed_csv.exists() and processed_csv.stat().st_size > 0:
        import ast

        df = pd.read_csv(processed_csv)
        for col in ["genres", "keywords", "cast"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
                )
    else:
        raw_df = load_raw_data()
        df = preprocess_data(raw_df)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_csv, index=False)

    rec = MovieRecommender()
    rec.fit(df)
    return rec


# --- UI HEADER ---
st.title("🎬 MovieInsight: Production-Grade Movie Recommender")
st.markdown(
    """
    **MovieInsight** transforms raw movie metadata into intelligent, content-based recommendations 
    powered by **TF-IDF vectorization**, **cosine similarity**, and **feature-level explainability**.
    """
)

# Sidebar settings
st.sidebar.header("⚙️ Recommendation Settings")
top_n = st.sidebar.slider("Number of Recommendations (Top-N)", min_value=1, max_value=20, value=6)

# Load recommender
try:
    recommender = get_local_recommender()
    all_genres = ["All"] + recommender.all_genres
except Exception as exc:
    st.error(f"Error loading movie catalog: {exc}")
    st.stop()

selected_genre = st.sidebar.selectbox("Filter by Genre", options=all_genres)
genre_filter = None if selected_genre == "All" else selected_genre

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Explainability Engine:**\nEvery recommendation highlights shared directors, actors, genres, and thematic keywords."
)

# Search Bar
search_col, button_col = st.columns([4, 1])
with search_col:
    query = st.text_input("🔍 Search Movie by Title", placeholder="e.g. Inception, The Dark Knight, Avatar...")

if query:
    matches = recommender.search_movies(query, limit=10)
    if not matches:
        st.warning(f"No movies found matching '{query}'. Try another search.")
    else:
        movie_options = {f"{m['title']} ({m['release_date'][:4] if m['release_date'] else 'N/A'})": m["id"] for m in matches}
        selected_display = st.selectbox("Select Movie from Results:", list(movie_options.keys()))
        selected_id = movie_options[selected_display]

        target_movie = recommender.get_movie_by_id(selected_id)

        # Display Selected Movie Info
        with st.expander("📌 Selected Movie Details", expanded=True):
            col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
            with col_info1:
                st.subheader(target_movie["title"])
                st.write(target_movie["overview"] or "No overview available.")
                st.write(f"**Genres:** {', '.join(target_movie['genres']) if target_movie['genres'] else 'N/A'}")
            with col_info2:
                st.metric("Rating", f"{target_movie['vote_average']} ⭐", f"{target_movie['vote_count']} votes")
                st.write(f"**Director:** {target_movie['director'] or 'N/A'}")
                st.write(f"**Top Cast:** {', '.join(target_movie['cast'][:3]) if target_movie['cast'] else 'N/A'}")
            with col_info3:
                st.write(f"**Budget:** ${target_movie['budget']:,}")
                st.write(f"**Revenue:** ${target_movie['revenue']:,}")
                st.write(f"**ROI:** {target_movie['roi']}x" if target_movie['roi'] is not None else "**ROI:** N/A")
                st.write(f"**Success Status:** `{target_movie['success']}`")

        # Get Recommendations
        st.markdown("### 🎯 Recommended Movies")
        recs = recommender.recommend_by_id(
            movie_id=selected_id,
            top_n=top_n,
            genre_filter=genre_filter,
        )

        if not recs:
            st.info("No recommendations found matching the criteria/filter.")
        else:
            # Display recommendations in a responsive 3-column grid
            cols = st.columns(3)
            for idx, rec in enumerate(recs):
                col = cols[idx % 3]
                with col:
                    with st.container(border=True):
                        st.subheader(f"{idx+1}. {rec['title']}")
                        score_pct = int(rec["similarity_score"] * 100)
                        st.progress(score_pct / 100, text=f"Match Score: {score_pct}% ({rec['similarity_score']})")

                        st.write(f"**Genres:** {', '.join(rec['genres'][:3])}")
                        if rec["director"]:
                            st.write(f"**Director:** {rec['director']}")
                        if rec["cast"]:
                            st.write(f"**Cast:** {', '.join(rec['cast'][:2])}")
                        st.write(f"**Rating:** {rec['vote_average']} ⭐")

                        # Explainability block
                        st.info(f"**Why recommended:**\n{rec['explanation']['summary']}")
else:
    st.info("👆 Enter a movie title in the search box above to get started!")
    st.markdown("#### Popular catalog seed suggestions:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Search 'Avatar'", on_click=lambda: None)
    with col2:
        st.button("Search 'The Dark Knight'", on_click=lambda: None)
    with col3:
        st.button("Search 'Interstellar'", on_click=lambda: None)
