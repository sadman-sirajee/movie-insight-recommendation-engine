"""Content-based Movie Recommendation Engine with TF-IDF and Explainability."""

import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from movieinsight.config import DEFAULT_TOP_N, MAX_TOP_N, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_STOP_WORDS

logger = logging.getLogger(__name__)


class MovieNotFoundError(Exception):
    """Raised when a requested movie cannot be located by ID or Title."""


class MovieRecommender:
    """Production-grade content-based recommendation engine for movies.

    Uses a TF-IDF vectorizer over engineered metadata soups and computes cosine
    similarity on-the-fly via sparse matrix dot products. This avoids storing
    dense similarity matrices in memory, ensuring scalability and low latency.
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        stop_words: str = TFIDF_STOP_WORDS,
        ngram_range: tuple = TFIDF_NGRAM_RANGE,
    ):
        """Initialize the recommender configuration and vectorizer.

        Args:
            max_features: Maximum number of TF-IDF vocabulary features.
            stop_words: Stop words configuration (default 'english').
            ngram_range: Tuple of (min_n, max_n) n-gram range.
        """
        self.max_features = max_features
        self.stop_words = stop_words
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            ngram_range=self.ngram_range,
        )

        self.movies_df: Optional[pd.DataFrame] = None
        self.tfidf_matrix = None
        self.id_to_idx: Dict[int, int] = {}
        self.title_to_ids: Dict[str, List[int]] = {}
        self.all_genres: List[str] = []

    def is_fitted(self) -> bool:
        """Check if recommender model has been fitted with movie data."""
        return self.movies_df is not None and self.tfidf_matrix is not None

    def fit(self, movies_df: pd.DataFrame) -> "MovieRecommender":
        """Fit TF-IDF matrix and build lookup indices from cleaned movie data.

        Args:
            movies_df: Processed DataFrame containing 'id', 'title', 'soup', 'genres', etc.

        Returns:
            self
        """
        if movies_df.empty:
            raise ValueError("Cannot fit recommender on an empty DataFrame.")

        logger.info("Fitting recommender on %d movies", len(movies_df))
        self.movies_df = movies_df.reset_index(drop=True).copy()

        # Build fast lookup indexes
        self.id_to_idx = {
            int(row["id"]): idx for idx, row in self.movies_df.iterrows()
        }

        self.title_to_ids = {}
        genre_set: Set[str] = set()

        for idx, row in self.movies_df.iterrows():
            t_clean = str(row["title"]).strip().lower()
            m_id = int(row["id"])
            self.title_to_ids.setdefault(t_clean, []).append(m_id)

            # Collect unique genres
            g_list = row.get("genres", [])
            if isinstance(g_list, list):
                genre_set.update(g_list)

        self.all_genres = sorted(list(genre_set))

        # Vectorize metadata soups into sparse CSR matrix
        soups = self.movies_df["soup"].fillna("").astype(str)
        logger.info("Computing TF-IDF matrix (vocabulary limit: %d)", self.max_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(soups)
        logger.info("TF-IDF matrix built successfully with shape: %s", self.tfidf_matrix.shape)

        return self

    def get_movie_by_id(self, movie_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve movie record by its TMDB movie ID.

        Args:
            movie_id: Integer movie ID.

        Returns:
            Optional[Dict[str, Any]]: Movie metadata or None if not found.
        """
        if not self.is_fitted():
            raise RuntimeError("Recommender must be fitted before querying.")

        idx = self.id_to_idx.get(movie_id)
        if idx is None:
            return None

        row = self.movies_df.iloc[idx]
        return self._format_movie_dict(row)

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search movies by title using substring and case-insensitive matching.

        Args:
            query: User search string.
            limit: Maximum results to return.

        Returns:
            List[Dict[str, Any]]: Matching movies ordered by relevance/popularity.
        """
        if not self.is_fitted():
            raise RuntimeError("Recommender must be fitted before querying.")

        q = query.strip().lower()
        if not q:
            return []

        # Find matches: exact matches first, then prefix matches, then substring matches
        exact_matches = []
        prefix_matches = []
        substr_matches = []

        for idx, row in self.movies_df.iterrows():
            title_lower = str(row["title"]).lower()
            if title_lower == q:
                exact_matches.append(idx)
            elif title_lower.startswith(q):
                prefix_matches.append(idx)
            elif q in title_lower:
                substr_matches.append(idx)

        all_matched_indices = exact_matches + prefix_matches + substr_matches
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in all_matched_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        matches_df = self.movies_df.iloc[unique_indices]
        # Sort by popularity or vote_count within matched subset if there are many
        if len(matches_df) > limit:
            matches_df = matches_df.head(limit)

        return [self._format_movie_dict(row) for _, row in matches_df.iterrows()]

    def recommend_by_id(
        self,
        movie_id: int,
        top_n: int = DEFAULT_TOP_N,
        genre_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate top-N movie recommendations for a given movie ID.

        Computes cosine similarity on-the-fly against the sparse TF-IDF matrix.
        Calculates feature-level explainability (shared genres, director, cast, keywords).

        Args:
            movie_id: Target movie ID.
            top_n: Number of recommendations to return (1 to MAX_TOP_N).
            genre_filter: Optional genre string to filter recommendations.

        Returns:
            List[Dict[str, Any]]: Recommended movies with scores and explanations.

        Raises:
            MovieNotFoundError: If target movie ID does not exist in dataset.
        """
        if not self.is_fitted():
            raise RuntimeError("Recommender must be fitted before querying.")

        idx = self.id_to_idx.get(movie_id)
        if idx is None:
            raise MovieNotFoundError(f"Movie with ID {movie_id} not found in database.")

        top_n = max(1, min(top_n, MAX_TOP_N))

        # Query vector from sparse matrix
        query_vec = self.tfidf_matrix[idx]

        # Compute cosine similarities via sparse dot product: shape (N,)
        # Note: TF-IDF vectors from TfidfVectorizer are already L2-normalized,
        # so dot product equals cosine similarity.
        sim_scores = self.tfidf_matrix.dot(query_vec.T).toarray().ravel()

        # Mask self to prevent recommending the query movie itself
        sim_scores[idx] = -1.0

        target_row = self.movies_df.iloc[idx]
        target_genres = set(target_row.get("genres", []))
        target_director = target_row.get("director")
        target_cast = set(target_row.get("cast", []))
        target_keywords = set(target_row.get("keywords", []))

        # Filter by genre if requested
        if genre_filter:
            genre_clean = genre_filter.strip().lower()
            valid_indices = [
                i
                for i, r in self.movies_df.iterrows()
                if genre_clean in [g.lower() for g in r.get("genres", [])] and i != idx
            ]
            if not valid_indices:
                return []
            # Extract scores for valid subset
            valid_scores = sim_scores[valid_indices]
            top_sub_idx = np.argsort(valid_scores)[::-1][:top_n]
            candidate_indices = [valid_indices[i] for i in top_sub_idx]
        else:
            # Sort top indices descending
            candidate_indices = np.argsort(sim_scores)[::-1][:top_n]

        recommendations: List[Dict[str, Any]] = []

        for cand_idx in candidate_indices:
            score = float(sim_scores[cand_idx])
            if score <= 0.0:
                continue

            cand_row = self.movies_df.iloc[cand_idx]
            cand_genres = set(cand_row.get("genres", []))
            cand_director = cand_row.get("director")
            cand_cast = set(cand_row.get("cast", []))
            cand_keywords = set(cand_row.get("keywords", []))

            # Feature-level overlap explainability
            shared_genres = sorted(list(target_genres.intersection(cand_genres)))
            shared_director = (
                str(cand_director)
                if (
                    isinstance(target_director, str)
                    and isinstance(cand_director, str)
                    and target_director.strip()
                    and cand_director.strip()
                    and target_director.strip().lower() == cand_director.strip().lower()
                )
                else None
            )
            shared_cast = sorted(list(target_cast.intersection(cand_cast)))
            shared_keywords = sorted(list(target_keywords.intersection(cand_keywords)))

            explanation_text = self._build_explanation_summary(
                shared_genres=shared_genres,
                shared_director=shared_director,
                shared_cast=shared_cast,
                shared_keywords=shared_keywords,
            )

            rec_dict = self._format_movie_dict(cand_row)
            rec_dict["similarity_score"] = round(score, 4)
            rec_dict["explanation"] = {
                "summary": explanation_text,
                "shared_genres": shared_genres,
                "shared_director": shared_director,
                "shared_cast": shared_cast,
                "shared_keywords": shared_keywords[:5],
            }
            recommendations.append(rec_dict)

        return recommendations

    def recommend_by_title(
        self,
        title: str,
        top_n: int = DEFAULT_TOP_N,
        genre_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using movie title.

        Args:
            title: Movie title to search for.
            top_n: Number of recommendations.
            genre_filter: Optional genre filter.

        Returns:
            List[Dict[str, Any]]: List of recommended movies.

        Raises:
            MovieNotFoundError: If no movie matches the provided title.
        """
        matches = self.search_movies(title, limit=1)
        if not matches:
            raise MovieNotFoundError(f"No movie matching title '{title}' was found.")

        movie_id = matches[0]["id"]
        return self.recommend_by_id(movie_id, top_n=top_n, genre_filter=genre_filter)

    @staticmethod
    def _build_explanation_summary(
        shared_genres: List[str],
        shared_director: Optional[str],
        shared_cast: List[str],
        shared_keywords: List[str],
    ) -> str:
        """Build human-readable reason why a movie is recommended."""
        reasons = []
        if shared_genres:
            reasons.append(f"genre{'s' if len(shared_genres) > 1 else ''} ({', '.join(shared_genres[:2])})")
        if shared_director:
            reasons.append(f"director ({shared_director})")
        if shared_cast:
            reasons.append(f"cast ({', '.join(shared_cast[:2])})")
        if shared_keywords and not shared_director and not shared_cast:
            reasons.append(f"themes ({', '.join(shared_keywords[:2])})")

        if reasons:
            return "Recommended based on shared " + " and ".join(reasons) + "."
        return "Recommended based on thematic and plot similarity."

    @staticmethod
    def _format_movie_dict(row: pd.Series) -> Dict[str, Any]:
        """Convert a pandas Series into a clean, JSON-serializable dictionary."""
        roi_val = row.get("roi")
        roi_float = None if pd.isna(roi_val) else round(float(roi_val), 3)

        return {
            "id": int(row["id"]),
            "title": str(row["title"]),
            "overview": str(row.get("overview", "")),
            "genres": list(row.get("genres", [])),
            "director": (
                str(row.get("director")).strip()
                if (
                    pd.notna(row.get("director"))
                    and isinstance(row.get("director"), str)
                    and row.get("director").strip()
                )
                else None
            ),
            "cast": list(row.get("cast", [])),
            "release_date": str(row.get("release_date", "")) if pd.notna(row.get("release_date")) else "",
            "vote_average": round(float(row.get("vote_average", 0.0)), 2),
            "vote_count": int(row.get("vote_count", 0)),
            "popularity": round(float(row.get("popularity", 0.0)), 2),
            "budget": int(row.get("budget", 0)),
            "revenue": int(row.get("revenue", 0)),
            "profit": int(row.get("profit", 0)),
            "roi": roi_float,
            "success": str(row.get("success", "Unknown")),
        }
