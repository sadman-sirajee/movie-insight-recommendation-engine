"""Data loading and schema validation module for MovieInsight."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from movieinsight.config import RAW_CREDITS_PATH, RAW_MOVIES_PATH

logger = logging.getLogger(__name__)

# Expected minimum columns in raw datasets
REQUIRED_MOVIES_COLUMNS = [
    "id",
    "title",
    "genres",
    "overview",
    "budget",
    "revenue",
    "popularity",
    "vote_average",
    "vote_count",
]

REQUIRED_CREDITS_COLUMNS = [
    "movie_id",
    "title",
    "cast",
    "crew",
]


class DataLoadingError(Exception):
    """Raised when dataset loading or validation fails."""


def validate_raw_files(movies_path: Path, credits_path: Path) -> None:
    """Validate that the required dataset files exist and are non-empty.

    Args:
        movies_path: Path to movies.csv.
        credits_path: Path to movies_credits.csv.

    Raises:
        DataLoadingError: If any required file is missing or empty.
    """
    for file_path, name in [(movies_path, "movies dataset"), (credits_path, "credits dataset")]:
        if not file_path.exists():
            raise DataLoadingError(
                f"Missing {name} at '{file_path}'. "
                "Please place 'movies.csv' and 'movies_credits.csv' in the data directory. "
                "You can obtain them from Kaggle's TMDB 5000 Movie Dataset."
            )
        if file_path.stat().st_size == 0:
            raise DataLoadingError(f"The {name} at '{file_path}' is empty.")


def load_raw_data(
    movies_path: Optional[Path] = None,
    credits_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load raw movies and credits CSVs and merge them into a single DataFrame.

    Args:
        movies_path: Optional custom path to movies.csv.
        credits_path: Optional custom path to movies_credits.csv.

    Returns:
        pd.DataFrame: Merged raw dataset with unified 'id' and 'title'.

    Raises:
        DataLoadingError: If files cannot be read or merged.
    """
    m_path = Path(movies_path) if movies_path else RAW_MOVIES_PATH
    c_path = Path(credits_path) if credits_path else RAW_CREDITS_PATH

    logger.info("Validating raw dataset paths: %s and %s", m_path, c_path)
    validate_raw_files(m_path, c_path)

    try:
        logger.info("Loading movies from %s", m_path)
        movies_df = pd.read_csv(m_path)
    except Exception as exc:
        raise DataLoadingError(f"Failed to read movies file '{m_path}': {exc}") from exc

    try:
        logger.info("Loading credits from %s", c_path)
        credits_df = pd.read_csv(c_path)
    except Exception as exc:
        raise DataLoadingError(f"Failed to read credits file '{c_path}': {exc}") from exc

    # Validate essential columns exist
    missing_movies = [c for c in REQUIRED_MOVIES_COLUMNS if c not in movies_df.columns]
    if missing_movies:
        raise DataLoadingError(f"Movies dataset missing required columns: {missing_movies}")

    missing_credits = [c for c in REQUIRED_CREDITS_COLUMNS if c not in credits_df.columns]
    if missing_credits:
        raise DataLoadingError(f"Credits dataset missing required columns: {missing_credits}")

    # Ensure id types match
    movies_df["id"] = pd.to_numeric(movies_df["id"], errors="coerce")
    credits_df["movie_id"] = pd.to_numeric(credits_df["movie_id"], errors="coerce")

    # Drop rows with invalid IDs
    movies_df = movies_df.dropna(subset=["id"]).copy()
    credits_df = credits_df.dropna(subset=["movie_id"]).copy()
    movies_df["id"] = movies_df["id"].astype(int)
    credits_df["movie_id"] = credits_df["movie_id"].astype(int)

    # Merge on movies.id == credits.movie_id
    logger.info("Merging movies (%d rows) and credits (%d rows)", len(movies_df), len(credits_df))
    merged_df = movies_df.merge(credits_df, left_on="id", right_on="movie_id", suffixes=("", "_credits"))

    # Resolve duplicate title columns cleanly
    if "title" not in merged_df.columns and "title_x" in merged_df.columns:
        merged_df["title"] = merged_df["title_x"]
    if "title_credits" in merged_df.columns:
        merged_df = merged_df.drop(columns=["title_credits"])
    if "title_y" in merged_df.columns:
        merged_df = merged_df.drop(columns=["title_y"])

    # Remove duplicate IDs if any exist
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    if len(merged_df) < initial_count:
        logger.warning("Dropped %d duplicate movie IDs during merge", initial_count - len(merged_df))

    logger.info("Successfully loaded and merged raw dataset: %d records", len(merged_df))
    return merged_df
