"""Data preprocessing, cleaning, feature engineering, and metadata soup creation."""

import ast
import json
import logging
import re
from typing import Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_parse_json_list(val: Any, key_name: str = "name", max_items: Optional[int] = None) -> List[str]:
    """Safely parse JSON or Python literal representations of dictionaries into a list of strings.

    Args:
        val: Input value (string representation or already parsed list).
        key_name: Dictionary key to extract.
        max_items: Optional maximum number of items to retain.

    Returns:
        List[str]: Extracted strings, stripped of excess whitespace.
    """
    if val is None:
        return []
    if isinstance(val, (float, int)) and pd.isna(val):
        return []

    parsed = None
    if isinstance(val, list):
        parsed = val
    elif isinstance(val, str):
        val_clean = val.strip()
        if not val_clean:
            return []
        try:
            parsed = ast.literal_eval(val_clean)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(val_clean)
            except Exception:
                return []
    else:
        return []

    if not isinstance(parsed, list):
        return []

    results = []
    for item in parsed:
        if isinstance(item, dict) and key_name in item:
            item_val = item[key_name]
            if item_val and isinstance(item_val, str):
                results.append(item_val.strip())
        elif isinstance(item, str):
            results.append(item.strip())

    if max_items is not None and max_items > 0:
        return results[:max_items]
    return results


def extract_director(crew_val: Any) -> Optional[str]:
    """Extract director name from crew JSON/literal.

    Args:
        crew_val: Crew field containing list of crew members.

    Returns:
        Optional[str]: Director's name if found, else None.
    """
    if crew_val is None:
        return None
    if isinstance(crew_val, (float, int)) and pd.isna(crew_val):
        return None

    parsed = None
    if isinstance(crew_val, list):
        parsed = crew_val
    elif isinstance(crew_val, str):
        val_clean = crew_val.strip()
        if not val_clean:
            return None
        try:
            parsed = ast.literal_eval(val_clean)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(val_clean)
            except Exception:
                return None
    else:
        return None

    if not isinstance(parsed, list):
        return None

    for person in parsed:
        if isinstance(person, dict) and person.get("job") == "Director":
            name = person.get("name")
            if name and isinstance(name, str):
                return name.strip()
    return None


def calculate_financials(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate profit, ROI, and categorize movie success.

    Preserves logic from original exploratory analysis while handling edge cases
    such as zero budget, missing values, and infinite ROI.

    Args:
        df: Input DataFrame with 'budget' and 'revenue' columns.

    Returns:
        pd.DataFrame: DataFrame with 'profit', 'roi', and 'success' columns.
    """
    df = df.copy()
    budget = pd.to_numeric(df["budget"], errors="coerce").fillna(0)
    revenue = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)

    profit = revenue - budget
    df["profit"] = profit

    # Calculate ROI safely: only when budget > 0
    valid_budget = budget > 0
    roi = pd.Series(np.nan, index=df.index, dtype=float)
    roi.loc[valid_budget] = profit.loc[valid_budget] / budget.loc[valid_budget]
    df["roi"] = roi

    # Categorize movie success
    def categorize(row_roi: float, has_valid_budget: bool) -> str:
        if not has_valid_budget or pd.isna(row_roi):
            return "Unknown"
        if row_roi > 1.0:
            return "Hit"
        if row_roi >= 0.0:
            return "Average"
        return "Flop"

    df["success"] = [
        categorize(r, v) for r, v in zip(df["roi"], valid_budget)
    ]
    return df


def _clean_token(name: str) -> str:
    """Transform entity names into single normalized tokens (e.g. 'Tom Hanks' -> 'tom_hanks')."""
    return re.sub(r"\s+", "_", name.strip().lower())


def create_metadata_soup(
    overview: str,
    genres: List[str],
    keywords: List[str],
    cast: List[str],
    director: Optional[str],
) -> str:
    """Create a single combined representation text ("soup") for TF-IDF feature extraction.

    Multi-word entity names (genres, cast, director) are joined with underscores to preserve
    entity integrity (e.g., 'Christopher Nolan' becomes 'christopher_nolan').
    Director and top cast are weighted by repeating tokens to ensure appropriate similarity emphasis.

    Args:
        overview: Plot summary.
        genres: List of movie genres.
        keywords: List of movie plot keywords.
        cast: Top actors.
        director: Movie director name.

    Returns:
        str: Clean combined textual representation.
    """
    soup_parts: List[str] = []

    # Overview normalized words
    if overview and isinstance(overview, str) and overview.strip():
        soup_parts.append(overview.strip().lower())

    # Normalized genres
    if genres:
        soup_parts.extend([_clean_token(g) for g in genres])

    # Normalized keywords
    if keywords:
        soup_parts.extend([_clean_token(k) for k in keywords])

    # Normalized director (repeated 2x for importance weighting)
    if director and isinstance(director, str) and director.strip():
        d_token = _clean_token(director)
        soup_parts.extend([d_token, d_token])

    # Normalized cast top 3 (repeated 2x for importance weighting)
    if cast:
        cast_tokens = [_clean_token(a) for a in cast[:3]]
        soup_parts.extend(cast_tokens)
        soup_parts.extend(cast_tokens)

    return " ".join(soup_parts)


def preprocess_data(raw_df: pd.DataFrame, max_cast: int = 3) -> pd.DataFrame:
    """Run end-to-end preprocessing pipeline on merged raw movie data.

    Args:
        raw_df: Raw merged DataFrame.
        max_cast: Number of top cast members to extract.

    Returns:
        pd.DataFrame: Fully cleaned and engineered DataFrame ready for recommendation.
    """
    logger.info("Starting preprocessing of %d records", len(raw_df))
    df = raw_df.copy()

    # Clean IDs and basic metadata
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype(int)
    df["title"] = df["title"].fillna("Untitled").astype(str).str.strip()
    df["overview"] = df["overview"].fillna("").astype(str)
    df["vote_average"] = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df.get("vote_count", 0), errors="coerce").fillna(0).astype(int)
    df["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0.0)
    df["release_date"] = df.get("release_date", "").fillna("").astype(str)

    # Extract JSON fields
    logger.info("Extracting genres, keywords, cast, and director")
    df["genres_clean"] = df["genres"].apply(lambda x: safe_parse_json_list(x, key_name="name"))
    df["keywords_clean"] = df.get("keywords", pd.Series()).apply(lambda x: safe_parse_json_list(x, key_name="name"))
    df["cast_clean"] = df.get("cast", pd.Series()).apply(lambda x: safe_parse_json_list(x, key_name="name", max_items=max_cast))
    df["director"] = df.get("crew", pd.Series()).apply(extract_director)

    # Calculate financial metrics & success category
    df = calculate_financials(df)

    # Generate metadata soup for recommender
    logger.info("Generating metadata soup for TF-IDF vectorization")
    soups = []
    for _, row in df.iterrows():
        soup = create_metadata_soup(
            overview=row["overview"],
            genres=row["genres_clean"],
            keywords=row["keywords_clean"],
            cast=row["cast_clean"],
            director=row["director"],
        )
        soups.append(soup)
    df["soup"] = soups

    # Select and organize output columns
    keep_cols = [
        "id",
        "title",
        "overview",
        "genres_clean",
        "keywords_clean",
        "director",
        "cast_clean",
        "budget",
        "revenue",
        "profit",
        "roi",
        "success",
        "vote_average",
        "vote_count",
        "popularity",
        "release_date",
        "soup",
    ]
    result_df = df[keep_cols].rename(
        columns={
            "genres_clean": "genres",
            "keywords_clean": "keywords",
            "cast_clean": "cast",
        }
    ).reset_index(drop=True)

    logger.info("Preprocessing complete. Output contains %d records.", len(result_df))
    return result_df
